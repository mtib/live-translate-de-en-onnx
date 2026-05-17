import Foundation
import AVFoundation
import Accelerate
import CSherpaOnnx

/// sherpa-onnx streaming RNN-T transcriber — replaces `WhisperCppTranscriber`.
///
/// **Pipeline per audio stream:**
///   1. Receive 48 kHz mono Float32 buffers (post-RNNoise).
///   2. Resample to 16 kHz (AVAudioConverter, persistent across buffers).
///   3. Push samples into a Silero-VAD instance for voiced/silence detection.
///   4. Push voiced samples into the sherpa-onnx streaming recognizer.
///   5. As ASR emits tokens, fire `.partial(text:)` so the UI shows live text.
///      The shown text is always the *active* portion of the hypothesis —
///      everything after the last mid-turn force-completion boundary.
///   6. Mid-turn sentence cap: once the active portion exceeds `maxCharsPerRow`
///      characters the accumulator watches for `. `, `? `, or `! ` boundaries.
///      For `.` the char before it must be a letter (skips decimals, versions).
///      When found it immediately fires `.completed` for the current chunk
///      (the text up to and including the period) and starts a fresh chunk for
///      the remainder. This keeps rows stable — no retroactive rewrites.
///   7. On endpoint (≥1 s trailing silence), only the *uncommitted* remainder
///      text is sent to the worker for VAD-gap splitting and final emission.
///
/// **Crosstalk suppression:** `markSystemVoiced` / `isSystemRecentlyVoiced`
/// — mic buffers are zeroed when system audio was recently heard.
final class SherpaTranscriber: Transcriber {

    // MARK: — Tunables

    /// Trailing-silence endpoint threshold fed to sherpa-onnx rule 1.
    static let endpointSilenceSeconds: Float = 1.0

    /// RMS used for the crosstalk gate.
    static let silenceRMSThreshold: Float = 0.012

    /// How long after system-voiced we keep treating mic as contaminated.
    static let crosstalkPersistSeconds: TimeInterval = 0.25

    /// Silence gap between two voiced segments that triggers a row split.
    /// 0.8 s sits just under the 1.0 s ASR endpoint threshold so only
    /// genuine between-utterance pauses split rows; shorter breath/clause
    /// pauses are left intact.
    private static let vadSplitGapSamples: Int = Int(0.8 * 16_000)

    /// Minimum words each split group must receive. If the available words
    /// can't satisfy this for every group, the split is skipped and the
    /// whole text is emitted as one row to avoid single-word burst rows.
    private static let minWordsPerVadGroup = 4

    /// Once the active partial exceeds this many characters the accumulator
    /// looks for a sentence-ending punctuation boundary (`. `, `? `, `! `)
    /// to force-complete the row so translation and TTS can start promptly.
    static let maxCharsPerRow = 30

    // MARK: — Shared recognizer (loaded once, reused)

    private var recognizer: OpaquePointer?
    private let recognizerLock = NSLock()
    private var loadError: Error?

    // MARK: — Lifecycle callback

    var onChunkLifecycle: (@Sendable (_ chunkID: UUID, _ source: SourceTag,
                                       _ event: ChunkLifecycle) -> Void)?

    enum ChunkLifecycle: Sendable {
        case listening
        /// Live partial ASR hypothesis — fires whenever the token stream
        /// changes during an open turn, so the UI can show rolling text
        /// instead of "transcribing".
        case partial(text: String)
        case completed(text: String, startSeconds: Double?, endSeconds: Double?)
        case dropped
    }

    // MARK: — Crosstalk state

    private var lastSystemVoicedAt: Date = .distantPast
    private let crosstalkLock = NSLock()

    func markSystemVoiced() {
        crosstalkLock.withLock { lastSystemVoicedAt = Date() }
    }

    func isSystemRecentlyVoiced() -> Bool {
        let now = Date()
        return crosstalkLock.withLock {
            now.timeIntervalSince(lastSystemVoicedAt) < Self.crosstalkPersistSeconds
        }
    }

    // MARK: — Model loading

    private func ensureRecognizerLoaded() throws -> OpaquePointer {
        recognizerLock.lock()
        defer { recognizerLock.unlock() }
        if let r = recognizer { return r }
        if let e = loadError   { throw e }

        let enc = resourcePath(ModelConfig.asrEncoder)
        let dec = resourcePath(ModelConfig.asrDecoder)
        let joi = resourcePath(ModelConfig.asrJoiner)
        let tok = resourcePath(ModelConfig.asrTokens)

        var cfg = SherpaOnnxOnlineRecognizerConfig()
        memset_zero(&cfg)

        cfg.feat_config.sample_rate = 16_000
        cfg.feat_config.feature_dim = 80
        cfg.enable_endpoint = 1
        cfg.rule1_min_trailing_silence = Self.endpointSilenceSeconds
        cfg.rule2_min_trailing_silence = 2.4
        cfg.rule3_min_utterance_length = 20.0
        cfg.model_config.num_threads = Int32(max(2, ProcessInfo.processInfo.activeProcessorCount - 2))
        cfg.model_config.debug = 0

        let method   = "greedy_search"
        let provider = ModelConfig.provider

        guard let r = method.withCString({ meth in
            cfg.decoding_method = meth
            return provider.withCString { prov in
                cfg.model_config.provider = prov
                return enc.withCString { encP in
                    cfg.model_config.transducer.encoder = encP
                    return dec.withCString { decP in
                        cfg.model_config.transducer.decoder = decP
                        return joi.withCString { joiP in
                            cfg.model_config.transducer.joiner = joiP
                            return tok.withCString { tokP in
                                cfg.model_config.tokens = tokP
                                return SherpaOnnxCreateOnlineRecognizer(&cfg)
                            }
                        }
                    }
                }
            }
        }) else {
            let err = TranscribeError.unavailable(
                "SherpaTranscriber: SherpaOnnxCreateOnlineRecognizer failed — check model paths")
            loadError = err
            throw err
        }

        recognizer = r
        Log.line("SherpaTranscriber: recognizer loaded (provider=\(provider))")
        return r
    }

    deinit {
        if let r = recognizer { SherpaOnnxDestroyOnlineRecognizer(r) }
    }

    // MARK: — Transcriber protocol

    func transcribe(
        audio: AsyncStream<AVAudioPCMBuffer>,
        locale: SourceLocale,
        source: SourceTag
    ) -> AsyncThrowingStream<SessionSnapshot, Error> {
        AsyncThrowingStream { continuation in
            let runner = Task {
                do {
                    let recog = try self.ensureRecognizerLoaded()
                    Log.line("SherpaTranscriber[\(source.rawValue)]: starting")
                    try await self.runChunkLoop(
                        recognizer: recog, audio: audio,
                        source: source, continuation: continuation
                    )
                    Log.line("SherpaTranscriber[\(source.rawValue)]: audio ended")
                    continuation.finish()
                } catch {
                    Log.line("SherpaTranscriber[\(source.rawValue)]: \(error)")
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in runner.cancel() }
        }
    }

    // MARK: — Internal data models

    /// A contiguous voiced interval bounded by VAD-detected silence.
    /// The accumulator records one per voiced run within a turn so that
    /// `processTurn` can split the endpoint text at silence gaps ≥ the
    /// `vadSplitGapSamples` threshold.
    private struct VoicedSegment: Sendable {
        let startSample: Int   // cumulative 16 kHz offset at segment onset
        let endSample: Int     // cumulative 16 kHz offset at segment end
        let sampleCount: Int   // voiced samples in this segment (for proportion)
    }

    private struct TurnRecord: Sendable {
        let chunkID: UUID
        let index: Int
        let text: String
        let voicedSegments: [VoicedSegment]
        let startSample: Int
        let endSample: Int
    }

    // MARK: — Chunk loop

    private func runChunkLoop(
        recognizer: OpaquePointer,
        audio: AsyncStream<AVAudioPCMBuffer>,
        source: SourceTag,
        continuation: AsyncThrowingStream<SessionSnapshot, Error>.Continuation
    ) async throws {
        let (queue, queueSink) = AsyncStream<TurnRecord>.makeStream()
        let recognizerBits = UInt(bitPattern: recognizer)

        async let accumulate: Void = {
            let recog = OpaquePointer(bitPattern: recognizerBits)!
            await self.accumulateTurns(
                recognizer: recog, audio: audio,
                source: source, sink: queueSink
            )
            Log.line("SherpaTranscriber[\(source.rawValue)]: accumulator done")
            queueSink.finish()
        }()

        async let process: Void = {
            for await turn in queue {
                if Task.isCancelled { return }
                self.processTurn(turn: turn, source: source, continuation: continuation)
            }
            Log.line("SherpaTranscriber[\(source.rawValue)]: worker done")
        }()

        _ = await (accumulate, process)
    }

    // MARK: — Accumulator

    private func accumulateTurns(
        recognizer: OpaquePointer,
        audio: AsyncStream<AVAudioPCMBuffer>,
        source: SourceTag,
        sink: AsyncStream<TurnRecord>.Continuation
    ) async {
        let tag = source.rawValue
        var resampler = SherpaResampler()
        let vad = makeVAD()
        defer { if let v = vad { SherpaOnnxDestroyVoiceActivityDetector(v) } }

        guard let stream = SherpaOnnxCreateOnlineStream(recognizer) else {
            Log.line("SherpaTranscriber[\(tag)]: failed to create ASR stream")
            return
        }
        defer { SherpaOnnxDestroyOnlineStream(stream) }

        // Turn-level bookkeeping.
        var turnStartSample = 0
        var cumulativeSamples = 0
        var chunkIndex = 0
        var currentChunkID: UUID? = nil
        var hadVoice = false

        // Voiced-segment tracking (for VAD-gap splitting at endpoint).
        var voicedSegments: [VoicedSegment] = []
        var currentSegStart = 0
        var currentSegSampleCount = 0
        var prevVoiced = false

        // Partial-text dedup and mid-turn force-completion state.
        var lastPartialText = ""
        /// How many characters of the full ASR hypothesis have already been
        /// force-completed into earlier rows this turn. The "active" portion
        /// of any new hypothesis is `hypothesis[committedLength...]`.
        var committedLength = 0

        Log.line("SherpaTranscriber[\(tag)]: accumulator started")

        for await buf in audio {
            if Task.isCancelled { break }
            guard let samples16k = resampler.convert(buf), !samples16k.isEmpty else { continue }
            let n = samples16k.count

            // Crosstalk gate for mic.
            var effective = samples16k
            if source == .mic && isSystemRecentlyVoiced() {
                effective = [Float](repeating: 0, count: n)
            }

            if source == .system {
                if computeRMS(effective) >= Self.silenceRMSThreshold { markSystemVoiced() }
            }

            let samplesBefore = cumulativeSamples
            cumulativeSamples += n

            effective.withUnsafeBufferPointer { ptr in
                if let v = vad {
                    SherpaOnnxVoiceActivityDetectorAcceptWaveform(v, ptr.baseAddress, Int32(n))
                }
                SherpaOnnxOnlineStreamAcceptWaveform(stream, 16_000, ptr.baseAddress, Int32(n))
            }

            let voiced: Bool
            if let v = vad {
                voiced = SherpaOnnxVoiceActivityDetectorDetected(v) != 0
                SherpaOnnxVoiceActivityDetectorClear(v)
            } else {
                voiced = computeRMS(effective) >= Self.silenceRMSThreshold
            }

            if voiced {
                if !hadVoice {
                    let id = UUID()
                    currentChunkID = id
                    turnStartSample = samplesBefore
                    chunkIndex += 1
                    Log.line("SherpaTranscriber[\(tag)]: voice onset #\(chunkIndex) (id=\(id.uuidString.prefix(8))) at \(String(format:"%.2f",Double(samplesBefore)/16_000))s")
                    onChunkLifecycle?(id, source, .listening)
                    hadVoice = true
                }
                if !prevVoiced {
                    // Voiced-segment onset.
                    currentSegStart = samplesBefore
                    currentSegSampleCount = 0
                }
                currentSegSampleCount += n
            } else if prevVoiced && currentSegSampleCount > 0 {
                // Voiced → silent: close the current segment.
                voicedSegments.append(VoicedSegment(
                    startSample: currentSegStart,
                    endSample: samplesBefore,
                    sampleCount: currentSegSampleCount
                ))
                currentSegSampleCount = 0
            }

            prevVoiced = voiced

            while SherpaOnnxIsOnlineStreamReady(recognizer, stream) != 0 {
                SherpaOnnxDecodeOnlineStream(recognizer, stream)
            }

            // Emit partial text whenever the hypothesis changes, and
            // force-complete the current row when it passes maxWordsPerRow
            // and a sentence boundary (`. ` after a letter) is found.
            if hadVoice, let id = currentChunkID,
               let result = SherpaOnnxGetOnlineStreamResult(recognizer, stream) {
                let hypothesis = String(cString: result.pointee.text)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                SherpaOnnxDestroyOnlineRecognizerResult(result)

                if !hypothesis.isEmpty && hypothesis != lastPartialText {
                    lastPartialText = hypothesis
                    // Active portion = everything not yet force-completed.
                    let clampedCommit = min(committedLength, hypothesis.count)
                    let active = clampedCommit == 0
                        ? hypothesis
                        : String(hypothesis.dropFirst(clampedCommit))

                    if active.count > Self.maxCharsPerRow,
                       let dotIdx = sentenceBoundary(in: active) {
                        // Force-complete: text up to and including ".".
                        let afterDot   = active.index(after: dotIdx) // index of " "
                        let completed  = String(active[...dotIdx])   // ends with "."
                        let remStart   = active.index(after: afterDot) // skip " "
                        let remainder  = remStart < active.endIndex
                            ? String(active[remStart...]) : ""

                        let startSec = Double(turnStartSample) / 16_000
                        let endSec   = Double(cumulativeSamples) / 16_000
                        onChunkLifecycle?(id, source, .completed(
                            text: completed,
                            startSeconds: startSec, endSeconds: endSec
                        ))

                        // committedLength now covers completed + the space.
                        committedLength = clampedCommit + completed.count + 1

                        // Open a new inflight row for the remainder.
                        let newID = UUID()
                        currentChunkID = newID
                        onChunkLifecycle?(newID, source, .listening)
                        if !remainder.isEmpty {
                            onChunkLifecycle?(newID, source, .partial(text: remainder))
                        }
                    } else {
                        // Normal partial update — show the active portion.
                        onChunkLifecycle?(id, source, .partial(text: active))
                    }
                }
            }

            // Endpoint check.
            if hadVoice && SherpaOnnxOnlineStreamIsEndpoint(recognizer, stream) != 0 {
                // Close any open segment (silence caused the endpoint,
                // so prevVoiced is likely false; guard for the edge case).
                if currentSegSampleCount > 0 {
                    voicedSegments.append(VoicedSegment(
                        startSample: currentSegStart,
                        endSample: cumulativeSamples,
                        sampleCount: currentSegSampleCount
                    ))
                    currentSegSampleCount = 0
                }

                // *** Read text BEFORE reset ***
                let fullText: String
                if let result = SherpaOnnxGetOnlineStreamResult(recognizer, stream) {
                    fullText = String(cString: result.pointee.text)
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    SherpaOnnxDestroyOnlineRecognizerResult(result)
                } else {
                    fullText = ""
                }
                // Only send the uncommitted remainder to the worker.
                let remainder = committedLength == 0 ? fullText
                    : String(fullText.dropFirst(min(committedLength, fullText.count)))
                        .trimmingCharacters(in: .whitespacesAndNewlines)

                Log.line("SherpaTranscriber[\(tag)]: endpoint chunk #\(chunkIndex) remainder=\"\(remainder.prefix(60))\"")

                // If the ASR revised the final hypothesis to append trailing
                // punctuation (e.g. "." or "?") after a mid-turn force-complete,
                // the remainder is purely punctuation — drop it rather than emit
                // a standalone "." / "?" row.
                let remainderHasContent = remainder.rangeOfCharacter(from: .alphanumerics) != nil
                if (remainder.isEmpty || !remainderHasContent) && committedLength > 0 {
                    if !remainder.isEmpty {
                        Log.line("SherpaTranscriber[\(tag)]: chunk #\(chunkIndex) punctuation-only tail \"\(remainder)\" → dropped")
                    }
                    onChunkLifecycle?(currentChunkID ?? UUID(), source, .dropped)
                } else {
                    sink.yield(TurnRecord(
                        chunkID: currentChunkID ?? UUID(),
                        index: chunkIndex,
                        text: remainder.isEmpty ? fullText : remainder,
                        voicedSegments: voicedSegments,
                        startSample: turnStartSample,
                        endSample: cumulativeSamples
                    ))
                }

                SherpaOnnxOnlineStreamReset(recognizer, stream)
                voicedSegments.removeAll(keepingCapacity: true)
                committedLength = 0
                turnStartSample = cumulativeSamples
                currentChunkID = nil
                hadVoice = false
                prevVoiced = false
                lastPartialText = ""
            }
        }

        // Flush in-flight turn on stream end.
        if hadVoice {
            if currentSegSampleCount > 0 {
                voicedSegments.append(VoicedSegment(
                    startSample: currentSegStart,
                    endSample: cumulativeSamples,
                    sampleCount: currentSegSampleCount
                ))
            }
            SherpaOnnxOnlineStreamInputFinished(stream)
            while SherpaOnnxIsOnlineStreamReady(recognizer, stream) != 0 {
                SherpaOnnxDecodeOnlineStream(recognizer, stream)
            }
            let fullText: String
            if let result = SherpaOnnxGetOnlineStreamResult(recognizer, stream) {
                fullText = String(cString: result.pointee.text)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                SherpaOnnxDestroyOnlineRecognizerResult(result)
            } else {
                fullText = ""
            }
            let remainder = committedLength == 0 ? fullText
                : String(fullText.dropFirst(min(committedLength, fullText.count)))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            let flushHasContent = remainder.rangeOfCharacter(from: .alphanumerics) != nil
            if (remainder.isEmpty || !flushHasContent) && committedLength > 0 {
                if !remainder.isEmpty {
                    Log.line("SherpaTranscriber[\(tag)]: flush chunk #\(chunkIndex) punctuation-only tail \"\(remainder)\" → dropped")
                }
                onChunkLifecycle?(currentChunkID ?? UUID(), source, .dropped)
            } else {
                sink.yield(TurnRecord(
                    chunkID: currentChunkID ?? UUID(),
                    index: chunkIndex,
                    text: remainder.isEmpty ? fullText : remainder,
                    voicedSegments: voicedSegments,
                    startSample: turnStartSample,
                    endSample: cumulativeSamples
                ))
            }
        }

        Log.line("SherpaTranscriber[\(tag)]: accumulator exited (chunks=\(chunkIndex))")
    }

    // MARK: — Worker

    /// Splits the turn at VAD silence gaps ≥ `vadSplitGapSamples` and emits
    /// one row per group. Sentence-boundary capping is handled mid-turn by
    /// the accumulator (force-completions), so the remainder text arriving
    /// here is already reasonably short.
    private func processTurn(
        turn: TurnRecord,
        source: SourceTag,
        continuation: AsyncThrowingStream<SessionSnapshot, Error>.Continuation
    ) {
        guard !turn.text.isEmpty else {
            Log.line("SherpaTranscriber[\(source.rawValue)]: chunk #\(turn.index) empty → dropped")
            onChunkLifecycle?(turn.chunkID, source, .dropped)
            return
        }
        guard turn.text.rangeOfCharacter(from: .alphanumerics) != nil else {
            Log.line("SherpaTranscriber[\(source.rawValue)]: chunk #\(turn.index) punctuation-only \"\(turn.text)\" → dropped")
            onChunkLifecycle?(turn.chunkID, source, .dropped)
            return
        }

        // Group voiced segments: merge when gap < threshold, split when ≥.
        struct SegGroup {
            let startSample: Int
            var endSample: Int
            var sampleCount: Int
        }
        var groups: [SegGroup] = []
        for (i, seg) in turn.voicedSegments.enumerated() {
            if !groups.isEmpty {
                let gap = seg.startSample - turn.voicedSegments[i - 1].endSample
                if gap >= Self.vadSplitGapSamples {
                    groups.append(SegGroup(startSample: seg.startSample,
                                           endSample: seg.endSample,
                                           sampleCount: seg.sampleCount))
                    continue
                }
            } else {
                groups.append(SegGroup(startSample: seg.startSample,
                                       endSample: seg.endSample,
                                       sampleCount: seg.sampleCount))
                continue
            }
            groups[groups.count - 1].endSample = seg.endSample
            groups[groups.count - 1].sampleCount += seg.sampleCount
        }
        if groups.isEmpty {
            groups = [SegGroup(startSample: turn.startSample,
                               endSample: turn.endSample,
                               sampleCount: 1)]
        }

        // Fast path: single group.
        if groups.count == 1 {
            Log.line("SherpaTranscriber[\(source.rawValue)]: chunk #\(turn.index) → \"\(turn.text.prefix(60))\"")
            emit(chunkID: turn.chunkID, source: source, text: turn.text,
                 startSeconds: Double(turn.startSample) / 16_000,
                 endSeconds:   Double(turn.endSample)   / 16_000,
                 continuation: continuation)
            return
        }

        // Multiple groups: split text proportionally by voiced sample count.
        let words = turn.text.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        guard !words.isEmpty else {
            onChunkLifecycle?(turn.chunkID, source, .dropped)
            return
        }

        // Don't split if there aren't enough words to give every group a
        // meaningful share — that's what produces single-word burst rows.
        guard words.count >= groups.count * Self.minWordsPerVadGroup else {
            Log.line("SherpaTranscriber[\(source.rawValue)]: chunk #\(turn.index) \(groups.count) VAD groups but only \(words.count) words — emitting as one row")
            emit(chunkID: turn.chunkID, source: source, text: turn.text,
                 startSeconds: Double(turn.startSample) / 16_000,
                 endSeconds:   Double(turn.endSample)   / 16_000,
                 continuation: continuation)
            return
        }

        let totalSamples = groups.reduce(0) { $0 + $1.sampleCount }
        var wordOffset = 0
        var cumSamples = 0

        for (gi, group) in groups.enumerated() {
            cumSamples += group.sampleCount
            let isLast = gi == groups.count - 1

            let wordEnd: Int
            if isLast || wordOffset >= words.count {
                wordEnd = words.count
            } else {
                let proportion = Double(cumSamples) / Double(max(totalSamples, 1))
                let ideal = Int((proportion * Double(words.count)).rounded())
                let remaining = groups.count - gi - 1
                wordEnd = max(wordOffset + 1, min(ideal, words.count - remaining))
            }

            guard wordOffset < words.count else { break }
            let groupText = words[wordOffset..<wordEnd].joined(separator: " ")
            wordOffset = wordEnd
            guard !groupText.isEmpty else { continue }

            let chunkID = gi == 0 ? turn.chunkID : UUID()
            Log.line("SherpaTranscriber[\(source.rawValue)]: chunk #\(turn.index).\(gi+1) split → \"\(groupText.prefix(60))\"")
            emit(chunkID: chunkID, source: source, text: groupText,
                 startSeconds: Double(group.startSample) / 16_000,
                 endSeconds:   Double(group.endSample)   / 16_000,
                 continuation: continuation)
        }
    }

    private func emit(
        chunkID: UUID, source: SourceTag, text: String,
        startSeconds: Double, endSeconds: Double,
        continuation: AsyncThrowingStream<SessionSnapshot, Error>.Continuation
    ) {
        onChunkLifecycle?(chunkID, source, .completed(
            text: text, startSeconds: startSeconds, endSeconds: endSeconds
        ))
        continuation.yield(SessionSnapshot(sentences: [
            SessionSentence(text: text, isFinal: true,
                            startSeconds: startSeconds, endSeconds: endSeconds)
        ]))
    }

    /// Find the earliest sentence-ending boundary in `text` that is followed
    /// by a space, returning the index of the punctuation character itself.
    ///
    /// Recognised patterns:
    ///   - `. ` — only where the character before `.` is a letter, to skip
    ///            decimals ("3.14 "), version numbers ("v1.2 "), etc.
    ///   - `? ` — always (unambiguous sentence end)
    ///   - `! ` — always (unambiguous sentence end)
    ///
    /// Returns the earliest such index, or nil if none found.
    private func sentenceBoundary(in text: String) -> String.Index? {
        var result: String.Index? = nil

        // ". " — only after a letter character
        var search = text.startIndex
        while let range = text.range(of: ". ", range: search..<text.endIndex) {
            let dot = range.lowerBound
            if dot > text.startIndex && text[text.index(before: dot)].isLetter {
                result = dot
                break
            }
            search = text.index(after: range.lowerBound)
        }

        // "? " — unambiguous; keep if earlier than current best
        if let r = text.range(of: "? ") {
            if result == nil || r.lowerBound < result! { result = r.lowerBound }
        }

        // "! " — unambiguous; keep if earlier than current best
        if let r = text.range(of: "! ") {
            if result == nil || r.lowerBound < result! { result = r.lowerBound }
        }

        return result
    }

    // MARK: — Helpers

    private func makeVAD() -> OpaquePointer? {
        let vadPath = resourcePath(ModelConfig.vadModel)
        guard FileManager.default.fileExists(atPath: vadPath) else {
            Log.line("SherpaTranscriber: VAD model not in bundle, using RMS fallback")
            return nil
        }
        var silero = SherpaOnnxSileroVadModelConfig()
        memset_zero(&silero)
        silero.threshold            = 0.5
        silero.min_silence_duration = 0.25
        silero.min_speech_duration  = 0.05
        silero.window_size          = 512

        var vadCfg = SherpaOnnxVadModelConfig()
        memset_zero(&vadCfg)
        vadCfg.sample_rate = 16_000
        vadCfg.num_threads = 1
        vadCfg.debug       = 0

        let path     = vadPath
        let provider = ModelConfig.provider

        return path.withCString { cPath -> OpaquePointer? in
            silero.model      = cPath
            vadCfg.silero_vad = silero
            return provider.withCString { cProv -> OpaquePointer? in
                vadCfg.provider = cProv
                return SherpaOnnxCreateVoiceActivityDetector(&vadCfg, 30)
            }
        }
    }

    private func resourcePath(_ relativeName: String) -> String {
        Bundle.main.bundleURL
            .appendingPathComponent("Contents/Resources")
            .appendingPathComponent(relativeName)
            .path
    }

    private func computeRMS(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        var ms: Float = 0
        samples.withUnsafeBufferPointer { buf in
            vDSP_measqv(buf.baseAddress!, 1, &ms, vDSP_Length(samples.count))
        }
        return sqrt(ms)
    }
}

// MARK: — Errors

enum TranscribeError: LocalizedError {
    case unavailable(String)
    var errorDescription: String? {
        if case .unavailable(let msg) = self { return msg }
        return nil
    }
}

// MARK: — Zero-init helper

private func memset_zero<T>(_ value: inout T) {
    withUnsafeMutableBytes(of: &value) {
        _ = $0.initializeMemory(as: UInt8.self, repeating: 0)
    }
}

// MARK: — 48 kHz → 16 kHz resampler

private struct SherpaResampler {
    private var converter: AVAudioConverter?
    private var srcFmt: AVAudioFormat?
    private let dstFmt: AVAudioFormat = {
        AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16_000,
            channels: 1,
            interleaved: false
        )!
    }()

    mutating func convert(_ input: AVAudioPCMBuffer) -> [Float]? {
        if converter == nil || srcFmt != input.format {
            converter = AVAudioConverter(from: input.format, to: dstFmt)
            srcFmt = input.format
        }
        guard let conv = converter else { return nil }
        let ratio = dstFmt.sampleRate / input.format.sampleRate
        let outCap = AVAudioFrameCount(Double(input.frameLength) * ratio) + 64
        guard let out = AVAudioPCMBuffer(pcmFormat: dstFmt, frameCapacity: outCap) else { return nil }
        var supplied = false
        var err: NSError?
        _ = conv.convert(to: out, error: &err) { _, status in
            if supplied { status.pointee = .noDataNow; return nil }
            supplied = true; status.pointee = .haveData; return input
        }
        guard let data = out.floatChannelData?[0], out.frameLength > 0 else { return [] }
        return Array(UnsafeBufferPointer(start: data, count: Int(out.frameLength)))
    }
}
