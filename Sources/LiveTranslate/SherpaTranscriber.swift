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
///   5. On each endpoint (≥1 s trailing silence), read the committed partial
///      text BEFORE resetting the stream, then send (text + samples) to the
///      worker task.
///   6. Worker assigns a speaker label via campplus embedding and fires the
///      lifecycle callback.
///
/// **Crosstalk suppression:** same `markSystemVoiced` / `isSystemRecentlyVoiced`
/// pattern as WhisperCppTranscriber — mic buffers are zeroed when system audio
/// was recently heard.
final class SherpaTranscriber: Transcriber {

    // MARK: — Tunables

    /// Trailing-silence endpoint threshold fed to sherpa-onnx rule 1.
    static let endpointSilenceSeconds: Float = 1.0

    /// RMS used for the crosstalk gate (mirrors WhisperCppTranscriber).
    static let silenceRMSThreshold: Float = 0.012

    /// How long after system-voiced we keep treating mic as contaminated.
    static let crosstalkPersistSeconds: TimeInterval = 0.25

    // MARK: — Shared recognizer (loaded once, reused)

    private var recognizer: OpaquePointer?
    private let recognizerLock = NSLock()
    private var loadError: Error?

    private let speakerTracker: SpeakerTracker?

    // MARK: — Lifecycle callback (same contract as WhisperCppTranscriber)

    var onChunkLifecycle: (@Sendable (_ chunkID: UUID, _ source: SourceTag,
                                       _ event: ChunkLifecycle) -> Void)?

    enum ChunkLifecycle: Sendable {
        case listening
        case transcribing
        case completed(text: String, startSeconds: Double?, endSeconds: Double?)
        case dropped
    }

    // MARK: — Crosstalk state

    private var lastSystemVoicedAt: Date = .distantPast
    private let crosstalkLock = NSLock()

    func markSystemVoiced() {
        let now = Date()
        crosstalkLock.withLock { lastSystemVoicedAt = now }
    }

    func isSystemRecentlyVoiced() -> Bool {
        let now = Date()
        return crosstalkLock.withLock {
            now.timeIntervalSince(lastSystemVoicedAt) < Self.crosstalkPersistSeconds
        }
    }

    // MARK: — Init

    init() {
        speakerTracker = SpeakerTracker()
    }

    deinit {
        if let r = recognizer { SherpaOnnxDestroyOnlineRecognizer(r) }
    }

    // MARK: — Model loading

    private func ensureRecognizerLoaded() throws -> OpaquePointer {
        recognizerLock.lock()
        defer { recognizerLock.unlock() }
        if let r = recognizer { return r }
        if let e = loadError   { throw e }

        // Resolve absolute paths from the app bundle.
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

        // C-string lifetimes must span the SherpaOnnx call. Use nested
        // withCString blocks so each pointer remains valid throughout.
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

    // MARK: — Internal turn record

    /// Everything the worker needs to label + fire the lifecycle event.
    private struct TurnRecord: Sendable {
        let chunkID: UUID
        let index: Int
        let text: String
        let samples16k: [Float]   // voiced audio for speaker embedding
        let startSample: Int      // cumulative 16 kHz offset at turn start
        let endSample: Int        // cumulative 16 kHz offset at turn end (after endpoint)
    }

    // MARK: — Chunk loop

    private func runChunkLoop(
        recognizer: OpaquePointer,
        audio: AsyncStream<AVAudioPCMBuffer>,
        source: SourceTag,
        continuation: AsyncThrowingStream<SessionSnapshot, Error>.Continuation
    ) async throws {
        let (queue, queueSink) = AsyncStream<TurnRecord>.makeStream()
        // OpaquePointer doesn't conform to Sendable. Round-trip through UInt
        // (same pattern as the previous WhisperCppTranscriber) so the closure
        // capture is a value type and Swift's Sendable checker is satisfied.
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

        var turnSamples16k: [Float] = []
        var turnStartSample = 0
        var cumulativeSamples = 0
        var chunkIndex = 0
        var currentChunkID: UUID? = nil
        var hadVoice = false

        Log.line("SherpaTranscriber[\(tag)]: accumulator started")

        for await buf in audio {
            if Task.isCancelled { break }
            guard let samples16k = resampler.convert(buf), !samples16k.isEmpty else { continue }
            let n = samples16k.count

            // Crosstalk gate for mic stream.
            var effective = samples16k
            if source == .mic && isSystemRecentlyVoiced() {
                effective = [Float](repeating: 0, count: n)
            }

            // Stamp system-voiced for crosstalk detection.
            if source == .system {
                let rms = computeRMS(effective)
                if rms >= Self.silenceRMSThreshold { markSystemVoiced() }
            }

            let samplesBefore = cumulativeSamples
            cumulativeSamples += n

            // Feed samples into both VAD and ASR.
            effective.withUnsafeBufferPointer { ptr in
                if let v = vad {
                    SherpaOnnxVoiceActivityDetectorAcceptWaveform(v, ptr.baseAddress, Int32(n))
                }
                SherpaOnnxOnlineStreamAcceptWaveform(stream, 16_000, ptr.baseAddress, Int32(n))
            }

            // Detect speech via VAD (or RMS fallback).
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
                    Log.line("SherpaTranscriber[\(tag)]: voice onset #\(chunkIndex + 1) (id=\(id.uuidString.prefix(8))) at \(String(format:"%.2f",Double(samplesBefore)/16_000))s")
                    onChunkLifecycle?(id, source, .listening)
                    hadVoice = true
                }
                turnSamples16k.append(contentsOf: effective)
            }

            // Decode whenever ready.
            while SherpaOnnxIsOnlineStreamReady(recognizer, stream) != 0 {
                SherpaOnnxDecodeOnlineStream(recognizer, stream)
            }

            // Endpoint check.
            if hadVoice && SherpaOnnxOnlineStreamIsEndpoint(recognizer, stream) != 0 {
                chunkIndex += 1
                if let id = currentChunkID {
                    onChunkLifecycle?(id, source, .transcribing)
                }

                // *** Read the committed text BEFORE resetting the stream. ***
                let text: String
                if let result = SherpaOnnxGetOnlineStreamResult(recognizer, stream) {
                    text = String(cString: result.pointee.text)
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    SherpaOnnxDestroyOnlineRecognizerResult(result)
                } else {
                    text = ""
                }

                Log.line("SherpaTranscriber[\(tag)]: endpoint chunk #\(chunkIndex) text=\"\(text.prefix(60))\"")

                sink.yield(TurnRecord(
                    chunkID: currentChunkID ?? UUID(),
                    index: chunkIndex,
                    text: text,
                    samples16k: turnSamples16k,
                    startSample: turnStartSample,
                    endSample: cumulativeSamples
                ))

                // Reset for the next turn.
                SherpaOnnxOnlineStreamReset(recognizer, stream)
                turnSamples16k.removeAll(keepingCapacity: true)
                turnStartSample = cumulativeSamples
                currentChunkID = nil
                hadVoice = false
            }
        }

        // Flush any in-flight turn when the audio stream ends.
        if hadVoice && !turnSamples16k.isEmpty {
            SherpaOnnxOnlineStreamInputFinished(stream)
            while SherpaOnnxIsOnlineStreamReady(recognizer, stream) != 0 {
                SherpaOnnxDecodeOnlineStream(recognizer, stream)
            }
            chunkIndex += 1
            if let id = currentChunkID {
                onChunkLifecycle?(id, source, .transcribing)
            }
            let text: String
            if let result = SherpaOnnxGetOnlineStreamResult(recognizer, stream) {
                text = String(cString: result.pointee.text)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                SherpaOnnxDestroyOnlineRecognizerResult(result)
            } else {
                text = ""
            }
            sink.yield(TurnRecord(
                chunkID: currentChunkID ?? UUID(),
                index: chunkIndex,
                text: text,
                samples16k: turnSamples16k,
                startSample: turnStartSample,
                endSample: cumulativeSamples
            ))
        }

        Log.line("SherpaTranscriber[\(tag)]: accumulator exited (chunks=\(chunkIndex))")
    }

    // MARK: — Worker

    /// Assign a speaker label, emit lifecycle event, yield snapshot.
    private func processTurn(
        turn: TurnRecord,
        source: SourceTag,
        continuation: AsyncThrowingStream<SessionSnapshot, Error>.Continuation
    ) {
        guard !turn.text.isEmpty else {
            Log.line("SherpaTranscriber[\(source.rawValue)]: chunk #\(turn.index) empty text → dropped")
            onChunkLifecycle?(turn.chunkID, source, .dropped)
            return
        }

        let speakerLabel: String
        if let tracker = speakerTracker, !turn.samples16k.isEmpty {
            speakerLabel = tracker.label(for: turn.samples16k)
        } else {
            speakerLabel = "Speaker 1"
        }

        let labeled = "[\(speakerLabel)] \(turn.text)"
        let startSeconds = Double(turn.startSample) / 16_000
        let endSeconds   = Double(turn.endSample)   / 16_000

        Log.line("SherpaTranscriber[\(source.rawValue)]: chunk #\(turn.index) \(speakerLabel) → \"\(labeled.prefix(60))\"")

        onChunkLifecycle?(turn.chunkID, source, .completed(
            text: labeled, startSeconds: startSeconds, endSeconds: endSeconds
        ))

        continuation.yield(SessionSnapshot(sentences: [
            SessionSentence(
                text: labeled, isFinal: true,
                startSeconds: startSeconds, endSeconds: endSeconds
            )
        ]))
    }

    // MARK: — Helpers

    /// Build a Silero-VAD instance from the bundled model. Returns nil when
    /// the model file is absent (we fall back to RMS-based detection).
    private func makeVAD() -> OpaquePointer? {
        let vadPath = resourcePath(ModelConfig.vadModel)
        guard FileManager.default.fileExists(atPath: vadPath) else {
            Log.line("SherpaTranscriber: VAD model not in bundle, using RMS fallback")
            return nil
        }
        let vadURL = URL(fileURLWithPath: vadPath)
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

        let path     = vadURL.path
        let provider = ModelConfig.provider

        return path.withCString { cPath -> OpaquePointer? in
            silero.model      = cPath
            vadCfg.silero_vad = silero
            return provider.withCString { cProv -> OpaquePointer? in
                vadCfg.provider = cProv
                // bufferSizeInSeconds: 30 s window keeps the VAD
                // state warm for the whole run.
                return SherpaOnnxCreateVoiceActivityDetector(&vadCfg, 30)
            }
        }
    }

    /// Absolute path to a bundled resource given its bundle-relative name
    /// (e.g. `"sherpa-onnx-streaming-zipformer-de.../encoder.onnx"`).
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

/// Fill a Swift struct with zeros (equivalent to `memset(&v, 0, sizeof(v))`).
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
