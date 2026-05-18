import Foundation
import AVFoundation
import Combine
import Translation

/// Local port that `LiveAudioServer` listens on when a TTS voice is
/// available for the current target language. Hard-coded — keeping it
/// stable means listeners can bookmark `http://<host>.local:8765/`.
private let liveStreamPort: UInt16 = 8765

// MARK: - Pipeline overview
//
//   Mic ──▶ Denoise ──▶ SourcePipeline(mic) ──┐
//                       (recorder, SRTs)       │   chunk lifecycle
//                                              ├──▶ Pipeline.applyLifecycle
//   System ─▶ Denoise ─▶ SourcePipeline(sys) ──┤        │
//                       (recorder, SRTs)       │        ▼
//                                              │   @Published inflightChunks
//                                              │        │  on .completed
//                                              │        ▼
//                                              │   Translator (async, cached)
//                                              │        │  on result
//                                              │        ▼
//                                              └──▶ @Published sentences
//                                                       │  on prune/drop
//                                                       ▼
//                                                   TranscriptArchive (.jsonl,
//                                                   source-tagged) + per-source
//                                                   SubtitleArchives
//
// Both streams share one `SherpaTranscriber`. The UI sees in-flight chunks as
// reserved rows that flip through .listening → .transcribing →
// .translating and graduate to a `Sentence` with the same UUID — so the
// SwiftUI row identity stays stable across the lifecycle.

@MainActor
final class Pipeline: ObservableObject {

    // MARK: - Published UI state

    @Published private(set) var status: PipelineStatus = .idle
    @Published private(set) var sentences: [Sentence] = []

    /// Chunks that have been detected but haven't graduated to a final
    /// `Sentence` yet — reserved UI rows that show the live state of
    /// the pipeline (listening / transcribing / translating). When a
    /// chunk graduates, we append to `sentences` and remove from this
    /// list, both keyed by the same UUID so the row identity stays
    /// stable through the transition.
    @Published private(set) var inflightChunks: [InflightChunk] = []

    /// True from when `run()` enters until its cleanup completes.
    /// Drives the UI Start/Stop button.
    @Published private(set) var isActive: Bool = false
    var isRunning: Bool { isActive }

    /// `http://<host>.local:8765/` when a translated-audio TTS stream
    /// is live for the current target language; nil otherwise. The UI
    /// shows the share icon when non-nil; clicking pops the URL + a
    /// QR code so a phone-with-headphones can listen along.
    @Published private(set) var liveStreamURL: String?

    /// True while the TTS pipeline is doing real work — the model has
    /// finished its lazy load AND there's at least one listener
    /// connected to `/live.wav`. UI shows this as a green stream
    /// icon. Flips off when the last listener disconnects (the model
    /// stays resident for the rest of the session, but it's no longer
    /// being driven).
    @Published private(set) var ttsActive: Bool = false

    /// Backing flags for `ttsActive`. Set by callbacks from the
    /// speaker / server hopping back to MainActor; combined in
    /// `recomputeTTSActive()`.
    private var ttsModelLoaded = false
    private var ttsListenerCount = 0

    // MARK: - Language (compile-time constant)

    /// Source language — compile-time constant from ModelConfig.
    /// Exposed as a property (not a published var) so existing code that reads
    /// `pipeline.source` and `pipeline.target` (e.g. TranscriptView's
    /// translationConfig) continues to work without changes.
    let source: SourceLocale = SourceLocale(identifier: "\(ModelConfig.sourceLanguage)-\(ModelConfig.sourceLanguage.uppercased())")

    /// Target language — compile-time constant from ModelConfig.
    let target: TargetLanguage = TargetLanguage(code: ModelConfig.targetLanguage, name: ModelConfig.targetLanguage)

    /// Non-protected sentence older than this is pruned.
    var maxAgeSeconds: TimeInterval = 300

    /// Hard cap on retained sentences.
    var maxSentenceCount: Int = 50

    // MARK: - Stages

    private let micSource: AudioSource
    private let systemSource: AudioSource
    private let transcriber: Transcriber
    private let translator: Translator

    // MARK: - Internal state

    /// Translation cache keyed by source text. Bounded.
    private var translationCache: [String: String] = [:]
    private let maxCacheEntries: Int = 200

    /// Last time we dispatched a partial translation per chunk. Used to
    /// throttle partial translations to at most once per second so we
    /// don't flood the translator with every token emission.
    private var partialTranslationTimers: [UUID: Date] = [:]

    private var runTask: Task<Void, Never>?
    /// Shared JSONL archive — sentences from all sources interleave
    /// here, distinguished by the `source` field.
    private var archive: TranscriptArchive?
    /// One per-stream pipeline per `SourceTag`. Each owns its recorder
    /// and per-source SRT writers. Held here so `stop()` can signal
    /// each one to drain its audio source, and so `recordSentence`
    /// can route per-source SRT writes to the right files.
    private var sourcePipelines: [SourceTag: SourcePipeline] = [:]
    /// Live merged-SRT writers, one per language. Updated whenever a
    /// sentence graduates so the file on disk tracks the session in
    /// real time. `MKVExporter` reads from these at session end —
    /// it doesn't re-merge.
    private var mergedSubtitles: [String: MergedSubtitleArchive] = [:]
    /// Where session artifacts live during the run + where the final
    /// zip lands. Captured at run start so the cleanup path can read
    /// it after `defer` clears the rest of the state.
    private var currentOutputs: Paths.Outputs?
    /// Speaks finalized translations as 24 kHz PCM16 buffers and pushes
    /// them into `liveAudioServer`. Created at run start only if the
    /// kitten-mini ONNX TTS model is bundled. Nil otherwise —
    /// the whole stream feature is skipped (icon stays hidden).
    private var ttsSpeaker: OnnxTTSSpeaker?
    private var liveAudioServer: LiveAudioServer?

    /// Set by a settings-change observer; read by run()'s defer. When
    /// true after the current run winds down, defer spawns a fresh run.
    /// Cleared by `stop()` so a user-initiated Stop doesn't auto-restart.
    private var restartRequested: Bool = false
    /// Wall-clock instant the current run's recording started. Used to
    /// compute SRT cue offsets (`sentence.createdAt - runStartedAt`).
    private var runStartedAt: Date = .distantPast

    init(
        micSource: AudioSource? = nil,
        systemSource: AudioSource? = nil,
        transcriber: Transcriber? = nil,
        translator: Translator? = nil
    ) {
        self.micSource = micSource ?? MicrophoneSource()
        self.systemSource = systemSource ?? SystemAudioSource()
        let sherpaTranscriber = transcriber ?? SherpaTranscriber()
        self.transcriber = sherpaTranscriber
        self.translator = translator ?? AppleTranslator()

        // Wire the transcriber's per-chunk lifecycle callback. The
        // accumulator + worker invoke it from background tasks;
        // `handleChunkLifecycle` hops to MainActor and runs the state
        // machine (inflight bookkeeping + graduation to Sentence +
        // translation dispatch).
        if let s = sherpaTranscriber as? SherpaTranscriber {
            s.onChunkLifecycle = { [weak self] id, source, event in
                self?.handleChunkLifecycle(id: id, source: source, event: event)
            }
        }
    }

    // MARK: - Chunk lifecycle handler

    /// Receives lifecycle events from `SherpaTranscriber` (called
    /// from off-MainActor tasks). All state mutation happens inside
    /// the `Task { @MainActor in ... }` so SwiftUI sees a single
    /// coherent change per event.
    nonisolated private func handleChunkLifecycle(
        id: UUID, source: SourceTag, event: SherpaTranscriber.ChunkLifecycle
    ) {
        Task { @MainActor in
            self.applyLifecycle(id: id, source: source, event: event)
        }
    }

    /// MainActor-isolated state-machine for one chunk's lifecycle.
    /// Maintains `inflightChunks` and graduates completed chunks
    /// (plus their translation, if any) to `sentences`.
    private func applyLifecycle(
        id: UUID, source: SourceTag, event: SherpaTranscriber.ChunkLifecycle
    ) {
        switch event {
        case .listening:
            // Reserve a row at voice onset.
            inflightChunks.append(InflightChunk(
                id: id, source: source, startedAt: Date(), state: .listening
            ))

        case .partial(let text):
            // Streaming ASR produced new tokens. Show the partial text in the
            // UI row and, at most once per second, kick off a translation so
            // the user sees a rolling translated preview.
            guard let idx = inflightChunks.firstIndex(where: { $0.id == id }) else { return }
            // Preserve any translation we already have for this chunk.
            let existingTranslation: String?
            if case .partial(_, let t) = inflightChunks[idx].state { existingTranslation = t }
            else { existingTranslation = nil }
            inflightChunks[idx].state = .partial(text: text, translation: existingTranslation)

            let srcLang = String(self.source.identifier.prefix(2))
            let tgtLang = self.target.code
            if srcLang == tgtLang {
                // Same language — translation IS the transcription.
                inflightChunks[idx].state = .partial(text: text, translation: text)
                return
            }
            // Throttle: at most one dispatch per second per chunk.
            let now = Date()
            guard now.timeIntervalSince(partialTranslationTimers[id] ?? .distantPast) >= 1.0 else { return }
            partialTranslationTimers[id] = now
            Task { @MainActor [weak self] in
                guard let self else { return }
                guard let translated = try? await self.translator.translate(text) else { return }
                // Only apply if the chunk is still showing this exact partial text.
                guard let idx = self.inflightChunks.firstIndex(where: { $0.id == id }),
                      case .partial(let cur, _) = self.inflightChunks[idx].state,
                      cur == text else { return }
                self.inflightChunks[idx].state = .partial(text: text, translation: translated)
            }

        case .completed(let text, let startSeconds, let endSeconds):
            // Whisper produced text. Either graduate immediately (no
            // translation needed / cached) or flip to "translating"
            // and dispatch the translator.
            let createdAt = startSeconds.map { runStartedAt.addingTimeInterval($0) } ?? Date()
            let endedAt = endSeconds.map { runStartedAt.addingTimeInterval($0) } ?? createdAt
            let srcLang = String(self.source.identifier.prefix(2))
            let tgtLang = self.target.code
            Log.line("lifecycle[\(source.rawValue)]: completed id=\(id.uuidString.prefix(8)) \"\(text.prefix(40))\" → \(srcLang)→\(tgtLang)")

            if srcLang == tgtLang {
                graduate(id: id, source: source, text: text, translation: text,
                         createdAt: createdAt, endsAt: endedAt)
                return
            }
            if let cached = translationCache[text] {
                Log.line("lifecycle[\(source.rawValue)]: cached translation hit for id=\(id.uuidString.prefix(8))")
                graduate(id: id, source: source, text: text, translation: cached,
                         createdAt: createdAt, endsAt: endedAt)
                return
            }
            // Need to translate. If a partial translation is already on
            // the row, keep showing it (no italic "translating" flash) —
            // graduate() will swap it for the final translation in place
            // when the translator returns. Only show the "translating"
            // placeholder when there's nothing better to display.
            if let idx = inflightChunks.firstIndex(where: { $0.id == id }) {
                if case .partial(_, let existing) = inflightChunks[idx].state,
                   let t = existing, !t.isEmpty {
                    inflightChunks[idx].state = .partial(text: text, translation: t)
                    Log.line("lifecycle[\(source.rawValue)]: endpoint with partial translation, holding (id=\(id.uuidString.prefix(8)))")
                } else {
                    inflightChunks[idx].state = .translating(text: text)
                    Log.line("lifecycle[\(source.rawValue)]: state → translating, dispatching translator (id=\(id.uuidString.prefix(8)))")
                }
            }
            // Explicit `@MainActor` on the Task closure so isolation
            // doesn't depend on Swift 5 inheritance heuristics. The
            // translator is @MainActor too; avoiding actor hops mid-
            // task is what makes the post-await `graduate` reliably
            // mutate `@Published` state on the right actor.
            Task { @MainActor [weak self] in
                guard let self else { return }
                do {
                    let translated = try await self.translator.translate(text)
                    Log.line("lifecycle[\(source.rawValue)]: translator returned for id=\(id.uuidString.prefix(8)): \"\(translated.prefix(40))\"")
                    self.cacheTranslation(source: text, translated: translated)
                    self.graduate(id: id, source: source, text: text, translation: translated,
                                  createdAt: createdAt, endsAt: endedAt)
                } catch {
                    Log.line("lifecycle[\(source.rawValue)]: translator error for id=\(id.uuidString.prefix(8)): \(error.localizedDescription)")
                    // Graduate with empty translation so the user still
                    // sees the transcription text — they can re-run for
                    // a retry.
                    self.graduate(id: id, source: source, text: text, translation: "",
                                  createdAt: createdAt, endsAt: endedAt)
                }
            }

        case .dropped:
            // Chunk filtered out by the worker (no voice, too short,
            // empty text). No sentence to graduate; just drop the row.
            inflightChunks.removeAll { $0.id == id }
            partialTranslationTimers.removeValue(forKey: id)
        }
    }

    /// Turn an inflight chunk into a `Sentence`. The Sentence reuses
    /// the chunk's UUID — the UI presents inflight rows and sentences
    /// as a single merged ForEach (see `TranscriptView.sentenceList`),
    /// so identity continuity here means SwiftUI sees a same-row
    /// content swap instead of remove+insert, eliminating the
    /// graduation flicker. The sentence is also archived immediately
    /// to JSONL + per-source SRT + merged SRT(s) so the work-dir
    /// files stay live throughout the session (not just at end).
    private func graduate(
        id: UUID, source: SourceTag, text: String, translation: String,
        createdAt: Date, endsAt: Date
    ) {
        let sentence = Sentence(
            id: id, text: text, translation: translation, source: source,
            createdAt: createdAt, endsAt: endsAt, lastModified: Date()
        )
        sentences.append(sentence)
        inflightChunks.removeAll { $0.id == id }
        partialTranslationTimers.removeValue(forKey: id)
        recordSentence(sentence)
        // Feed the translation into the live audio stream — but only
        // if someone's actually listening. With zero subscribers on
        // /live.wav the speaker would synthesize into the void; gating
        // here skips the synthesis and (combined with `OnnxTTSSpeaker`'s
        // lazy load) keeps the model entirely unloaded while no one
        // has ever connected this run.
        if !translation.isEmpty,
           let server = liveAudioServer,
           server.audioListenerCount > 0 {
            ttsSpeaker?.enqueue(translation)
        }
        enforceMaxCount()
    }

    /// Write a freshly-graduated sentence to disk (shared JSONL +
    /// per-language merged SRTs) and broadcast the same JSONL line on
    /// the listen-page SSE channel. All writes are queue-backed so
    /// this returns immediately.
    private func recordSentence(_ s: Sentence) {
        archive?.append(s)
        if let line = TranscriptArchive.encodeLine(s) {
            liveAudioServer?.publishTranscript(jsonLine: line)
        }
        let start = s.createdAt.timeIntervalSince(runStartedAt)
        let end = max(start, s.endsAt.timeIntervalSince(runStartedAt))
        let srcLang = String(source.identifier.prefix(2))
        mergedSubtitles[srcLang]?.add(text: s.text, startSeconds: start, endSeconds: end)
        let tgtLang = target.code
        if tgtLang != srcLang, !s.translation.isEmpty {
            mergedSubtitles[tgtLang]?.add(text: s.translation, startSeconds: start, endSeconds: end)
        }
    }

    // MARK: - Public controls

    func toggle() {
        if runTask != nil { stop() } else { runTask = Task { await run() } }
    }

    func stop() {
        Log.line("Pipeline.stop()")
        restartRequested = false
        // Flip to `.finalizing` immediately so the UI shows the
        // spinner + "Stopping…" the moment the button is pressed.
        // run()'s end-of-session path will also set this, but by the
        // time it does (audio drain + background cancel) seconds may
        // have passed; the user would see "Stop" the whole time
        // without this.
        if isActive {
            status = .finalizing
        }
        stopActiveSources()
    }

    /// Graceful shutdown: stopping each source pipeline ends its
    /// audio broadcaster, which drains the per-stream accumulator +
    /// worker + recording loops. Without this, cancellation would
    /// abort the pipeline mid-flight and drop trailing audio.
    private func stopActiveSources() {
        for pipeline in sourcePipelines.values {
            Task { await pipeline.stop() }
        }
    }

    func clear() {
        sentences = []
        inflightChunks = []
    }

    /// Load a canned set of mic/system sentences into the UI for
    /// screenshots and visual-regression checks. Triggered from the
    /// macOS menu bar (`Debug → Load fixture sentences`). Doesn't
    /// touch the audio pipeline; appends directly to `sentences`.
    func loadDebugFixtures() {
        let now = Date()
        // (source, transcription, translation)
        let fixtures: [(SourceTag, String, String)] = [
            (.mic, "Ich spreche diesen Text auf Deutsch",
             "I'm speaking this text in German"),
            (.mic, "Alles, was ich sage oder was der PC ausgibt, wird live ins Englische übersetzt",
             "Everything I say or what the PC outputs is translated live into English"),
            (.system, "Hallo, dies ist ein test audio playback",
             "Hello, this is a test audio playback"),
            (.mic, "Der Text wird auch in eine Datei geschrieben, die ich später auslesen kann",
             "The text is also written in a file that I can read later"),
        ]
        for (i, (source, text, translation)) in fixtures.enumerated() {
            let created = now.addingTimeInterval(Double(i) * 2)
            sentences.append(Sentence(
                id: UUID(), text: text, translation: translation, source: source,
                createdAt: created, endsAt: created.addingTimeInterval(1.5),
                lastModified: created
            ))
        }
    }

    /// The View calls this from `.translationTask` to hand us a fresh
    /// `TranslationSession`. Hides the AppleTranslator downcast.
    func installTranslationSession(_ session: TranslationSession?) {
        (translator as? AppleTranslator)?.setSession(session)
    }

    /// Block until queued writes hit disk. Sentences are already
    /// archived on graduate (`recordSentence`); this just awaits the
    /// per-writer dispatch queues so nothing is in flight when the
    /// process exits. Safe to call from `applicationWillTerminate`.
    func flushPendingSentences() {
        archive?.flush()
        for sp in sourcePipelines.values { sp.flush() }
        for merged in mergedSubtitles.values { merged.flush() }
    }

    // MARK: - Run loop

    private func run() async {
        isActive = true
        // The defer only clears terminal state. Disk flushes + MKV +
        // zip happen explicitly below so the exporter and zipper see
        // complete files.
        defer {
            if case .stopped = status { } else { status = .idle }
            runTask = nil
            isActive = false
            archive = nil
            sourcePipelines.removeAll()
            mergedSubtitles.removeAll()
            currentOutputs = nil
            inflightChunks.removeAll()
            partialTranslationTimers.removeAll()
            ttsSpeaker?.stop()
            ttsSpeaker = nil
            liveAudioServer?.stop()
            liveAudioServer = nil
            liveStreamURL = nil
            ttsModelLoaded = false
            ttsListenerCount = 0
            recomputeTTSActive()
            if restartRequested {
                restartRequested = false
                sentences = []
                runTask = Task { await run() }
            }
        }

        // 1. Mic permission. (SCK prompts on its own when capture starts.)
        status = .requestingPermissions
        let micGranted: Bool = await withCheckedContinuation { c in
            AVCaptureDevice.requestAccess(for: .audio) { c.resume(returning: $0) }
        }
        guard micGranted else { status = .stopped(reason: "Microphone permission denied"); return }

        // 2. Per-stream `DenoisingAudioSource`s. The mic instance gets
        //    a crosstalk gate that queries the transcriber's
        //    `lastSystemVoicedAt` and zeros the buffer when system was
        //    recently voiced. Applied upstream of the broadcaster so
        //    BOTH consumers (AudioRecorder + transcriber accumulator)
        //    see the same muted audio — without this, `.mic.wav` still
        //    had the raw speaker bleed even though the transcript
        //    suppressed it.
        let sherpa = self.transcriber as? SherpaTranscriber
        let micDenoised = DenoisingAudioSource(
            micSource,
            label: "mic",
            muteWhen: { [weak sherpa] in sherpa?.isSystemRecentlyVoiced() ?? false }
        )
        let systemDenoised = DenoisingAudioSource(systemSource, label: "system")
        let denoised: [SourceTag: AudioSource] = [.mic: micDenoised, .system: systemDenoised]

        status = .starting
        do {
            async let m: Void = micDenoised.start()
            async let s: Void = systemDenoised.start()
            _ = try await (m, s)
        } catch {
            status = .stopped(reason: "Audio: \(error.localizedDescription)"); return
        }

        // 3. Open output files. All session artifacts go into a temp
        //    work dir; we zip + delete it after the MKV is built.
        runStartedAt = Date()
        let srcLangCode = String(source.identifier.prefix(2))
        let tgtLangCode = target.code
        let outputs: Paths.Outputs
        do {
            outputs = try Paths.newRunOutputs(now: runStartedAt)
            archive = try TranscriptArchive(at: outputs.transcript)
        } catch {
            Log.line("Pipeline: opening work dir failed: \(error.localizedDescription)")
            for src in denoised.values { await src.stop() }
            status = .stopped(reason: "Output: \(error.localizedDescription)")
            return
        }
        currentOutputs = outputs

        // 4. Build per-source pipelines (just a recorder per stream —
        //    SRT writing happens at the merged level in
        //    `recordSentence`).
        for tag in SourceTag.allCases {
            guard let src = denoised[tag] else { continue }
            sourcePipelines[tag] = SourcePipeline(
                source: tag,
                audioSource: src,
                transcriber: self.transcriber,
                locale: self.source,
                recorder: try? AudioRecorder(at: outputs.recording(tag))
            )
        }
        // 4b. Open live merged-SRT archives, one per distinct language.
        let allLangs: [String] = (srcLangCode == tgtLangCode) ? [srcLangCode] : [srcLangCode, tgtLangCode]
        for lang in allLangs {
            if let merged = try? MergedSubtitleArchive(at: outputs.mergedSubtitle(lang)) {
                mergedSubtitles[lang] = merged
            }
        }
        Log.line("Run outputs: \(outputs.workDir.path) → \(outputs.zipDestination.lastPathComponent)")

        // 4c. Spin up the live translated-audio stream IF
        //     (a) src != tgt language (otherwise it's just an echo),
        //     (b) the kitten-mini ONNX TTS model is bundled.
        //     If either fails, leave `ttsSpeaker` / `liveAudioServer` nil
        //     so the UI stream icon stays hidden.
        if srcLangCode != tgtLangCode, OnnxTTSSpeaker.isAvailable() {
            let server = LiveAudioServer(port: liveStreamPort)
            do {
                try server.start()
                let speaker = OnnxTTSSpeaker(onPCM: { [weak server] pcm in
                    server?.append(pcm)
                }, onActivityChanged: { [weak server] active in
                    server?.setSpeaking(active)
                }, onModelLoaded: { [weak self] in
                    Task { @MainActor [weak self] in
                        self?.ttsModelLoaded = true
                        self?.recomputeTTSActive()
                    }
                })
                server.onAudioListenerCountChanged = { [weak self] count in
                    Task { @MainActor [weak self] in
                        self?.ttsListenerCount = count
                        self?.recomputeTTSActive()
                    }
                }
                self.liveAudioServer = server
                self.ttsSpeaker = speaker
                self.liveStreamURL = LiveAudioServer.streamURL(port: liveStreamPort)
                Log.line("Live audio stream: \(self.liveStreamURL ?? "?") (kitten-mini ONNX TTS)")
            } catch {
                Log.line("LiveAudioServer.start failed: \(error.localizedDescription) — TTS stream disabled this run")
            }
        } else if srcLangCode == tgtLangCode {
            Log.line("Live audio stream: skipped (src == tgt)")
        } else {
            Log.line("Live audio stream: kitten-mini model not bundled — TTS stream disabled this run")
        }

        status = .running

        // 5. Background prune loop (the translation worker is gone —
        //    translation happens inline in the lifecycle handler).
        //    Cancellable separately from the audio path.
        let backgroundTask = Task {
            await self.runPruneLoop()
        }

        // 6. Run each SourcePipeline. They emit chunk lifecycle events
        //    via the SherpaTranscriber callback; UI state is
        //    managed in `applyLifecycle` (graduation, translation).
        Log.line("Pipeline: entering audio-path TaskGroup with \(sourcePipelines.count) pipelines")
        await withTaskGroup(of: Void.self) { group in
            for sp in sourcePipelines.values {
                group.addTask { await sp.run() }
            }
        }
        Log.line("All source pipelines drained")

        // 7. Cancel background workers and wait.
        backgroundTask.cancel()
        _ = await backgroundTask.value

        // 8. Final audio cleanup. Each `stop()` is idempotent.
        for sp in sourcePipelines.values { await sp.stop() }

        // 9. Finalize: flush writers, build MKV, zip work dir → docs.
        //    UI shows a spinner throughout. Sentences themselves are
        //    already archived (`recordSentence` ran on each graduate),
        //    so flushing just awaits the disk queues to drain.
        status = .finalizing
        archive?.flush()
        for sp in sourcePipelines.values { sp.flush() }
        for merged in mergedSubtitles.values { merged.flush() }
        Log.line("Pipeline: finalize — writers flushed, building MKV")
        await MKVExporter.export(outputs: outputs, langs: allLangs)
        Log.line("Pipeline: packing \(outputs.shippedFiles.count) files → \(outputs.zipDestination.lastPathComponent)")
        await ZipArchiver.zipFilesAndCleanup(
            outputs.shippedFiles,
            into: outputs.zipDestination,
            workDir: outputs.workDir
        )
    }

    // MARK: - Translation cache

    private func cacheTranslation(source: String, translated: String) {
        translationCache[source] = translated
        // LRU-ish: drop ~10% oldest by insertion order when over cap.
        if translationCache.count > maxCacheEntries {
            let toDrop = translationCache.count - (maxCacheEntries * 9 / 10)
            for key in translationCache.keys.prefix(toDrop) {
                translationCache.removeValue(forKey: key)
            }
        }
    }

    // MARK: - Prune

    private func runPruneLoop() async {
        while !Task.isCancelled {
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            prune()
        }
    }

    /// We never drop the most-recent sentence so the UI is never empty
    /// mid-stream. Returned as a `Set<UUID>` so callers can use it as
    /// a quick membership check inside a loop.
    private func protectedIDs() -> Set<UUID> {
        guard let id = sentences.last?.id else { return [] }
        return [id]
    }

    private func prune() {
        let protected = protectedIDs()
        let cutoff = Date().addingTimeInterval(-maxAgeSeconds)
        var i = sentences.count - 1
        while i >= 0 {
            let s = sentences[i]
            if !protected.contains(s.id) && s.lastModified < cutoff {
                sentences.remove(at: i)
            }
            i -= 1
        }
    }

    /// Recompute `ttsActive` from the two backing flags. Called
    /// whenever either changes (load completion, listener count
    /// fluctuation) or when the stream is torn down at run end.
    private func recomputeTTSActive() {
        ttsActive = ttsModelLoaded && ttsListenerCount > 0
    }

    private func enforceMaxCount() {
        while sentences.count > maxSentenceCount {
            let protected = protectedIDs()
            if let i = sentences.firstIndex(where: { !protected.contains($0.id) }) {
                sentences.remove(at: i)
            } else {
                break
            }
        }
    }
}
