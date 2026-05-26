---
name: livetranslate-architecture
description: Use whenever editing or reasoning about LiveTranslate Swift sources — covers data flow (audio → ASR → translation → UI/TTS/SSE/OBS), per-file roles, state machines, UUID continuity, sentence splitting, crosstalk, and latency knobs. Read before touching any `Sources/LiveTranslate/*.swift`.
---

# LiveTranslate architecture

After any meaningful code change that contradicts this skill, update this file in the same commit. After learning a new non-obvious interaction, add it here.

## Pipeline overview

```mermaid
flowchart TD
  Mic --> DenMic[DenoisingAudioSource mic\nRNNoise+AGC+crosstalk gate]
  Sys[System/SCK] --> DenSys[DenoisingAudioSource sys\nAGC only]
  DenMic --> RecMic[AudioRecorder .mic.wav 48k Int16]
  DenSys --> RecSys[AudioRecorder .system.wav 48k Int16]
  DenMic --> TMic[transcribe() per stream]
  DenSys --> TSys[transcribe() per stream]
  TMic & TSys --> Pipe[Pipeline.applyLifecycle @MainActor]
  Pipe --> UI[sentences + inflightChunks]
  Pipe --> JSONL & SRT & Server[LiveAudioServer 8765]
  Pipe -->|listeners>0| TTS[OnnxTTSSpeaker → 24k PCM16] --> Server
  Pipe -->|en detected| OBS[en→de obsTranslator → /obs-events]
```

## Per-stream pipelines (no shared locks)

Mic and system have independent `DenoisingAudioSource` instances and independent `transcribe()` calls on a shared sherpa-onnx `OnlineRecognizer` (recognizer loaded once via `NSLock`-guarded `ensureRecognizerLoaded`; `OnlineStream` creation is cheap). Per-source flags:
- mic: `denoise=true, applyAGC=true, muteWhen=sherpa.isSystemRecentlyVoiced`
- system: `denoise=false, applyAGC=true` (SCK is already clean)

Never use `AVAudioEngine.setVoiceProcessingEnabled` — see bug history #43.

## Audio format invariant

- All sources standardize on **48 kHz mono Float32** via `AVAudioConverter`.
- RNNoise wants ±32768-scaled Float32 in 480-sample (10 ms) frames at 48 kHz.
- `SherpaResampler` (per-stream) resamples 48→16 kHz internally for ASR.
- `AudioRecorder` writes 48 kHz mono **Int16** WAV (not 16 kHz). AVAudioFile flush requires `file = nil` to finalize header (bug #31).
- Mic tap buffer: 512 samples (~10.7 ms) — latency knob.

## VAD: energy + ZCR (no Silero)

`SherpaTranscriber.isVoiced`: `RMS ≥ silenceRMSThreshold (0.012)` AND ZCR ∈ `[minZeroCrossRate, maxZeroCrossRate]`. ~hundreds of ns per 10 ms chunk. Drives chunk-onset, voiced/silent transitions, crosstalk timestamp.

## Lifecycle state machine

`SherpaTranscriber.onChunkLifecycle: (UUID, SourceTag, ChunkLifecycle) -> Void` is the single source of truth. The `SessionSnapshot` stream from `transcribe()` is drained and ignored.

`ChunkLifecycle` cases:
- `.listening` — voice onset; reserve row with new UUID
- `.partial(text)` — fires per hypothesis change; UI shows active portion after last force-complete
- `.completed(text, startSeconds, endSeconds)` — endpoint or force-complete; final hypothesis; sample timestamps at 16 kHz
- `.dropped` — voiceless/empty/punctuation-only; collapse row

`Pipeline.handleChunkLifecycle` is `nonisolated`; hops to `@MainActor` via `Task { @MainActor in }` to run `applyLifecycle`.

`InflightChunk.State` (distinct from lifecycle; see `Types.swift`):
- `.listening`
- `.partial(text, translation?)`
- `.translating(text)` — only when no prior partial translation existed
- (NO `.transcribing` — sherpa streams tokens live)

## UUID continuity (flicker-free graduation)

Chunk UUID set at `.listening`, carried through partial/completed, **reused** by `Sentence` on `graduate()`. `TranscriptView.displayRows` merges `sentences + inflightChunks` into `[DisplayRow]` keyed by UUID; `ForEach` does in-place update instead of remove+insert. `DisplayRow` enum has `case sentence(Sentence)` / `case inflight(InflightChunk)`, both expose `id: UUID`. Use `.contentTransition(.identity)` on partial Text (bug #53 — `.opacity` flickers).

## Sentence splitting (three mechanisms)

1. **Force-complete at `maxCharsPerRow=30` + boundary.** `sentenceBoundary` finds earliest `. `(after letter — skips decimals), `? `, `! `. Emit `.completed` up to+including punctuation; advance `committedLength` by `count+1`; reset `chunkStartSample=cumulativeSamples`; fire new `.listening` + `.partial(remainder)`.
2. **Endpoint at `endpointSilenceSeconds=1.2s`** (sherpa rule-1). Semantic suppression: if `!hasTerminalPunct && trailingSilence<2.4s && utterance<59s`, IGNORE endpoint — German breaths. If suppressed, do NOT reset `committedLength`/`chunkStartSample` (bug #32). If remainder is punctuation-only and `committedLength>0`, fire `.dropped`.
3. **Stream end (Stop)** — `InputFinished` + final decode + emit.

Worker drop conditions: empty text OR no alphanumeric chars.

## normalizeHypothesis

Strips leading `.?! ` chars from every hypothesis. German zipformer prepends sentence-final punct of the previous utterance after stream reset.

## Timestamps

`chunkStartSample` (16 kHz idx at voice onset, reset on endpoint/force-complete) and `lastVoicedEndSample` (16 kHz idx at voiced→silent transition). `endSample = max(chunkStartSample, lastVoicedEndSample)` so SRT cues end at last voiced sample, not trailing silence. Worker converts to seconds via `/16_000`. Pipeline anchors via `runStartedAt`. Recorder shares the broadcaster — sample positions = WAV positions.

## Partial-translation throttle

In `applyLifecycle(.partial)`: `partialTranslationTimers[id]` throttles to **0.3 s** per chunk. Apply result only if chunk is still `.partial` with the same text (stale-skip). On `.completed`: if a partial translation is visible, keep showing it (`.partial(text, translation: existing)`) until final lands — don't flash to `.translating`.

## Translation framework quirks

- `TranslationSession` only via SwiftUI `.translationTask` modifier (no public init).
- Park the closure with `AsyncStream<Never>` (`Task.sleep(.max)` trips precondition on macOS 15).
- Language codes must be bare (`"de"`, `"en"`) — `.prefix(2)` of identifier.
- `Task { @MainActor [weak self] in ... }` is load-bearing on graduation; Swift's actor inheritance is unreliable without explicit annotation.
- Translation cache `[String:String]` capped at 200; drops ~10% oldest on overflow.
- Second `AppleTranslator` instance (`obsTranslator`) for `en→de` (OBS), wired via `installOBSTranslationSession`.

## Crosstalk suppression

- System accumulator: stamps `lastSystemVoicedAt` when buffer RMS ≥ 0.012.
- `SherpaTranscriber.isSystemRecentlyVoiced()` returns true within `crosstalkPersistSeconds=0.25s` (`NSLock`-guarded).
- Mic accumulator: zeros 16 kHz effective buffer if true (recognizer-side gate).
- `DenoisingAudioSource(mic).muteWhen = { sherpa?.isSystemRecentlyVoiced() }` — broadcaster-side gate, so BOTH recorder and recognizer see muted audio.

## Compile-time language pair

`ModelConfig.sourceLanguage = "de"`, `targetLanguage = "en"`. Flows into `Pipeline.source/target` (`let`, not `@Published`), `TranscriptView.translationConfig`, static label, run-time srcLang/tgtLang comparisons. No UserDefaults, no pickers, no `@AppStorage` for language. Retarget = change constants, swap ASR model dir, rebuild.

## isActive gate

`Pipeline.isActive` is set false in `run()`'s `defer` AFTER MKV+zip finalize (so legitimate during-finalize completions still graduate). Both `applyLifecycle` and `graduate` early-return on `!isActive` — drops late translation task hops that would otherwise resurrect cleared rows (bug #40).

## AppSettings

`ObservableObject` with `@Published` properties; `didSet` persists to `UserDefaults`. **Never use `@AppStorage` inside an `ObservableObject`** (bug #42 — no `objectWillChange`). Inject via `.environmentObject` at root. `Pipeline.bindSettings` forwards via Combine to web (throttled 200 ms). `OptionsView` is the SwiftUI `Settings` scene; `windowOpacity` applied via SwiftUI `.opacity()` on ZStack background (bug #44 — never NSWindow `alphaValue`).

## UI layout decisions

- **Main window content = sentence list ONLY.** Controls live in `MenuBarView` popover and `OptionsView`. Reintroducing controls in `TranscriptView` is a regression (bug #41).
- `NSWindow.level = .statusBar` + `[.canJoinAllSpaces, .fullScreenAuxiliary]` + `orderFrontRegardless()` on space change observer.
- `SummaryView` is a separate window managed by `SummaryWindowController` watching `aiAnalysisEnabled`.
- `AppSettings.LayoutMode`: `.mixed` / `.sideBySide` / `.compact`. No top-level `compactMode` chevron — Options is the single layout surface (bug #47).
- Row animation: `.transition(.opacity)` ONLY on row add/remove (~0.09s); content uses `.contentTransition(.identity)` (bug #53).

## LiveAudioServer (port 8765)

Hand-rolled HTTP/1.1 on `NWListener`. Routes:
- `/` — listen page HTML (SSE settings + hypothesis + transcript)
- `/live.wav` — open-ended WAV (24 kHz mono PCM16 LE, `0xFFFFFFFF` data size)
- `/events` — SSE transcript (replays buffer cap 200; named `hypothesis`/`hypothesis-done` events NOT replayed)
- `/obs` — transparent overlay HTML (top-right speech-bubble badge)
- `/obs-events` — dedicated SSE for OBS subs (separate dict)

Started unconditionally at run begin. Heartbeat: 200 ms tick (50 ms silence if idle >100 ms and speaker inactive; SSE `: ping` every 25 ticks).

Read accumulation: `receiveRequest` accumulates until `\r\n\r\n`, cap 16 KB (bug #48). SSE subscriber registration must precede replay snapshot, both under same lock (bug #49). `jsonEscape` must handle all `<0x20` control chars (bug #50).

`streamURL` prefers private IPv4 (skips `utun`/`ipsec`/`tun`) over `.local` over `localhost`.

## TTS gating

Three flags: `ttsListenerCount` (server callback), `ttsModelLoaded` (kitten-mini lazy-load callback), `ttsActive = loaded && count>0` (drives green icon, bug #30). Synthesis only dispatched if `audioListenerCount>0`. Backpressure drops OLDEST (cap 5).

## OBS English detection

`NLLanguageRecognizer.dominantLanguage` synchronous in `graduate()` on final English translation. On-device, <1 ms, >99% accuracy for DE→EN pair. Guards dispatch to `obsTranslator` for German subtitles via `publishOBSSubtitle` (JSONL).

Listen page: insert translation row ABOVE transcription (`insertBefore`, bug #51). On `hypothesis-done` use `data-pending="1"` then in-place swap; only brand-new rows use `.fresh` fadein (bug #52).

## AI summary loop (FoundationModels)

`TopicSummarizer` actor; fresh `LanguageModelSession` per 60 s cycle (no stale history). Gate triggers on `inflightChunks.isEmpty && >=1.5s idle` since last sentence — ANE is single-tenant, ASR (`provider="coreml"`) preempts otherwise (bug #46). Free-text generation (no `@Generable` — macro plugin unavailable in CLI builds); parse `Topic:` / `Summary:` lines.

## Pruning

Background task once/s drops sentences older than `maxAgeSeconds=300` (protects last sentence). Hard cap `maxSentenceCount=50` enforced every `graduate()`.

## Per-run output

Work dir `NSTemporaryDirectory()/livetranslate-<stamp>/`. Zipped to `~/Documents/LiveTranslate/<stamp>.zip` via `/usr/bin/zip -j -q -X`. Ships only `.jsonl` transcript + `.mkv` (if ffmpeg present). WAVs/SRTs are intermediates. `CrashRecovery` re-runs leftover dirs on next launch.

## Latency budget

| Path | Typical | Dominant |
|---|---|---|
| Speech → partial transcript | ~50–90 ms | Sherpa decode (CoreML/ANE) |
| Speech → partial translation | +0–300 ms | 0.3 s throttle |
| Speech → final transcript/translation | ~1.3–1.9 s | 1.2 s endpoint + ~150 ms translate |
| Speech → OBS German | + ~100 ms | Same endpoint |

Knobs: `endpointSilenceSeconds=1.2`, `rule2SilenceSeconds=1.8`, `maxUtteranceSeconds=60`, partial throttle 0.3 s, mic tap 512 samples, RNNoise 10 ms.

Watch startup: `SherpaTranscriber: recognizer loaded (provider=coreml)` — if it says `provider=cpu`, decode ~2× (bug #23).

## Files

| File | Role |
|---|---|
| App.swift | `@main`; NSWindow (`.statusBar` level, translucent, all-spaces); MenuBarExtra; `WindowAccessor`; willTerminate flush; `CrashRecovery` at launch |
| TranscriptView.swift | UI: sentence list only; `.translationTask` parked via `AsyncStream<Never>`; `displayRows` merge; `TranscriptRow`; `StreamShareView`+QR |
| TopicSummarizer.swift | actor; FoundationModels free-text; fresh session/cycle; `isAvailable` checks `SystemLanguageModel.default.availability` |
| Pipeline.swift | `@MainActor` orchestrator; owns sentences/inflight/cache/timers/tts/server/obsTranslator; `applyLifecycle` state machine; `isActive` gate |
| SourcePipeline.swift | Per-stream; concurrent recording+recognition async-lets; drains SessionSnapshot but ignores |
| Types.swift | SourceLocale, TargetLanguage, SourceTag, InflightChunk(+State), Sentence, PipelineStatus, Session*; protocols |
| ModelConfig.swift | Compile-time constants for language pair + model paths |
| SherpaTranscriber.swift | OnlineRecognizer once (NSLock); per-call OnlineStream; accumulator+worker; energy/ZCR VAD; chunkStart/lastVoicedEnd; normalizeHypothesis; sentenceBoundary; crosstalk lock |
| OnnxTTSSpeaker.swift | kitten-mini ONNX; lazy load; serial queue; batch up to 5; drop-oldest; 24 kHz PCM16 LE; voice `sid=2` |
| LiveAudioServer.swift | NWListener:8765; routes /,/live.wav,/events,/obs,/obs-events; SSE; heartbeat; subscriber dicts |
| TranscriptArchive.swift | Per-run JSONL; sorted keys; shared encoder with SSE |
| MergedSubtitleArchive.swift | Per-language SRT; atomic rewrite on add |
| SubtitleArchive.swift | Per-(source,lang) SRT — retained but currently NOT instantiated |
| BufferBroadcaster.swift | Fan-out; `stream` returns FRESH AsyncStream per access; `finishAll()` drives graceful drain |
| MicrophoneSource.swift | AVAudioEngine; permanent tap (install once); 48k mono Float32 |
| SystemAudioSource.swift | SCK audio-only (2×2 video required); 48k stereo→mono; `CMSampleBufferCopyPCMDataIntoAudioBufferList` (bug #13); reconnect w/ backoff on system stop (bug #38) |
| DenoisingAudioSource.swift | Wraps source; optional RNNoise+AGC+muteWhen; idempotent `start()` (bug #45) |
| RNNoiseProcessor.swift | Wrapper; 480-sample frames; ±32768 scale; 10 ms latency |
| AppleTranslator.swift | `@MainActor`; holds injected TranslationSession |
| AudioRecorder.swift | AVAudioFile 48k Int16; `flush()` must nil file to finalize header (bug #31) |
| Paths.swift | enum Paths; Outputs struct; `shippedFiles` = jsonl+mkv |
| MKVExporter.swift | ffmpeg shell-out; 640×360 lavfi video; amix WAVs; embed SRTs with ISO-639-3; ZipArchiver |
| CrashRecovery.swift | Scans temp dirs; re-exports MKV+zip; idempotent |
| Log.swift | `/tmp/livetranslate.log`; truncate >5 MB; serial queue |
| AppSettings.swift | `@Published`+UserDefaults; webPayload; overlayBackgroundColor dynamic |
| OptionsView.swift | Settings scene; typography/colors/layout/AI/window opacity |
| SummaryView.swift | Standalone window; auto-open/dismiss via SummaryWindowController; WindowConfigurer NSViewRepresentable |

## SDKs in use

AVAudioEngine, AVAudioConverter, Accelerate (vDSP), ScreenCaptureKit, sherpa-onnx (C API via CSherpaOnnx bridge — `OnlineRecognizer` zipformer + `OfflineTts` kitten-mini), Translation (`TranslationSession`+`.translationTask`), Network (NWListener/NWConnection), CoreImage (QR), FoundationModels (LanguageModelSession), SwiftUI.

## Roadmap (status)

Done: streaming RNN-T DE, energy/ZCR VAD, sentence splitting (force/endpoint/stream-end), partial translations, normalizeHypothesis, UUID continuity, DisplayRow, live LAN stream w/ hypothesis SSE, OBS overlay (with speech-bubble indicator), Options window, TTS gating, crosstalk gate, per-run zip, CrashRecovery, FoundationModels summary loop, multi-segment screen recording.

Open: speaker diarization (campplus scaffolded, not active), retargeting docs, per-app SCK filter, global hotkey, click-through overlay.
