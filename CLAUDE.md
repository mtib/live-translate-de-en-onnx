# LiveTranslate — context for Claude

A minimal, no-Xcode macOS app that does on-device speech transcription and
translation from one or two audio sources at the same time. A learning / DIY
clone of [transcrybe.app](https://transcrybe.app).

> **Process rule for future edits**
>
> Any meaningful change to source layout, data flow, protocols, settings,
> or runtime behavior **must be reflected in this file in the same commit**.
> If you change `Sentence`'s shape, the pipeline order, the permissions
> needed, the build script, or any of the "Things that have bitten us"
> entries: update CLAUDE.md.
>
> **Persist learnings.** When a bug bites — a race, an actor-isolation
> surprise, a confused-by-the-API moment — add a numbered entry to
> the "Things that have bitten us" section explaining what went wrong
> AND the shape of the fix. This is the durable institutional memory;
> without it the same trap will be re-stepped in a later session.
>
> The reason: this file is the only durable orientation document. Source
> comments cover *what* a function does; CLAUDE.md covers *why the design
> is shaped this way* and *what to never do again*. Future sessions read
> this first.

> **Eagerly load the Swift sources at session start**
>
> Before changing any code in this project, read **all** of
> `Sources/LiveTranslate/*.swift` and the relevant bridge headers. The
> data flow crosses several files (audio source → mixer → denoiser →
> transcriber → pipeline → translator → archives + UI), and surprising
> interactions live at the boundaries. Skimming or grepping for one
> symbol misses the patterns. Read everything first, *then* edit.

> **Always build with the signing identity**
>
> The user keeps `export LIVETRANSLATE_SIGN_IDENTITY=LiveTranslateDev`
> in `~/.zshrc` so their interactive shell sessions sign with a stable
> self-signed cert (keeps TCC grants across rebuilds). Non-interactive
> `bash` invocations — including the agent's Bash tool — DO NOT source
> `.zshrc`, so without the variable set explicitly every build the
> agent triggers is ad-hoc-signed and re-prompts for mic + screen
> recording. Always prefix builds the agent runs with the env var:
>
> ```sh
> LIVETRANSLATE_SIGN_IDENTITY=LiveTranslateDev ./build.sh
> ```

## How it's built

- **No `.xcodeproj`.** Pure SwiftPM. Built with Command Line Tools
  (`/Library/Developer/CommandLineTools`). No Xcode required.
- `swift-tools-version: 6.0`, but the executable target is pinned to
  `.swiftLanguageMode(.v5)` because the Translation APIs are awkward
  under Swift 6 strict concurrency.
- `./build.sh` first runs `tools/download-sherpa.sh` (idempotent —
  downloads sherpa-onnx v1.13.2 osx-arm64 shared dylib into
  `external/sherpa-onnx/` and all ONNX models into
  `build/sherpa-models/`), then `swift build -c release`, then wraps
  the binary into `build/LiveTranslate.app/`, copies the dylibs into
  `Contents/Frameworks/`, copies models into `Contents/Resources/`,
  and codesigns (ad-hoc by default; set `LIVETRANSLATE_SIGN_IDENTITY`
  to a self-signed cert name to persist TCC grants across rebuilds).
- **Language pair is compile-time.** Default is `de` → `en` (German →
  English). Change `ModelConfig.sourceLanguage` / `targetLanguage` and
  swap the ASR model to retarget. No runtime picker.
- **Always launch via `open build/LiveTranslate.app`** — never run the
  binary directly. TCC associates permission grants with the bundle,
  not the executable path; direct exec leads to the system thinking
  the usage-description keys are missing.

## Architecture

```
  Mic ──▶ Denoise ──▶ SourcePipeline(mic) ──┐
                       (recorder, SRTs)      │   chunk lifecycle
                                             ├──▶ Pipeline.applyLifecycle
  System ─▶ Denoise ─▶ SourcePipeline(sys) ──┤        │
                       (recorder, SRTs)      │        ▼
                                             │   @Published inflightChunks
                                             │        │  on .completed
                                             │        ▼
                                             │   Translator (async, cached)
                                             │        │  on result
                                             │        ▼
                                             └──▶ @Published sentences
                                                      │  on prune/drop
                                                      ▼
                                                  TranscriptArchive (.jsonl,
                                                  source-tagged) +
                                                  per-source SubtitleArchives
```

Both streams share one `SherpaTranscriber`. The UI sees in-flight
chunks as reserved rows that flip through `.listening → .transcribing
→ .translating`, then graduate to a `Sentence` with the **same UUID**
— so SwiftUI row identity stays stable across the lifecycle.

### Key design decisions

- **Per-stream pipelines, never mixed.** Mic and system are captured
  in parallel; each goes through its own `RNNoise` (via
  `DenoisingAudioSource`), an `AudioRecorder` writing
  `<stamp>.<source>.wav`, and a `SherpaTranscriber.transcribe(audio:locale:source:)`
  call. Per-`(source, language)` SRT writers archive sentences as they
  drop. The transcribers share one sherpa-onnx recognizer (loaded once
  on first call); each `transcribe()` invocation creates its own stream
  on the recognizer, so mic and system run in parallel without locks.
- **Inflight-chunk UI model.** Every chunk reserves a UI row at voice
  onset (state `.listening`). The state then progresses through
  `.transcribing` and (if translation is needed) `.translating` as the
  pipeline advances. On translation completion the chunk graduates to
  a `Sentence` with the **same UUID** as the inflight row, so SwiftUI
  animates a smooth content swap in place rather than removing one row
  and adding another. Chunks whisper rejects (no voice / under 1 s / 0
  segments) fire `.dropped` and the row collapses out.
- **Lifecycle callback over snapshot stream.** `WhisperCppTranscriber`
  emits `onChunkLifecycle(chunkID, source, .listening | .transcribing
  | .completed(text) | .dropped)` from background tasks. Pipeline
  hops to MainActor in `handleChunkLifecycle` and runs the state
  machine in `applyLifecycle`. The legacy `transcribe()` AsyncStream
  return value (a stream of `SessionSnapshot`) is drained but
  ignored — the lifecycle callback is the single source of truth.
  This is what lets translation (per-chunk, async) fit cleanly into
  the same state machine.
- **Audio format invariant.** Both sources standardize on **48 kHz
  mono Float32** (RNNoise's native rate). The transcriber downsamples
  to 16 kHz internally for the recognizer; each `.wav` writer downcasts to
  16-bit Int on the
  write path.
- **RNNoise per stream.** A vendored copy of xiph/rnnoise v0.1.1
  (BSD 3-clause, GRU weights embedded in `rnn_data.c`, ~400 KB
  static, zero runtime dependencies) runs inside `DenoisingAudioSource`
  — one instance per upstream (mic, system). Each wraps the raw
  source's broadcaster and re-emits denoised buffers. RNNoise wants
  ±32768-scaled Float32 in 480-sample frames at 48 kHz — the wrapper
  (`RNNoiseProcessor`) buffers arbitrary input sizes and handles the
  scale conversion. Algorithmic latency: 10 ms.
- **sherpa-onnx streaming RNN-T is the transcriber.** `SherpaTranscriber`
  downsamples the 48 kHz post-RNNoise stream to 16 kHz (AVAudioConverter),
  feeds samples into both a Silero-VAD instance (for voiced/silence
  gating) and a sherpa-onnx streaming zipformer recognizer. When the
  recognizer fires its built-in endpoint (≥1 s trailing silence, rule 1),
  the committed text is read BEFORE resetting the stream, then a
  `SpeakerTracker` assigns a `[Speaker N]` label via campplus embedding +
  cosine-sim clustering. The labeled text is emitted as `.completed`.
- **ONNX models: bundled only.** All models are copied into
  `Contents/Resources/` by `build.sh` (downloaded by
  `tools/download-sherpa.sh`). Language pair is a compile-time
  constant in `ModelConfig.swift` — change `sourceLanguage`,
  `targetLanguage`, and the `asrModel*` path constants and rebuild.
  No runtime override; what you built is what you run.
- **Auto-gain control.** `DenoisingAudioSource` runs a per-instance
  envelope-follower AGC after RNNoise and before the crosstalk gate:
  measure RMS via `vDSP_measqv`, EMA the input level on voiced
  buffers, target ~0.1 RMS, smooth the applied gain (slow ramp to
  avoid pumping), multiply via `vDSP_vsmul`. Caps at 8× boost; never
  attenuates (`agcMinGain=1`). This lets mic and system arrive at
  comparable loudness without manual tuning.
- **Transcribers emit one sentence per closed turn.** The transcriber
  owns turn boundaries (via sherpa-onnx endpoint detection) and speaker
  labeling. Pipeline never edits a `Sentence` in place; ingest just
  appends.
- **Translation is per-chunk, inline.** When `.completed(text)` fires
  in `applyLifecycle`, Pipeline either graduates immediately (src ==
  tgt language, or cache hit) or sets the inflight row to
  `.translating(text)` and dispatches a `Task { @MainActor in
  translator.translate(text) }`. Explicit `@MainActor` on the task is
  load-bearing — without it, Swift 5's actor-inheritance heuristics
  let the post-await `graduate` run off-actor on some paths, and
  `@Published` mutations from the wrong actor don't surface in the UI.
- **Translation cache.** `Pipeline.translationCache: [String: String]`
  keyed by source text. Identical strings during a run reuse the
  cached translation. LRU-ish eviction at 200 entries.
- **Live translated-audio HTTP stream (optional, opt-in by capability).**
  When the run starts, Pipeline checks `OnnxTTSSpeaker.isAvailable()`.
  If the kitten-mini model is bundled and src != tgt language, it
  spins up `LiveAudioServer` on port 8765 and an `OnnxTTSSpeaker` that
  synthesizes each graduated sentence's translation into the stream. The
  UI shows a small share icon (radio-waves SF Symbol) in the bar
  while this is active; clicking it pops a panel with the URL and a
  QR code so a phone on the same Wi-Fi can listen. Wire format: 24 kHz
  mono PCM16 LE, served as an open-ended WAV (`0xFFFFFFFF` chunk sizes).
  200 ms heartbeat broadcasts 50 ms of silence when idle so VLC doesn't
  tear the socket down.
- **Pruning.** Non-protected sentences whose `lastModified` is older
  than 5 minutes get dropped once per second. Hard cap at **50**
  retained — generous so the user can scroll back through history.
  "Protected" means just the most-recent sentence (so the UI is never
  briefly empty mid-stream).
- **Per-run output: temp dir then zip.** Each session writes into a
  fresh temp working directory (`NSTemporaryDirectory()/livetranslate-<stamp>/`).
  All artifacts (per-source WAVs, per-source SRTs, live-merged
  per-language SRTs, JSONL log) land there immediately as they're
  produced. At Stop the MKV is built in-place (ffmpeg, if available),
  then `/usr/bin/zip` packs the directory into
  `~/Documents/LiveTranslate/<stamp>.zip` and the temp dir is
  deleted. Layout inside the zip:

  ```
  <stamp>/
      <stamp>.jsonl                       ← every sentence (source-tagged)
      <stamp>.mic.<src>.srt               ← per-source SRTs, source language
      <stamp>.mic.<tgt>.srt
      <stamp>.system.<src>.srt
      <stamp>.system.<tgt>.srt
      <stamp>.<lang>.srt                  ← merged SRT per language ([Mic]/[Sys])
      <stamp>.mic.wav                     ← post-denoise + AGC audio
      <stamp>.system.wav
      <stamp>.mkv                         ← 640×360 black + amix audio + SRTs
  ```

  SRTs are written live during the session — both per-source files
  (one cue per sentence as it graduates) and the merged ones
  (in-memory re-sort + atomic rewrite on each graduate). MKVExporter
  consumes the already-merged files; it doesn't re-merge. If ffmpeg
  isn't installed, no `.mkv` is produced and the zip just contains
  the audio + SRTs + JSONL.

  The transcript line shape:
  ```json
  {"end":"2026-05-16T22:13:09.581Z","start":"2026-05-16T22:13:07.123Z","transcription":"…","translation":"…"}
  ```
  Keys sorted for grep/diff stability, ISO-8601 timestamps with
  fractional seconds. `start` / `end` are derived from the chunk's
  voiced span in the audio stream (anchored to `runStartedAt`), so
  they line up sample-accurately with the matching position in the
  paired `.wav`. The SRT cue uses the same offsets.
  `transcription` / `translation` always present; `translation` may be
  empty if the translator hadn't gotten to it yet.

  The `.wav` is 48 kHz mono signed-16-bit linear PCM (AVAudioFile auto-
  converts our Float32 buffers on the write path). Recording is taken
  **after** RNNoise, so one sample = exactly what the recognizer heard
  — audio and transcript line up. Paths are centralised in `Paths.swift`.

### Files

| File | Role |
|---|---|
| `App.swift` | `@main` entry. SwiftUI `Window` scene. Configures the NSWindow for floating / translucent / movable-from-background behavior. |
| `TranscriptView.swift` | The whole UI. Renders one `SentenceRow` per sentence, with opacity fade for older rows. Hosts `.translationTask` (the only way to get a `TranslationSession`). Background is a flat translucent color — no blur. |
| `Pipeline.swift` | `@MainActor ObservableObject` orchestrator. Owns the shared `sentences` array, the JSONL archive, the translator, and the prune loop. Spawns one `SourcePipeline` per `SourceTag` and merges their `Sentence` streams into the visible array. Persists user settings via UserDefaults. |
| `SourcePipeline.swift` | Self-contained per-stream pipeline (mic OR system). Owns its denoised audio source, recorder, source/target SRT writers, and a `transcribe()` call. Emits `Sentence`s via an `AsyncStream` that `Pipeline` consumes. |
| `BufferBroadcaster.swift` | Helper that fans audio buffers out to any number of subscribed `AsyncStream`s. `finishAll()` ends every subscription on audio-source stop, which is what drains the recognition pipeline naturally. |
| `Types.swift` | `SourceLocale`, `TargetLanguage`, `SourceTag` (mic/system), `Sentence`, `PipelineStatus`, `SessionSentence` / `SessionSnapshot`. Protocols: `AudioSource`, `Transcriber`, `Translator`. |
| `MicrophoneSource.swift` | `AVAudioEngine` mic capture, emits 48 kHz mono Float32. |
| `SystemAudioSource.swift` | `ScreenCaptureKit`-based system audio capture, emits 48 kHz mono Float32. |
| `DenoisingAudioSource.swift` | Wraps any `AudioSource`, applies its own `RNNoiseProcessor`, re-broadcasts. One per input stream so denoiser state is independent. |
| `RNNoiseProcessor.swift` | Swift wrapper around the vendored RNNoise C library. Owns the `DenoiseState`, buffers arbitrary-sized input into 480-sample frames, handles ±32768 ↔ ±1 scaling, emits denoised samples via `drain(into:count:)`. |
| `CRNNoise/` | Vendored xiph/rnnoise v0.1.1 as a SwiftPM C target. BSD 3-clause; GRU weights statically linked. See `Sources/CRNNoise/README.md`. |
| `SherpaTranscriber.swift` | **The transcriber.** Two structured-concurrency child tasks (accumulator + worker) per `transcribe()` call. Accumulator resamples 48→16 kHz, feeds Silero-VAD + sherpa-onnx streaming recognizer, fires endpoint events. Worker reads committed text, runs speaker embedding, emits `.completed([Speaker N] text)`. |
| `SpeakerTracker.swift` | campplus speaker embedding extraction + cosine-sim clustering. One instance per audio stream; `label(for:)` synchronously returns "Speaker N". |
| `ModelConfig.swift` | Compile-time constants: source/target language codes, ONNX model paths relative to `Contents/Resources/`. |
| `CSherpaOnnx/` | SwiftPM C bridge target: `c-api.h` header + stub `.c` file. Linker flags point at `external/sherpa-onnx/lib/`. |
| `AppleTranslator.swift` | Holds a `TranslationSession` that the View injects via `Pipeline.installTranslationSession(_:)`. |
| `TranscriptArchive.swift` | One-per-run JSONL archive. Rows carry a `source` field (`"mic"` / `"system"`) plus the audio-anchored `start`/`end` timestamps. |
| `AudioRecorder.swift` | One-per-stream `.wav` writer fed by a parallel consumer of its source's broadcaster. 48 kHz mono Int16. |
| `SubtitleArchive.swift` | One-per-`(source,language)` SRT writer. Cue times are offsets into the matching `<stamp>.<source>.wav`. |
| `Paths.swift` | Single source of truth for `~/Documents/LiveTranslate/{transcripts,recordings}/<stamp>.<source>[.<lang>].{wav,srt}` plus the shared `<stamp>.jsonl`. |
| `Log.swift` | Append-only file logger at `/tmp/livetranslate.log`. Truncates on launch if > 5 MB. |
| `OnnxTTSSpeaker.swift` | Synthesizes finalized translations via kitten-mini ONNX TTS to 24 kHz PCM16 LE. Serial queue, drops oldest past 5. `isAvailable()` checks for the model file in the bundle. |
| `LiveAudioServer.swift` | Hand-rolled HTTP/1.1 server on a `NWListener` that streams 24 kHz mono PCM16 LE WAV to any client. Header advertises `0xFFFFFFFF` data size → "read until close" (works in VLC / mpv / iOS Safari / Chrome). 200 ms heartbeat task pushes 50 ms of silence when idle to keep VLC's socket alive. |

## Key behaviors / non-obvious bits

### One turn = one sentence

`SherpaTranscriber`'s accumulator feeds audio into the sherpa-onnx
streaming recognizer continuously; when the recognizer fires an endpoint
(≥1 s trailing silence) the committed hypothesis is read, labeled with
a `[Speaker N]` prefix, and emitted as a single sentence. The Pipeline
gets one `SessionSentence` per closed turn, which lands as one `Sentence`
row, which writes one JSONL line and one SRT cue.

The transcriber owns sentence segmentation. The Pipeline never splits,
the Pipeline never edits-in-place.

### Audio-stream timing (SRT/JSONL ↔ WAV alignment)

The accumulator tracks a cumulative 16 kHz sample counter
(`samplesEverEmitted16k`) across all chunks; the chunk's own start
offset is the counter's value at chunk-open. From there the
sentence's `startSeconds` / `endSeconds` come from
`(chunkStartSample16k + voiceStart) / 16_000` etc. — i.e. **seconds
into the audio stream**.

Pipeline anchors those at `runStartedAt`, so `Sentence.createdAt /
endsAt` are wall-clock Dates but the **offsets between them and
`runStartedAt`** match audio-stream positions. The recorder consumes
the same audio broadcaster, so audio-stream position = WAV position.
SRT cues therefore line up sample-accurately with the `.wav` and the
JSONL `start`/`end` ISO timestamps are usable as audio offsets.

Backends that don't report timing pass `nil` for `startSeconds` /
`endSeconds` and Pipeline falls back to `Date()` at ingest.

### Concurrent accumulator + worker

Speaker embedding takes a few hundred ms. If we ran it synchronously
in the audio pump, audio would accumulate in the broadcaster unread
during that time.

The fix is two structured child tasks under one `async let`:

- **Accumulator** reads audio forever, feeds the streaming recognizer,
  emits a `TurnRecord` (text + samples) on each endpoint.
- **Worker** drains the `TurnRecord` queue, computes speaker embedding,
  fires lifecycle events. Sees turns in order.

The queue is unbounded; backpressure isn't a concern at our rates.

### sherpa-onnx text must be read before Reset()

The streaming recognizer accumulates a committed hypothesis in `stream`.
After an endpoint, the hypothesis is read with
`SherpaOnnxGetOnlineStreamResult(recognizer, stream)`, **then** the stream
is reset with `SherpaOnnxOnlineStreamReset`. Reading after Reset would
return an empty string. The accumulator does this in the right order.

### Per-source crosstalk suppression (speaker bleed into mic)

The mic always picks up some of what the system is playing through
the speakers. To stop those phantom transcriptions on the mic side,
`WhisperCppTranscriber` carries shared state — `lastSystemVoicedAt`
(`Date`, NSLock-protected) — updated by the system accumulator on
every voiced buffer. The mic accumulator queries it per buffer; if
system was voiced within `crosstalkPersistSeconds` (250 ms, covers
RNNoise envelope follower lag + room reverberation), the mic's
current buffer is replaced with silence in the 16 kHz sample array
AND the voiced-span markers aren't credited. The chunk's timing keeps
advancing (so WAV alignment stays correct) but no false-positive
voice gets attributed to the mic during system playback.

Note: this affects only what's fed to whisper. The mic `.wav`
recording still contains the raw bleed — that's a separate concern
(the WAV is recorded from the broadcaster, upstream of the
transcriber's per-source muting logic).

### Broadcaster pattern (the "won't restart after Stop" problem)
`AsyncStream` is single-consumer. The previous design exposed a single
stored AsyncStream as `buffers` — when a second consumer tried to read
from it after the first iterator was gone, it got no data. Both
`MicrophoneSource` and `SystemAudioSource` now build a fresh AsyncStream
per `buffers` access and fan tap callbacks out to all current subscribers.

### Translation framework quirks
- A `TranslationSession` is **only** obtainable via SwiftUI's
  `.translationTask` modifier. There is no public way to create one
  programmatically. We work around this by parking the modifier's closure
  on `Task.sleep(.max)` and stuffing the session into `AppleTranslator`
  for the Pipeline to use.
- First time a language pair is used, macOS prompts to download translation
  models. The user must accept. The download can be triggered ahead of time
  via **System Settings → Apple Intelligence & Siri → Translation Languages**
  or by opening the Translate app once.
- `Configuration` source/target use bare language codes (`"de"`, `"en"`),
  not full BCP-47 (`"de-DE"`). We trim the region in `translationConfig`.

### Persisted settings

User-facing settings are stored in `UserDefaults` and restored on launch:
`translateEnabled`, `source` (BCP-47 locale), `target` (language code +
display name). `compactMode` is stored separately via `@AppStorage`
because it's a pure View concern. Mic-on / system-on are no longer user
settings — both are always captured.

### Permissions
The bundle declares:
- `NSMicrophoneUsageDescription`
- `NSScreenCaptureUsageDescription` (for system audio via SCK)

Mic prompts via `AVCaptureDevice.requestAccess`. Screen recording
prompts when `SCStream.startCapture()` runs the first time. No speech
recognition permission — sherpa-onnx runs locally against bundled ONNX
models and doesn't touch Apple's Speech APIs.

Reset stale grants with:
```sh
tccutil reset Microphone local.mtib.livetranslate
tccutil reset ScreenCapture local.mtib.livetranslate
```

**Persisting grants across rebuilds.** Ad-hoc signing (`codesign
--sign -`, the default) produces a fresh cdhash each build → TCC
re-prompts. Set `LIVETRANSLATE_SIGN_IDENTITY` to a self-signed
code-signing certificate name (created via Keychain Access →
Certificate Assistant) and `build.sh` will use it. TCC keys grants
on the certificate identity rather than the binary hash, so future
builds reuse the existing grant. See README for the one-time setup.

### Window
- Real macOS app (not menu-bar). `LSUIElement = false`.
- Translucent (`NSVisualEffectView.Material.hudWindow`), floating
  (`NSWindow.level = .floating`), movable from anywhere
  (`isMovableByWindowBackground = true`), persists across Spaces
  (`canJoinAllSpaces`).
- Compact mode (`@AppStorage("compactMode")`) hides the controls and just
  shows the sentence list — useful as a slim hover overlay.
- **No in-window Quit / Copy / Clear buttons** — use the native macOS
  quit (Cmd+Q / app menu) and select text in a row to copy. The archive
  file is the durable record; no manual export needed.

## Tools / SDKs in use

- `AVAudioEngine`, `AVAudioConverter` — mic capture + sample-rate conversion
- `Accelerate` (`vDSP_vadd`) — SIMD per-sample sum in `MixedAudioSource`
- `ScreenCaptureKit` — system audio capture
- `Speech` (`SFSpeechRecognizer`, `SFSpeechAudioBufferRecognitionRequest`)
- `Translation` (`TranslationSession`, `.translationTask`)
- SwiftUI

## Roadmap (rough)

- [ ] Per-app audio capture (instead of whole-machine) via SCK's filter
- [x] RNNoise denoising on the merged stream (vendored, BSD 3-clause)
- [x] sherpa-onnx streaming RNN-T (zipformer) replaces whisper.cpp
- [x] Silero-VAD endpoint detection
- [x] campplus speaker diarization (`[Speaker N]` labels)
- [x] kitten-mini ONNX TTS replaces AVSpeechSynthesizer
- [ ] Speaker name assignment / correction UI
- [ ] OpenRouter fallback as an alternative `Translator` impl
- [ ] Global hotkey to start/stop
- [ ] Click-through floating overlay mode
- [ ] Persist transcript history

## Build / run / debug commands

```sh
./build.sh                                       # build & bundle
open build/LiveTranslate.app                     # launch (always via `open`!)
tail -f /tmp/livetranslate.log                      # see log output
pkill -f LiveTranslate                           # kill all instances
```

## Things that have bitten us already

1. **Running the binary directly** (not via `open`) loses bundle context,
   TCC complains about missing usage-description keys, app crashes on
   first permission request.
2. **Reinstalling the audio tap** between recognition sessions caused
   recognition to silently stop working after ~1 minute. Keep the tap
   permanent.
3. **`requiresOnDeviceRecognition = true`** hard-fails when the language
   model isn't installed yet. We set it to `false` so the system can fall
   back to cloud if needed.
4. **`NSLog`** doesn't reliably appear in `log show` for ad-hoc-signed
   apps on macOS 26. Use `Log.line(_:)` → `/tmp/livetranslate.log` instead.
5. **Command Line Tools don't ship XCTest or Swift Testing.** No `swift test`
   support without installing full Xcode. Tests deliberately omitted.
6. **Single-consumer AsyncStream** silently breaks every Start after the
   first one. Audio sources must broadcast to per-subscriber streams.
7. **Stalled when audio plays out the speakers and the mic source is on.**
   The recognizer choked on speaker bleed + room noise. Fix: use the
   System Audio source instead (ScreenCaptureKit).
8. **Index-only snapshot reconciliation** left orphan rows whenever the
   recognizer revised away a sentence boundary. Always handle the "snapshot
   shrunk" case explicitly.
9. **`DispatchSemaphore.wait()` on the MainActor to block on an async
   operation that itself hops to MainActor is an instant deadlock.** The
   first version of `SystemAudioSource.start()` did this; the app froze on
   Start when system audio was enabled. Rule: never block the main thread
   with a semaphore for async work. `AudioSource.start()` is now `async
   throws` so backends can implement it natively.
10. **"Don't drop active-session sentences" was too aggressive.** The
    original `prune` / `enforceMaxCount` exempted every sentence in the
    active recognition session — and since a session can run for ~60
    seconds emitting many sentences, the list kept growing forever. Only
    the *live* (last-active) sentence per source needs protection.
11. **Hoisting `let audio = source.buffers` out of the recognition-cycle
    while-loop** silently broke session restarts: `AsyncStream` is
    single-consumer, so the second session's pump task iterated an
    already-drained stream and the recognizer hit "No speech detected".
    Rule: `buffers` is a *fresh subscription factory* — call it per
    consumer, never cache.
12. **Unstructured `Task { ... }` children inside a cancellable parent
    don't inherit cancellation.** The original `run()` spawned its
    translation/prune workers and per-source recognition cycles as
    independent Tasks, then awaited their `.value`. When the parent
    Task was cancelled (via `Pipeline.stop()`), the await woke up but
    the child Tasks kept running independently — recognition continued
    producing transcripts, and a second Stop press would actually start
    a *fresh* run on top (creating a duplicate JSONL archive). Fix:
    spawn children inside `withTaskGroup` so cancellation cascades.
    Related rule: don't `runTask = nil` inside `stop()` — leave it set
    so `toggle()` no-ops during the wind-down rather than starting a
    new run on top.
13. **`CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer` with
    `bufferListSize: MemoryLayout<AudioBufferList>.size` only fits ONE
    AudioBuffer.** ScreenCaptureKit delivers non-interleaved stereo
    Float32 — two separate AudioBuffers — which made the call fail with
    `kCMSampleBufferError_ArrayTooSmall` on *every* sample. Diagnostic
    counters (`SystemAudio: heartbeat received=X yielded=0 convFails=X`)
    were the giveaway. Fix: use
    `CMSampleBufferCopyPCMDataIntoAudioBufferList(_:at:frameCount:into:)`
    with `AVAudioPCMBuffer.mutableAudioBufferList` — the destination
    is already correctly sized for its format (separate buffers for
    non-interleaved, one for interleaved).
14. **Apple Speech serializes recognition tasks per-app — not just
    on-device.** Two concurrent `SFSpeechRecognizer`s preempt each
    other on every restart, both fast-failing with "No speech detected"
    within ~0.3 s. Forcing one to the server (via
    `recognizer.supportsOnDeviceRecognition = false`) does NOT help —
    the contention is at the recognition-task level, not the on-device
    model level. The only fix that works is to send a **single mixed
    audio stream to one recognizer**. We did that via `MixedAudioSource`.
    Trade-off: source attribution is lost (we removed `SentenceKind`
    and color-coding from the UI as part of this).
15a. **`SFTranscriptionSegment.timestamp` / `.duration` are zero on
    partial results.** Apple only populates them on final results.
    Pause-based sentence splitting in `splitIntoSentences` therefore
    only fires when a recognition session ends (~60 s on-device, sooner
    on errors/restarts) — at that point the text gets retroactively
    re-split using the gaps. During a live session only punctuation
    splits fire. Confirmed by dumping `[timestamp+duration substring]`
    for every snapshot; partials looked like
    `[0.00+0.00 Hello] [0.00+0.00 world]…` until the final result came
    in with real timings. No fix from our side; just a known limit.
15. **Naive "interleave buffer streams" mixing tanked recognition
    latency.** Forwarding every upstream buffer as it arrived doubled
    the recognizer's audio-time-to-wall-time ratio (it received ~2 s of
    audio per real second). Transcription content was correct but
    emission lagged badly. Fix: mix at the SAMPLE level — mic clocks
    the output, each mic buffer produces one summed output buffer,
    system samples are pulled from a small bounded queue and added per-
    sample. 1:1 audio-to-wall ratio restored, recognition is instant
    again. The per-sample sum runs through `vDSP_vadd` (Accelerate) on
    a reusable `UnsafeMutablePointer<Float>` scratch buffer — SIMD on
    NEON / AVX, no per-call malloc. **Mixing has since been removed
    entirely** in favour of independent per-stream pipelines (see #18);
    this lesson is kept for the audio-clocking principle.
16. **Whisper silently drops audio under ~1 s.** Its mel-spectrogram
    threshold is 100 frames at 10 ms each. The symptom was chunks
    coming back with `segments=0` in 0.01 s — no error, just empty.
    Two-layer defence: (a) `accumulator` only allows silence-close
    once the *total* chunk length clears 1.1 s; (b) `processChunk`
    pads short trimmed clips with trailing zeros up to 1.1 s as a
    final safety net. Gating on trim length instead of total length
    was a separate bug — short utterances kept the trim small and
    silence-close never fired, leading to max-chunk drops.
17. **Cancelling `runTask` aborts the recognition mid-flight and
    drops trailing audio.** The old `Pipeline.stop()` cancelled the
    run Task; that cancellation propagated to the accumulator's
    `for await buf in audio`, which exits without emitting a final
    in-flight chunk, and to the worker's `for await chunk in queue`,
    which exits before draining. Anything still mid-utterance when
    Stop is pressed was lost. Fix: shutdown is driven by *ending the
    audio source*, not cancelling the task. Each `AudioSource.stop()`
    calls `BufferBroadcaster.finishAll()` to close its subscriptions,
    so the recognition pipeline drains naturally:
    `audio source closes → accumulator's for-await ends → final chunk
    emitted → queue closed → worker drains → run() exits`.
    Background workers (translation, prune) still need cancellation
    because they have `while !Task.isCancelled` loops with no
    natural termination — they run in a separate `Task` cancelled
    *after* the audio path drains.
18. **Per-stream pipelines instead of mixing.** The old design
    sample-summed mic + system before transcribing (lesson #14), then
    later denoised the mix (lesson #15-era), losing source attribution.
    The current design runs each input through its own
    `DenoisingAudioSource`, its own `SourcePipeline` (recorder + SRT
    writers + transcribe call), and only shares the whisper context
    (with an `NSLock` around `whisper_full`), the JSONL archive (with
    a per-row `source` field), and the visible UI sentence array.
    No mixing, no attribution loss.
19. **Two concurrent `whisper_init_from_file_with_params` calls fail
    on Metal contexts.** When mic and system pipelines both invoke
    `transcribe()` at the same instant, both reach
    `ensureContextLoaded()` and both see `ctx == nil`. Both then call
    `whisper_init_from_file_with_params` on the same path; one
    succeeds, the other fails with "failed to load model". Symptom in
    the log: `Whisper.transcribe[system]: error … failed to load
    model`. Fix: serialize `ensureContextLoaded()` with an `NSLock`
    so the second caller waits, finds `ctx` already set, and reuses
    it. Without this the system pipeline never produced chunks (and
    crosstalk suppression never activated because the system
    accumulator was never running).
20. **Crosstalk: mic always picks up some of the system's audio
    through the speakers.** Affects transcription quality (mic
    transcribes the bleed). Mitigation: `SherpaTranscriber`
    carries a shared `lastSystemVoicedAt: Date` (NSLock-protected);
    system accumulator stamps it per voiced buffer; mic accumulator
    queries it per buffer; if system was voiced within
    `crosstalkPersistSeconds` (250 ms), the mic's current buffer is
    zeroed before being fed to the recognizer. **Caveat**: this affects
    only what the recognizer sees; the mic `.wav` recording still has
    the raw bleed because it consumes the broadcaster upstream.
21. **`SourcePipeline` shouldn't be `@MainActor`.** Pipeline is
    @MainActor, and naively making child classes follow suit would
    serialize the per-stream accumulators on the MainActor — both
    audio paths plus the worker would queue behind UI updates. Keep
    `SourcePipeline`, `SherpaTranscriber`, `DenoisingAudioSource`
    as plain classes; their methods run on whatever executor the
    Swift runtime chose (cooperative pool from `withTaskGroup`).
    Only the UI-state writes hop back to MainActor (via
    `Task { @MainActor in ... }`).
22. **(Retired — was specific to whisper.cpp header mirroring.)**
23. **sherpa-onnx CoreML EP is only in the shared dylib build.**
    The `xcframework` release does NOT include CoreML. You must use
    `osx-arm64-shared.tar.bz2` and bundle the dylibs in `Frameworks/`.
    Without this, `provider = "coreml"` silently falls back to CPU.
24. **`SherpaOnnxOfflineTtsGenerate` is deprecated.** Always use
    `SherpaOnnxOfflineTtsGenerateWithConfig` with a `SherpaOnnxGenerationConfig`
    struct. The old function still compiles but triggers a deprecation
    warning; the new one also supports callback-based progress and
    reference-audio voice cloning.
25. **kitten-mini TTS has no `lexicon.txt`.** Its model directory
    contains `model.onnx`, `voices.bin`, `tokens.txt`, and
    `espeak-ng-data/`. The `SherpaOnnxOfflineTtsKittenModelConfig`
    struct has fields `model`, `voices`, `tokens`, `data_dir` —
    no `lexicon` field exists. Don't add one; it'll crash.
26. **`Bundle.main.url(forResource:withExtension:)` doesn't handle
    subdirectory paths.** `ModelConfig.ttsModel` is
    `"kitten-mini-en-v0_8/model.onnx"` — the slash means Bundle API
    returns nil. Always construct paths via
    `Bundle.main.bundleURL.appendingPathComponent("Contents/Resources").appendingPathComponent(relative)`.
    The `resourcePath(_:)` helper in `SherpaTranscriber` /
    `OnnxTTSSpeaker` does this correctly.
27. **sherpa-onnx GitHub release tag for speaker models has a typo.**
    The tag is `speaker-recongition-models` (missing the 'i' in
    recognition). `tools/download-sherpa.sh` uses the exact typo-d URL;
    do not "fix" it.
