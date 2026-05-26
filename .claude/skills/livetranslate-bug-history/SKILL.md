---
name: livetranslate-bug-history
description: Use before changing audio capture, sherpa-onnx integration, lifecycle/state machines, SwiftUI translation hookup, SCK/macOS interaction, or any subtle concurrency in LiveTranslate. Numbered catalogue of bugs that have bitten us. NEVER delete entries — append new ones in same commit when a new bug is fixed.
---

# Things that have bitten us already

Append-only institutional memory. When a new non-obvious bug is fixed, add a numbered entry: what went wrong, why it was hard to diagnose, shape of the fix.

1. **Direct binary launch** loses bundle/TCC; always `open build/LiveTranslate.app`.
2. **Reinstalling AVAudioEngine tap** between sessions silently stops recognition after ~1 min. Install tap once on first `start()`.
3. **`requiresOnDeviceRecognition=true`** (retired Apple Speech) hard-failed without language model. Kept as memory.
4. **`NSLog`** unreliable in `log show` for ad-hoc apps on macOS 26. Use `Log.line` → `/tmp/livetranslate.log`.
5. **No `swift test`** without full Xcode (CLT lacks XCTest/Swift Testing). Tests omitted.
6. **Single-consumer AsyncStream** breaks every Start after the first. Use `BufferBroadcaster` for fan-out.
7. **Speaker bleed via mic** stalled recognizer. Fix: SCK system audio + crosstalk gate.
8. **Index-only snapshot reconciliation** (Apple Speech) left orphan rows on revision. Retired.
9. **`DispatchSemaphore.wait()` on MainActor** for async work = instant deadlock. `AudioSource.start()` is `async throws` now.
10. **Active-session prune exemption** caused unbounded growth in long sessions. Only protect the LAST sentence.
11. **Hoisting `let audio = source.buffers`** out of restart loop drained second session. `buffers` is a fresh-subscription factory — call per consumer.
12. **Unstructured `Task { }` children** don't inherit cancellation. Use `withTaskGroup`. Don't nil `runTask` inside `stop()`.
13. **`CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer`** with fixed AudioBufferList size only fits ONE buffer; SCK non-interleaved stereo = two. Use `CMSampleBufferCopyPCMDataIntoAudioBufferList` with `mutableAudioBufferList`.
14. **Apple Speech serializes per-app** — two concurrent recognizers fast-fail "No speech detected". Moved to sherpa-onnx. (Memory only.)
15. **Naive buffer-stream interleaving** doubled latency. Mix at SAMPLE level via mic-clock + bounded queue + `vDSP_vadd`. (Mixing later removed; principle retained.)
16. **Whisper drops audio <1s** (100 mel frames). Pad+gate. (Retired — sherpa streams.)
17. **Cancelling `runTask`** aborted mid-flight audio. Shutdown via `BufferBroadcaster.finishAll()` (graceful drain), not task cancel.
18. **Per-stream pipelines, not mixing** — preserves source attribution.
19. **Two concurrent `whisper_init`** failed on Metal. `NSLock` around `ensureContextLoaded`. Same pattern now in `ensureRecognizerLoaded`.
20. **Crosstalk** — see architecture skill. Mic accumulator zeros effective buffer when system voiced within 0.25 s; `DenoisingAudioSource(mic).muteWhen` applies gate upstream so recorder also sees muted audio.
21. **`SourcePipeline` is NOT `@MainActor`** — would serialize accumulators on main thread. Only UI writes hop to MainActor.
22. (Retired — whisper.cpp header mirroring.)
23. **sherpa-onnx CoreML EP** only in `osx-arm64-shared.tar.bz2`, NOT in xcframework. `provider="coreml"` silently falls back to CPU otherwise.
24. **`SherpaOnnxOfflineTtsGenerate`** is deprecated. Use `SherpaOnnxOfflineTtsGenerateWithConfig` + `SherpaOnnxGenerationConfig`.
25. **kitten-mini has no `lexicon.txt`**. Kitten model config has `model/voices/tokens/data_dir` only. Adding lexicon crashes.
26. **`Bundle.main.url(forResource:withExtension:)`** doesn't handle subdir paths like `kitten-mini-en-v0_8/model.onnx`. Use `bundleURL.appendingPathComponent("Contents/Resources")...path`.
27. **sherpa-onnx speaker-models tag has typo** `speaker-recongition-models`. Do not "fix" — script uses exact URL.
28. **Leading `.?! ` from ASR after stream reset.** German zipformer prepends prev-utterance terminal punctuation. `normalizeHypothesis` strips it.
29. **Standalone punctuation rows.** If endpoint remainder is punctuation-only after a force-complete, fire `.dropped` instead of yielding a `.`-only turn. Worker has alphanumeric guard as safety net.
30. **`ttsActive` needs TWO signals**: `ttsModelLoaded && ttsListenerCount>0`. Listener-only gate showed green before model ready.
31. **`AudioRecorder.flush()`** must `self.file = nil` (deallocate AVAudioFile) to finalize WAV header. Otherwise ffmpeg sees 0-length file.
32. **Semantic endpoint suppression** must NOT reset `committedLength`/`chunkStartSample`. Resetting double-emits already-committed text on next force-complete.
33. **`OnnxTTSSpeaker` drops OLDEST** on backpressure (not newest). Max queue 5. `pumpLocked` combines pending into one synth call for prosody continuity.
34. **`SCContentSharingPicker.shared.isActive=true` kills other SCStreams.** Self-rolled SwiftUI chooser over `SCShareableContent` (`ScreenPicker.swift`). Don't reintroduce without registering every concurrent SCStream.
35. **SCK letterbox is origin-aligned, not centered.** Pick writer dims to match source aspect (`ScreenVideoRecorder.pickDimensions`); ffmpeg `pad=...:(ow-iw)/2:(oh-ih)/2` centers on final canvas.
36. **AVAssetWriter MOV `-c:v copy` into MKV** failed (ffmpeg 183) on mid-segment resize. Always re-encode via libx264 from filter_complex.
37. **AVAssetWriter `startSession(atSourceTime:)` + offset sidecar + `setpts=PTS-STARTPTS+<offset>/TB`** — all three needed for multi-segment screen video to land on audio timeline.
38. **SCK `SystemAudioSource` silently dies ~44 s on macOS 26.** `didStopWithError` was no-op. Fix: `intentionalStop` flag distinguishes user vs system; on system stop, exponential-backoff reconnect (1→2→4→8→16 s); broadcaster NOT finished between attempts.
39. **`ScreenVideoRecorder` didn't auto-resume.** Same SCK lifecycle as #38 but only finalized MOV — no new segment. Added `intentionalStop` + `onSystemStop` callback; `Pipeline.openScreenSegment` reopens segment.
40. **Late translation tasks after `stop()` resurrect cleared rows.** Gate `applyLifecycle` and `graduate` on `Pipeline.isActive`. Set false in `run()` `defer` AFTER finalize await.
41. **Main window must render sentence list ONLY.** All controls live in MenuBar popover + OptionsView. Re-adding bars to TranscriptView is a regression.
42. **`@AppStorage` inside ObservableObject does NOT publish** — no `objectWillChange`. Use `@Published var x { didSet { UserDefaults... } }`.
43. **`AVAudioEngine.setVoiceProcessingEnabled(true)`** ducks OTHER-APP audio at speakers. FaceTime works because its own playback IS the VP output. For LiveTranslate the audio we want to hear IS other-app audio (we're capturing it via SCK). Also exposes multi-channel input — needs `converter.channelMap=[0]`. Current path: no VP; RNNoise+AGC+crosstalk gate.
44. **NSWindow `alphaValue` fades text too.** Apply opacity on ZStack background `Color(...).opacity(...)`; text stays full alpha. Same for SummaryView.
45. **`DenoisingAudioSource.start()` not idempotent** — duplicate pump task double-broadcast. Guard `if pumpTask != nil { return }`.
46. **ANE single-tenant.** FoundationModels LLM and sherpa-onnx CoreML EP both target ANE; LLM gen stalls ASR. Gate summary triggers on `inflightChunks.isEmpty && >=1.5s idle`.
47. **`@AppStorage("compactMode")`** in TranscriptView was load-bearing for layout. Replaced by `AppSettings.LayoutMode.compact`. Chevron toggle removed — Options is single layout surface.
48. **HTTP request reads can't assume one packet.** `receiveRequest` accumulates until `\r\n\r\n`, cap 16 KB.
49. **SSE replay-snapshot race** — register subscriber FIRST under lock, then snapshot replay+settings under same lock, then send preamble. Same pattern for OBS event stream.
50. **`jsonEscape` must escape all `<0x20` control chars** — `\b`/`\f`/`\t` explicit; everything else `<0x20` → `\u00XX`. Bare control chars break `JSON.parse` on the listen page.
51. **Translation row must be ABOVE transcription** on web. `insertBefore(tDiv, row.firstChild)` not `appendChild`.
52. **Don't animate hypothesis→finalized crossover.** Tag `data-pending="1"` on hypothesis-done; in-place swap on onFinalized. New rows use `.fresh` class for fadein.
53. **`.contentTransition(.opacity)` on partial text fades through alpha 0** = flicker. Use `.contentTransition(.identity)`; keep `.transition(.opacity)` only on row add/remove.
