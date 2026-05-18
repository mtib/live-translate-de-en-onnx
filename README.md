# LiveTranslate

Floating, translucent macOS 26+ app that captures your **microphone and
system audio** in parallel, transcribes both on-device via
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (streaming
zipformer RNN-T), and translates the result with Apple's `Translation`
framework. Each session lands as a zip in
`~/Documents/LiveTranslate/<stamp>.zip` containing both `.wav`s, the
per-source + merged SRTs, the JSONL log, and a ready-to-watch `.mkv`
(640×360, both subtitle tracks embedded).

The app also **streams synthesized translations over the LAN** using an
on-device ONNX TTS model — open `http://<mac-ip>:8765/` on a phone with
headphones and listen to near-real-time translated audio.

![Default layout](docs/default.png)
![Compact layout](docs/compact.png)

## Requirements

- **macOS 26 (Tahoe)** — uses the `Translation` framework and
  `ScreenCaptureKit`. Will not build or run on earlier releases.
- **Apple Silicon** — the sherpa-onnx CoreML execution provider is used
  for ASR and TTS inference. Intel is not supported.
- **ffmpeg** (optional) — only needed for `.mkv` packaging at session
  end. Without it the zip still contains the WAVs + SRTs + JSONL.

## Build & run

```sh
brew install ffmpeg            # optional, for MKV output
./build.sh                    # downloads sherpa-onnx + ONNX models, then compiles
open build/LiveTranslate.app
```

`build.sh` calls `tools/download-sherpa.sh` (idempotent) which fetches:

| Asset | Size | Purpose |
|---|---|---|
| `sherpa-onnx` v1.13.2 osx-arm64 shared dylib | ~30 MB | ASR + VAD + TTS runtime |
| `sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06` | ~120 MB | German streaming ASR |
| `silero_vad.onnx` | ~2 MB | Voice activity detection |
| `kitten-mini-en-v0_8` | ~50 MB | English on-device TTS |

The language pair is **fixed at compile time** (default: German → English).
To retarget, edit `ModelConfig.swift` and swap the ASR model in
`tools/download-sherpa.sh` / `build.sh`.

## One-time macOS setup

- **Translation language pack.** Apple downloads pairs on demand — add
  yours under **System Settings → Apple Intelligence & Siri →
  Translation Languages** before first run.
- **Permissions** (Microphone + Screen Recording). Prompted on first
  launch via `open build/LiveTranslate.app`. Never run the binary
  directly — TCC associates grants with the bundle.
- **Persistent permissions across rebuilds.** Ad-hoc signing churns the
  `cdhash` every build and macOS re-prompts. Create a self-signed cert
  in Keychain Access → Certificate Assistant (name e.g.
  `LiveTranslateDev`, Code Signing, Self Signed Root), then
  `export LIVETRANSLATE_SIGN_IDENTITY=LiveTranslateDev` — `build.sh`
  picks it up and all future rebuilds reuse the same TCC grants.

## Live translated-audio stream

When the bundled kitten-mini TTS model is present and source ≠ target
language, a radio-waves icon (⋰) appears in the toolbar. Click it to see:

- The stream URL (`http://<lan-ip>:8765/`) — click to copy
- A QR code to scan with a phone on the same Wi-Fi

The stream is a plain HTTP WAV — open it in VLC, mpv, or iOS Safari.
Chrome works. QuickTime buffers heavily (30+ s), so avoid it.

## How it works

**Audio capture:** Mic via `AVAudioEngine`, system audio via
`ScreenCaptureKit`. Each stream runs through its own `RNNoise` instance
(on-device denoising, recorded to WAV after this stage) and an
envelope-follower AGC (SIMD via Accelerate's `vDSP`).

**Transcription:** Both streams share one `SherpaTranscriber`. Each
`transcribe()` call creates its own sherpa-onnx streaming recognizer
stream (parallel, no locks needed). Audio is resampled from 48 kHz to
16 kHz and fed simultaneously to Silero-VAD (for speech gating and
speaker-boundary detection) and the streaming zipformer RNN-T recognizer.

**Sentence splitting:** Live ASR hypotheses are shown immediately as
partial text. Once a partial exceeds 30 characters and a sentence-ending
boundary (`. `, `? `, `! `) is found, that sentence is force-completed
mid-turn and translation + TTS kick off straight away — no waiting for
the full turn. If the speaker pauses for ≥ 0.8 s mid-turn (a speaker
change), the hypothesis so far is split into a new row. The ASR
endpoint (≥ 1 s trailing silence) closes the final row.

**Translation:** Apple's `Translation` framework, per-chunk, with a 1 s
throttle on partial updates and a final pass when the sentence is
complete.

**TTS + streaming:** Finalized translations are synthesized by
kitten-mini (ONNX, on-device) and streamed as 24 kHz PCM16 LE WAV over
a hand-rolled `NWListener` HTTP server.

**Crosstalk suppression:** Mic samples are zeroed while system audio is
active (250 ms window), so the mic track doesn't transcribe speaker
bleed.

**Output:** Non-overlapping, voice-onset-anchored timestamps in both
JSONL and SRT. End times reflect when speech actually stopped (last
voiced sample), not the end of the trailing silence buffer. At Stop,
ffmpeg wraps the WAVs + SRTs into an MKV and everything is zipped to
`~/Documents/LiveTranslate/<stamp>.zip`.

See [CLAUDE.md](CLAUDE.md) for the full file-by-file map, architecture
diagram, and lessons learned.

## Debug log

```sh
tail -f /tmp/livetranslate.log
```
