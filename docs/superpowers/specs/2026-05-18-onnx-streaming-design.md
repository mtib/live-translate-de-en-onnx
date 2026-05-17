# ONNX Streaming ASR + TTS Design

**Date:** 2026-05-18  
**Branch:** `feature/onnx-streaming`

## Goal

Replace whisper.cpp with sherpa-onnx streaming RNN-T, add Silero-VAD and speaker
diarization, replace AVSpeechSynthesizer TTS with ONNX (kitten-mini-en-v0_8).
Language pair is **compile-time constant** — default German → English.

## Approved decisions

| Question | Answer |
|---|---|
| ASR backend | sherpa-onnx streaming zipformer RNN-T |
| VAD | Silero-VAD via sherpa-onnx |
| Speaker diarization | Full per-stream: eres2net embeddings + cosine-sim clustering |
| Chunking trigger | Sherpa endpoint detection (1 s silence) OR speaker change |
| Translation trigger | One call per closed speaker-turn |
| TTS | `kitten-mini-en-v0_8` via sherpa-onnx offline TTS |
| Language selection | Compile-time constants, no runtime picker |
| Default language | Source: `de` (German), Target: `en` (English) |
| CoreML/Metal | sherpa-onnx arm64 shared dylib (CoreML EP enabled) |
| Archives | Speaker label in transcription text only (`[Speaker 1] text`) |

## Architecture

```
AudioSource (48 kHz Float32)
  → DenoisingAudioSource (RNNoise — kept for recording quality)
  → SherpaTranscriber
      ├─ resample 48 → 16 kHz (AVAudioConverter)
      ├─ Silero-VAD  (frame-level speech/silence detection)
      ├─ SherpaStreamingRecognizer (zipformer RNN-T, partial text, endpoint)
      └─ SpeakerTracker (eres2net embedding per turn → "Speaker N" label)
  → onChunkLifecycle callbacks → Pipeline (unchanged contract)
  → Apple Translation API (per closed turn)
  → LiveAudioServer + OnnxTTSSpeaker (kitten-mini)
```

## File map

### New / rewritten
| File | What |
|---|---|
| `Sources/LiveTranslate/SherpaTranscriber.swift` | Replaces WhisperCppTranscriber. Drives Silero-VAD + RNN-T + SpeakerTracker per stream. |
| `Sources/LiveTranslate/SpeakerTracker.swift` | eres2net embedding extraction + cosine-sim speaker ID assignment. |
| `Sources/LiveTranslate/OnnxTTSSpeaker.swift` | Replaces TTSSpeaker. kitten-mini via sherpa-onnx offline TTS. |
| `Sources/LiveTranslate/ModelConfig.swift` | Compile-time constants: language codes, model paths, provider string. |
| `Sources/CSherpaOnnx/` | SwiftPM C target bridging the sherpa-onnx shared dylib. |
| `tools/download-sherpa.sh` | Downloads dylib + all models (VAD, ASR, speaker, TTS). |

### Deleted
- `Sources/LiveTranslate/WhisperCppTranscriber.swift`
- `Sources/LiveTranslate/TTSSpeaker.swift`
- `Sources/CWhisper/` (entire directory)
- `tools/build-whisper.sh`

### Modified
- `Package.swift` — remove CWhisper, add CSherpaOnnx
- `build.sh` — bundle sherpa dylib into Frameworks/, set RPATH; remove whisper steps
- `Sources/LiveTranslate/Pipeline.swift` — swap TTSSpeaker → OnnxTTSSpeaker; remove whisper init
- `Sources/LiveTranslate/Types.swift` — remove whisper-specific fields if any; add speakerID to SessionSentence
- `README.md`, `CLAUDE.md` — updated

## Key interfaces

```swift
// ModelConfig.swift
enum ModelConfig {
    static let sourceLanguage = "de"
    static let targetLanguage = "en"
    static let provider       = "coreml"   // or "cpu" fallback

    // Paths relative to app bundle Resources/
    static let asrModel       = "sherpa-onnx-streaming-zipformer-de"
    static let vadModel       = "silero_vad.onnx"
    static let speakerModel   = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
    static let ttsModel       = "kitten-mini-en-v0_8"
}

// SherpaTranscriber — same Transcriber protocol as WhisperCppTranscriber
// onChunkLifecycle fires: .listening / .transcribing(partial, speakerID) / .completed(text) / .dropped

// SpeakerTracker
class SpeakerTracker {
    func embedding(for pcm16: [Float]) -> [Float]   // eres2net
    func speakerID(for embedding: [Float]) -> Int   // cosine-sim, assigns new ID if novel
    func reset()
}

// OnnxTTSSpeaker — same public API as TTSSpeaker
init(onPCM: @escaping (Data)->Void, onActivityChanged: @escaping (Bool)->Void)
func enqueue(_ text: String)
func stop()
static func isAvailable() -> Bool   // checks kitten-mini model on disk
```

## Build / distribution

`tools/download-sherpa.sh` fetches (idempotent):
1. `sherpa-onnx-v1.13.2-osx-arm64-shared.tar.bz2` → `external/sherpa-onnx/`
2. Silero-VAD model → `build/sherpa-models/`
3. German streaming zipformer model → `build/sherpa-models/`
4. eres2net speaker embedding model → `build/sherpa-models/`
5. `kitten-mini-en-v0_8` TTS model → `build/sherpa-models/`

`build.sh` copies `external/sherpa-onnx/lib/*.dylib` into
`build/LiveTranslate.app/Contents/Frameworks/` and passes
`-rpath @executable_path/../Frameworks` to the linker.
