# ONNX Streaming ASR + TTS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace whisper.cpp STT + AVSpeechSynthesizer TTS with sherpa-onnx streaming RNN-T, Silero-VAD, speaker diarization, and kitten-mini ONNX TTS. Language pair fixed at compile time: German → English.

**Architecture:** sherpa-onnx arm64 shared dylib (CoreML EP) linked into the app; all models bundled in Resources/. SherpaTranscriber implements the existing Transcriber protocol — Pipeline is unchanged except for swapping the concrete type. OnnxTTSSpeaker has the same public API as TTSSpeaker.

**Tech Stack:** sherpa-onnx v1.13.2, Silero-VAD, zipformer RNN-T (German), eres2net speaker embeddings, kitten-mini-en-v0_8 TTS, Apple Translation (unchanged), SwiftPM + bash build.

---

## Task 1: Download sherpa-onnx dylib + models

**Files:**
- Create: `tools/download-sherpa.sh`

- [ ] Write `tools/download-sherpa.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

SHERPA_VERSION="1.13.2"
SHERPA_DIR="external/sherpa-onnx"
MODELS_DIR="build/sherpa-models"
BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download"

mkdir -p "${MODELS_DIR}"

# --- dylib ---
if [[ ! -f "${SHERPA_DIR}/lib/libsherpa-onnx-c-api.dylib" ]]; then
  echo "→ downloading sherpa-onnx ${SHERPA_VERSION} arm64 shared"
  TMP=$(mktemp -d)
  curl -L --fail --progress-bar \
    -o "${TMP}/sherpa.tar.bz2" \
    "${BASE}/v${SHERPA_VERSION}/sherpa-onnx-v${SHERPA_VERSION}-osx-arm64-shared.tar.bz2"
  mkdir -p "${SHERPA_DIR}"
  tar -xjf "${TMP}/sherpa.tar.bz2" -C "${SHERPA_DIR}" --strip-components=1
  rm -rf "${TMP}"
  echo "✓ sherpa-onnx dylib"
else
  echo "✓ sherpa-onnx dylib already present"
fi

# --- Silero-VAD ---
VAD_MODEL="${MODELS_DIR}/silero_vad.onnx"
if [[ ! -f "${VAD_MODEL}" ]]; then
  echo "→ downloading silero_vad.onnx"
  curl -L --fail --progress-bar \
    -o "${VAD_MODEL}" \
    "${BASE}/asr-models/silero_vad.onnx"
  echo "✓ silero_vad.onnx"
else
  echo "✓ silero_vad.onnx already present"
fi

# --- German streaming ASR ---
DE_DIR="${MODELS_DIR}/sherpa-onnx-streaming-zipformer-de-2024-12-13"
if [[ ! -d "${DE_DIR}" ]]; then
  echo "→ downloading German streaming zipformer"
  TMP=$(mktemp -d)
  curl -L --fail --progress-bar \
    -o "${TMP}/asr-de.tar.bz2" \
    "${BASE}/asr-models/sherpa-onnx-streaming-zipformer-de-2024-12-13.tar.bz2"
  mkdir -p "${MODELS_DIR}"
  tar -xjf "${TMP}/asr-de.tar.bz2" -C "${MODELS_DIR}"
  rm -rf "${TMP}"
  echo "✓ German ASR model"
else
  echo "✓ German ASR model already present"
fi

# --- Speaker embedding (eres2net) ---
SPKR_MODEL="${MODELS_DIR}/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
if [[ ! -f "${SPKR_MODEL}" ]]; then
  echo "→ downloading eres2net speaker embedding"
  curl -L --fail --progress-bar \
    -o "${SPKR_MODEL}" \
    "${BASE}/speaker-recog-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
  echo "✓ eres2net speaker model"
else
  echo "✓ eres2net speaker model already present"
fi

# --- kitten-mini TTS ---
KITTEN_DIR="${MODELS_DIR}/kitten-mini-en-v0_8"
if [[ ! -d "${KITTEN_DIR}" ]]; then
  echo "→ downloading kitten-mini-en-v0_8 TTS"
  TMP=$(mktemp -d)
  curl -L --fail --progress-bar \
    -o "${TMP}/tts.tar.bz2" \
    "${BASE}/tts-models/kitten-mini-en-v0_8.tar.bz2"
  mkdir -p "${MODELS_DIR}"
  tar -xjf "${TMP}/tts.tar.bz2" -C "${MODELS_DIR}"
  rm -rf "${TMP}"
  echo "✓ kitten-mini TTS model"
else
  echo "✓ kitten-mini TTS model already present"
fi

echo ""
echo "All sherpa-onnx assets ready."
```

- [ ] `chmod +x tools/download-sherpa.sh && ./tools/download-sherpa.sh`

- [ ] Verify:
```bash
ls external/sherpa-onnx/lib/libsherpa-onnx-c-api.dylib
ls build/sherpa-models/silero_vad.onnx
ls build/sherpa-models/sherpa-onnx-streaming-zipformer-de-2024-12-13/
ls build/sherpa-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
ls build/sherpa-models/kitten-mini-en-v0_8/
```

- [ ] Add to `.gitignore`:
```
external/sherpa-onnx/
build/sherpa-models/
```

- [ ] Commit:
```bash
git add tools/download-sherpa.sh .gitignore
git commit -m "Add download-sherpa.sh: fetches dylib + all ONNX models"
```

---

## Task 2: Swift package bridge for sherpa-onnx C API

**Files:**
- Create: `Sources/CSherpaOnnx/include/module.modulemap`
- Create: `Sources/CSherpaOnnx/include/sherpa-onnx-c-api.h` (copy from external)
- Modify: `Package.swift`

- [ ] Mirror the C API header:
```bash
mkdir -p Sources/CSherpaOnnx/include
cp external/sherpa-onnx/include/sherpa-onnx/c-api/c-api.h \
   Sources/CSherpaOnnx/include/sherpa-onnx-c-api.h
```

- [ ] Create `Sources/CSherpaOnnx/include/module.modulemap`:
```
module CSherpaOnnx {
    header "sherpa-onnx-c-api.h"
    export *
}
```

- [ ] Create `Sources/CSherpaOnnx/CSherpaOnnx.c` (empty stub so SwiftPM treats it as a C target):
```c
// Empty — this target exists only to expose the sherpa-onnx C API to Swift.
```

- [ ] Update `Package.swift` — remove CWhisper, add CSherpaOnnx:

```swift
// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "LiveTranslate",
    platforms: [.macOS(.v15)],
    targets: [
        .target(
            name: "CRNNoise",
            path: "Sources/CRNNoise",
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("."),
                .unsafeFlags([
                    "-Wno-implicit-function-declaration",
                    "-Wno-null-dereference",
                ]),
            ]
        ),
        .target(
            name: "CSherpaOnnx",
            path: "Sources/CSherpaOnnx",
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("include"),
            ],
            linkerSettings: [
                .unsafeFlags([
                    "-L./external/sherpa-onnx/lib",
                    "-lsherpa-onnx-c-api",
                ]),
                .linkedFramework("Foundation"),
                .linkedFramework("CoreML"),
                .linkedFramework("Metal"),
                .linkedFramework("Accelerate"),
            ]
        ),
        .executableTarget(
            name: "LiveTranslate",
            dependencies: ["CRNNoise", "CSherpaOnnx"],
            path: "Sources/LiveTranslate",
            swiftSettings: [
                .swiftLanguageMode(.v5)
            ]
        )
    ]
)
```

- [ ] Verify the module compiles:
```bash
swift build 2>&1 | head -30
```
Expected: CSherpaOnnx compiles (LiveTranslate will fail — that's fine, we haven't written the Swift yet)

- [ ] Commit:
```bash
git add Sources/CSherpaOnnx/ Package.swift
git commit -m "Add CSherpaOnnx bridge target for sherpa-onnx C API"
```

---

## Task 3: ModelConfig + delete dead code

**Files:**
- Create: `Sources/LiveTranslate/ModelConfig.swift`
- Delete: `Sources/LiveTranslate/WhisperCppTranscriber.swift`
- Delete: `Sources/LiveTranslate/TTSSpeaker.swift`
- Delete: `Sources/CWhisper/` (directory)
- Delete: `tools/build-whisper.sh`

- [ ] Create `Sources/LiveTranslate/ModelConfig.swift`:

```swift
import Foundation

/// Compile-time configuration for ASR, VAD, speaker diarization, and TTS.
/// To change the language pair, edit these constants and re-download models
/// with `tools/download-sherpa.sh`.
enum ModelConfig {
    // MARK: - Language
    /// BCP-47 code of the spoken source language.
    static let sourceLanguageCode = "de"
    /// BCP-47 code for the translation target.
    static let targetLanguageCode = "en"

    // MARK: - ONNX provider
    /// "coreml" uses Apple Neural Engine / GPU on Apple Silicon.
    /// Fall back to "cpu" if CoreML EP is unavailable.
    static let provider = "coreml"

    // MARK: - Model directory (inside app bundle Resources/)
    static func resourceURL(_ name: String) -> URL {
        Bundle.main.resourceURL!.appendingPathComponent(name)
    }

    // VAD
    static var vadModelPath: String {
        resourceURL("silero_vad.onnx").path
    }

    // ASR (German streaming zipformer)
    static let asrModelDir = "sherpa-onnx-streaming-zipformer-de-2024-12-13"
    static var asrEncoderPath: String {
        resourceURL("\(asrModelDir)/encoder-epoch-99-avg-1.int8.onnx").path
    }
    static var asrDecoderPath: String {
        resourceURL("\(asrModelDir)/decoder-epoch-99-avg-1.int8.onnx").path
    }
    static var asrJoinerPath: String {
        resourceURL("\(asrModelDir)/joiner-epoch-99-avg-1.int8.onnx").path
    }
    static var asrTokensPath: String {
        resourceURL("\(asrModelDir)/tokens.txt").path
    }

    // Speaker embedding
    static var speakerModelPath: String {
        resourceURL("3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx").path
    }

    // TTS
    static let ttsModelDir = "kitten-mini-en-v0_8"
    static var ttsModelPath: String {
        resourceURL("\(ttsModelDir)/model.onnx").path
    }
    static var ttsTokensPath: String {
        resourceURL("\(ttsModelDir)/tokens.txt").path
    }
    static var ttsLexiconPath: String {
        resourceURL("\(ttsModelDir)/lexicon.txt").path
    }
    static var ttsDataDir: String {
        resourceURL("\(ttsModelDir)/espeak-ng-data").path
    }
}
```

- [ ] Delete dead files:
```bash
rm Sources/LiveTranslate/WhisperCppTranscriber.swift
rm Sources/LiveTranslate/TTSSpeaker.swift
rm -rf Sources/CWhisper/
rm tools/build-whisper.sh
```

- [ ] Commit:
```bash
git add -A
git commit -m "Add ModelConfig; delete whisper + old TTS dead code"
```

---

## Task 4: SpeakerTracker

**Files:**
- Create: `Sources/LiveTranslate/SpeakerTracker.swift`

- [ ] Create `Sources/LiveTranslate/SpeakerTracker.swift`:

```swift
import Foundation
import CSherpaOnnx

/// Assigns stable "Speaker N" labels to audio turns within one stream.
/// Uses eres2net speaker embeddings + cosine-similarity clustering.
/// Not thread-safe — call from the transcriber's serial queue only.
final class SpeakerTracker {
    /// Cosine distance threshold above which a new speaker cluster is created.
    private let newSpeakerThreshold: Float = 0.5
    /// Stored centroid embeddings, one per discovered speaker (1-indexed).
    private var centroids: [[Float]] = []
    private let extractor: OpaquePointer?

    init() {
        var config = SherpaOnnxSpeakerEmbeddingExtractorConfig()
        let modelPath = ModelConfig.speakerModelPath
        modelPath.withCString { ptr in
            config.model = ptr
            config.num_threads = 2
            config.debug = 0
            let providerStr = ModelConfig.provider
            providerStr.withCString { p in config.provider = p }
        }
        extractor = SherpaOnnxCreateSpeakerEmbeddingExtractor(&config)
    }

    deinit {
        if let e = extractor { SherpaOnnxDestroySpeakerEmbeddingExtractor(e) }
    }

    /// Returns "Speaker N" label for the given 16 kHz mono Float32 samples.
    /// `N` is 1-based and stable within a run.
    func label(for samples: [Float]) -> String {
        guard let ext = extractor else { return "Speaker 1" }
        var stream = SherpaOnnxCreateSpeakerEmbeddingExtractorStream(ext)
        samples.withUnsafeBufferPointer { buf in
            SherpaOnnxSpeakerEmbeddingExtractorStreamAcceptWaveform(
                stream, 16000, buf.baseAddress, Int32(buf.count))
        }
        SherpaOnnxSpeakerEmbeddingExtractorStreamInputFinished(stream)
        guard SherpaOnnxSpeakerEmbeddingExtractorStreamIsReady(ext, stream) != 0 else {
            SherpaOnnxDestroySpeakerEmbeddingExtractorStream(stream)
            return "Speaker 1"
        }
        let dim = Int(SherpaOnnxSpeakerEmbeddingExtractorDim(ext))
        var embedding = [Float](repeating: 0, count: dim)
        embedding.withUnsafeMutableBufferPointer { buf in
            SherpaOnnxSpeakerEmbeddingExtractorStreamGetEmbedding(stream, buf.baseAddress)
        }
        SherpaOnnxDestroySpeakerEmbeddingExtractorStream(stream)
        let id = clusterID(for: embedding)
        return "Speaker \(id)"
    }

    func reset() { centroids.removeAll() }

    // MARK: - Private

    private func clusterID(for emb: [Float]) -> Int {
        for (i, centroid) in centroids.enumerated() {
            if cosineDistance(emb, centroid) < newSpeakerThreshold {
                // Update centroid (running mean)
                centroids[i] = zip(centroid, emb).map { ($0 + $1) / 2 }
                return i + 1
            }
        }
        centroids.append(emb)
        return centroids.count
    }

    private func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0; var na: Float = 0; var nb: Float = 0
        for i in 0..<min(a.count, b.count) {
            dot += a[i] * b[i]; na += a[i]*a[i]; nb += b[i]*b[i]
        }
        guard na > 0, nb > 0 else { return 1 }
        return 1 - dot / (sqrt(na) * sqrt(nb))
    }
}
```

- [ ] Commit:
```bash
git add Sources/LiveTranslate/SpeakerTracker.swift
git commit -m "Add SpeakerTracker: eres2net embeddings + cosine-sim clustering"
```

---

## Task 5: SherpaTranscriber

**Files:**
- Create: `Sources/LiveTranslate/SherpaTranscriber.swift`
- Modify: `Sources/LiveTranslate/Types.swift` (add speakerLabel to SessionSentence)

- [ ] Add `speakerLabel` to `SessionSentence` in `Types.swift`:

Find the `SessionSentence` struct (around line 95) and add the field:
```swift
struct SessionSentence: Sendable, Equatable {
    let text: String
    let isFinal: Bool
    let startSeconds: Double?
    let endSeconds: Double?
    let speakerLabel: String   // e.g. "Speaker 1"; empty string if unknown

    init(text: String, isFinal: Bool,
         startSeconds: Double? = nil, endSeconds: Double? = nil,
         speakerLabel: String = "") {
        self.text = text
        self.isFinal = isFinal
        self.startSeconds = startSeconds
        self.endSeconds = endSeconds
        self.speakerLabel = speakerLabel
    }
}
```

- [ ] Create `Sources/LiveTranslate/SherpaTranscriber.swift`:

```swift
import Foundation
import AVFoundation
import CSherpaOnnx

/// Streaming ASR using sherpa-onnx zipformer RNN-T + Silero-VAD.
/// Fires `onChunkLifecycle` callbacks identical to the old WhisperCppTranscriber
/// so Pipeline requires no changes.
///
/// One instance shared between mic and system pipelines; each call to
/// `transcribe(audio:locale:source:)` runs independently on the cooperative
/// pool. The sherpa-onnx recognizer is NOT thread-safe — each transcribe()
/// call creates its own recognizer stream; the underlying model handle is
/// shared and sherpa-onnx serialises internally.
final class SherpaTranscriber: @unchecked Sendable, Transcriber {

    // MARK: - Chunk lifecycle callback (same contract as WhisperCppTranscriber)
    enum ChunkEvent {
        case listening
        case transcribing(partial: String)
        case completed(text: String)
        case dropped
    }
    var onChunkLifecycle: ((UUID, SourceTag, ChunkEvent) -> Void)?

    // MARK: - Crosstalk suppression (same logic as WhisperCppTranscriber)
    private let crosstalkLock = NSLock()
    private var lastSystemVoicedAt: Date = .distantPast
    private let crosstalkPersistSeconds: TimeInterval = 0.25

    func markSystemVoiced() {
        crosstalkLock.lock(); lastSystemVoicedAt = Date(); crosstalkLock.unlock()
    }
    private var systemIsVoiced: Bool {
        crosstalkLock.lock()
        defer { crosstalkLock.unlock() }
        return Date().timeIntervalSince(lastSystemVoicedAt) < crosstalkPersistSeconds
    }

    // MARK: - Shared model handles (init once, reuse per stream)
    private let recognizerHandle: OpaquePointer
    private let vadHandle: OpaquePointer
    private let speakerTracker = SpeakerTracker()
    private let trackerLock = NSLock()

    init() {
        // --- VAD ---
        var sileroConfig = SherpaOnnxSileroVadModelConfig()
        ModelConfig.vadModelPath.withCString { p in sileroConfig.model = p }
        sileroConfig.threshold = 0.5
        sileroConfig.min_silence_duration = 0.15
        sileroConfig.min_speech_duration  = 0.25
        sileroConfig.window_size          = 512
        sileroConfig.max_speech_duration  = 30.0

        var vadConfig = SherpaOnnxVadModelConfig()
        vadConfig.silero_vad   = sileroConfig
        vadConfig.sample_rate  = 16000
        vadConfig.num_threads  = 2
        ModelConfig.provider.withCString { p in vadConfig.provider = p }
        vadHandle = SherpaOnnxCreateVoiceActivityDetector(&vadConfig, 30.0)

        // --- Streaming RNN-T recognizer ---
        var transducer = SherpaOnnxOnlineTransducerModelConfig()
        ModelConfig.asrEncoderPath.withCString { p in transducer.encoder = p }
        ModelConfig.asrDecoderPath.withCString { p in transducer.decoder = p }
        ModelConfig.asrJoinerPath.withCString  { p in transducer.joiner  = p }

        var modelConfig = SherpaOnnxOnlineModelConfig()
        modelConfig.transducer   = transducer
        ModelConfig.asrTokensPath.withCString { p in modelConfig.tokens = p }
        modelConfig.num_threads  = 4
        ModelConfig.provider.withCString { p in modelConfig.provider = p }

        var featConfig = SherpaOnnxFeatureConfig()
        featConfig.sample_rate = 16000
        featConfig.feature_dim = 80

        var recognizerConfig = SherpaOnnxOnlineRecognizerConfig()
        recognizerConfig.feat_config      = featConfig
        recognizerConfig.model_config     = modelConfig
        recognizerConfig.enable_endpoint  = 1
        recognizerConfig.rule1_min_trailing_silence = 1.0
        recognizerConfig.rule2_min_trailing_silence = 1.2
        recognizerConfig.rule3_min_utterance_length = 20.0
        "greedy_search".withCString { p in recognizerConfig.decoding_method = p }

        recognizerHandle = SherpaOnnxCreateOnlineRecognizer(&recognizerConfig)
        Log.line("SherpaTranscriber: models loaded (provider=\(ModelConfig.provider))")
    }

    deinit {
        SherpaOnnxDestroyOnlineRecognizer(recognizerHandle)
        SherpaOnnxDestroyVoiceActivityDetector(vadHandle)
    }

    // MARK: - Transcriber protocol

    func transcribe(
        audio: AsyncStream<AVAudioPCMBuffer>,
        locale: SourceLocale,
        source: SourceTag
    ) -> AsyncThrowingStream<SessionSnapshot, Error> {
        AsyncThrowingStream { continuation in
            Task.detached { [weak self] in
                guard let self else { continuation.finish(); return }
                await self.run(audio: audio, source: source, continuation: continuation)
            }
        }
    }

    // MARK: - Core loop

    private func run(
        audio: AsyncStream<AVAudioPCMBuffer>,
        source: SourceTag,
        continuation: AsyncThrowingStream<SessionSnapshot, Error>.Continuation
    ) async {
        let resampler = AudioResampler(from: 48000, to: 16000)
        let recStream = SherpaOnnxCreateOnlineStream(recognizerHandle)
        defer { SherpaOnnxDestroyOnlineStream(recStream) }

        var chunkID   = UUID()
        var chunkOpen = false
        var chunkSamples: [Float] = []
        var chunkStartTime: Double = 0
        var samplesProcessed: Double = 0   // cumulative 16 kHz samples

        func openChunk() {
            chunkID = UUID(); chunkOpen = true; chunkSamples = []
            chunkStartTime = samplesProcessed / 16000
            onChunkLifecycle?(chunkID, source, .listening)
        }
        func closeChunk() {
            guard chunkOpen else { return }
            chunkOpen = false
            let text = finalText(from: recStream)
            let endTime = samplesProcessed / 16000
            if text.isEmpty {
                onChunkLifecycle?(chunkID, source, .dropped)
            } else {
                trackerLock.lock()
                let label = speakerTracker.label(for: chunkSamples)
                trackerLock.unlock()
                let labeled = "[\(label)] \(text)"
                onChunkLifecycle?(chunkID, source, .completed(text: labeled))
                let sentence = SessionSentence(
                    text: labeled, isFinal: true,
                    startSeconds: chunkStartTime, endSeconds: endTime,
                    speakerLabel: label)
                continuation.yield(SessionSnapshot(sentences: [sentence]))
            }
            SherpaOnnxOnlineStreamReset(recStream)
        }

        for await buf in audio {
            if Task.isCancelled { break }

            // Resample 48→16 kHz
            guard let samples16k = resampler.process(buf) else { continue }

            // Crosstalk gate for mic
            var gated = samples16k
            if source == .mic && systemIsVoiced {
                gated = [Float](repeating: 0, count: samples16k.count)
            } else if source == .system {
                markSystemVoiced()
            }

            samplesProcessed += Double(gated.count)

            // Feed VAD
            gated.withUnsafeBufferPointer { ptr in
                SherpaOnnxVoiceActivityDetectorAcceptWaveform(
                    vadHandle, ptr.baseAddress, Int32(ptr.count))
            }

            // Feed recognizer
            gated.withUnsafeBufferPointer { ptr in
                SherpaOnnxOnlineStreamAcceptWaveform(
                    recStream, 16000, ptr.baseAddress, Int32(ptr.count))
            }
            while SherpaOnnxIsOnlineStreamReady(recognizerHandle, recStream) != 0 {
                SherpaOnnxDecodeOnlineStream(recognizerHandle, recStream)
            }

            // Track partial text → .transcribing
            let partial = partialText(from: recStream)
            if !partial.isEmpty {
                if !chunkOpen { openChunk() }
                chunkSamples.append(contentsOf: gated)
                onChunkLifecycle?(chunkID, source, .transcribing(partial: partial))
            }

            // Endpoint → close turn
            if SherpaOnnxOnlineStreamIsEndpoint(recognizerHandle, recStream) != 0 {
                closeChunk()
            }
        }

        closeChunk()
        continuation.finish()
    }

    // MARK: - Helpers

    private func partialText(from stream: OpaquePointer?) -> String {
        guard let result = SherpaOnnxGetOnlineStreamResult(recognizerHandle, stream) else { return "" }
        defer { SherpaOnnxDestroyOnlineRecognizerResult(result) }
        guard let text = result.pointee.text else { return "" }
        return String(cString: text).trimmingCharacters(in: .whitespaces)
    }

    private func finalText(from stream: OpaquePointer?) -> String {
        partialText(from: stream)
    }
}

// MARK: - AudioResampler (48→16 kHz via AVAudioConverter)

private final class AudioResampler {
    private let converter: AVAudioConverter
    private let inFormat:  AVAudioFormat
    private let outFormat: AVAudioFormat

    init(from inRate: Double, to outRate: Double) {
        inFormat  = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                  sampleRate: inRate,  channels: 1, interleaved: false)!
        outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                  sampleRate: outRate, channels: 1, interleaved: false)!
        converter = AVAudioConverter(from: inFormat, to: outFormat)!
    }

    func process(_ buf: AVAudioPCMBuffer) -> [Float]? {
        // Convert to mono Float32 at input rate if needed
        let mono: AVAudioPCMBuffer
        if buf.format == inFormat {
            mono = buf
        } else {
            guard let cv = AVAudioConverter(from: buf.format, to: inFormat),
                  let tmp = AVAudioPCMBuffer(pcmFormat: inFormat,
                                             frameCapacity: buf.frameLength) else { return nil }
            var err: NSError?
            var supplied = false
            cv.convert(to: tmp, error: &err) { _, st in
                if supplied { st.pointee = .noDataNow; return nil }
                supplied = true; st.pointee = .haveData; return buf
            }
            mono = tmp
        }

        let ratio = outFormat.sampleRate / inFormat.sampleRate
        let outFrames = AVAudioFrameCount(Double(mono.frameLength) * ratio) + 64
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: outFrames) else { return nil }

        var supplied = false
        var err: NSError?
        converter.convert(to: outBuf, error: &err) { _, st in
            if supplied { st.pointee = .noDataNow; return nil }
            supplied = true; st.pointee = .haveData; return mono
        }
        guard let ch = outBuf.floatChannelData?[0], outBuf.frameLength > 0 else { return nil }
        return Array(UnsafeBufferPointer(start: ch, count: Int(outBuf.frameLength)))
    }
}
```

- [ ] Commit:
```bash
git add Sources/LiveTranslate/SherpaTranscriber.swift Sources/LiveTranslate/Types.swift
git commit -m "Add SherpaTranscriber: Silero-VAD + zipformer RNN-T + speaker tracking"
```

---

## Task 6: OnnxTTSSpeaker

**Files:**
- Create: `Sources/LiveTranslate/OnnxTTSSpeaker.swift`

- [ ] Create `Sources/LiveTranslate/OnnxTTSSpeaker.swift`:

```swift
import Foundation
import CSherpaOnnx

/// TTS using sherpa-onnx kitten-mini-en-v0_8.
/// Identical public API to the old TTSSpeaker so Pipeline/LiveAudioServer
/// require no changes.
final class OnnxTTSSpeaker: @unchecked Sendable {

    private let onPCM: (Data) -> Void
    private let onActivityChanged: (Bool) -> Void
    private let q = DispatchQueue(label: "OnnxTTSSpeaker.queue")
    private var pending: [String] = []
    private var busy = false
    private let maxQueue = 5
    private let tts: OpaquePointer?

    static func isAvailable() -> Bool {
        FileManager.default.fileExists(atPath: ModelConfig.ttsModelPath)
    }

    init(onPCM: @escaping (Data) -> Void,
         onActivityChanged: @escaping (Bool) -> Void = { _ in }) {
        self.onPCM = onPCM
        self.onActivityChanged = onActivityChanged

        guard Self.isAvailable() else {
            Log.line("OnnxTTSSpeaker: kitten-mini model not found — TTS disabled")
            tts = nil
            return
        }

        var vitsConfig = SherpaOnnxOfflineTtsVitsModelConfig()
        ModelConfig.ttsModelPath.withCString   { p in vitsConfig.model   = p }
        ModelConfig.ttsTokensPath.withCString  { p in vitsConfig.tokens  = p }
        ModelConfig.ttsLexiconPath.withCString { p in vitsConfig.lexicon = p }
        ModelConfig.ttsDataDir.withCString     { p in vitsConfig.data_dir = p }

        var modelConfig = SherpaOnnxOfflineTtsModelConfig()
        modelConfig.vits        = vitsConfig
        modelConfig.num_threads = 2
        ModelConfig.provider.withCString { p in modelConfig.provider = p }

        var config = SherpaOnnxOfflineTtsConfig()
        config.model = modelConfig

        tts = SherpaOnnxCreateOfflineTts(&config)
        Log.line("OnnxTTSSpeaker: kitten-mini loaded")
    }

    deinit { if let t = tts { SherpaOnnxDestroyOfflineTts(t) } }

    func stop() {
        q.async { [weak self] in self?.pending.removeAll() }
    }

    func enqueue(_ text: String) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty, tts != nil else { return }
        q.async { [weak self] in
            guard let self else { return }
            pending.append(t)
            if pending.count > maxQueue { pending.removeFirst(pending.count - maxQueue) }
            pumpLocked()
        }
    }

    // MARK: - Internals

    private func pumpLocked() {
        if !busy { onActivityChanged(false) }
        guard !busy, !pending.isEmpty else { return }
        let text = pending.removeFirst()
        busy = true
        onActivityChanged(true)
        q.async { [weak self] in
            guard let self, let handle = self.tts else {
                self?.busy = false; return
            }
            if let pcm = Self.synthesize(text: text, handle: handle) {
                self.onPCM(pcm)
            }
            self.q.asyncAfter(deadline: .now() + 0.5) {
                self.busy = false
                self.pumpLocked()
            }
        }
    }

    private static func synthesize(text: String, handle: OpaquePointer) -> Data? {
        guard let result = SherpaOnnxOfflineTtsGenerate(handle, text, 0, 1.0) else { return nil }
        defer { SherpaOnnxDestroyOfflineTtsGeneratedAudio(result) }
        let count = Int(result.pointee.n)
        guard count > 0, let samples = result.pointee.samples else { return nil }
        // Convert Float32 samples → PCM16 LE at the TTS sample rate
        let sampleRate = Int(SherpaOnnxOfflineTtsSampleRate(handle))
        _ = sampleRate   // LiveAudioServer uses 24 kHz; we rely on the model being 24 kHz
        var data = Data(capacity: count * 2)
        for i in 0..<count {
            let s = max(-1, min(1, samples[i]))
            let i16 = Int16(s * 32767)
            withUnsafeBytes(of: i16.littleEndian) { data.append(contentsOf: $0) }
        }
        return data
    }
}
```

- [ ] Commit:
```bash
git add Sources/LiveTranslate/OnnxTTSSpeaker.swift
git commit -m "Add OnnxTTSSpeaker: kitten-mini-en-v0_8 via sherpa-onnx"
```

---

## Task 7: Wire Pipeline to new backends

**Files:**
- Modify: `Sources/LiveTranslate/Pipeline.swift`

- [ ] In `Pipeline.swift`, find every reference to `WhisperCppTranscriber` and `TTSSpeaker`. Replace:

Replace the transcriber instantiation (search for `WhisperCppTranscriber()`):
```swift
// OLD:
// let transcriber = WhisperCppTranscriber()
// NEW:
let transcriber = SherpaTranscriber()
```

Replace the TTS speaker instantiation (search for `TTSSpeaker.bestVoice`):
```swift
// OLD:
// if srcLangCode != tgtLangCode,
//    let voice = TTSSpeaker.bestVoice(forTargetCode: tgtLangCode) {
//    let server = LiveAudioServer(port: liveStreamPort)
//    ...
//    let speaker = TTSSpeaker(voice: voice, onPCM: ..., onActivityChanged: ...)
// NEW:
if OnnxTTSSpeaker.isAvailable() {
    let server = LiveAudioServer(port: liveStreamPort)
    do {
        try server.start()
        let speaker = OnnxTTSSpeaker(onPCM: { [weak server] pcm in
            server?.append(pcm)
        }, onActivityChanged: { [weak server] active in
            server?.setSpeaking(active)
        })
        self.liveAudioServer = server
        self.ttsSpeaker = speaker
        self.liveStreamURL = LiveAudioServer.streamURL(port: liveStreamPort)
        Log.line("Live audio stream: \(self.liveStreamURL ?? "?") (kitten-mini)")
    } catch {
        Log.line("LiveAudioServer.start failed: \(error.localizedDescription)")
    }
}
```

Also change the `ttsSpeaker` property type from `TTSSpeaker?` to `OnnxTTSSpeaker?`.

Also remove the language picker if it references `SourceLocale` in UI — the source locale is now fixed at `ModelConfig.sourceLanguageCode`.

- [ ] Commit:
```bash
git add Sources/LiveTranslate/Pipeline.swift
git commit -m "Wire Pipeline to SherpaTranscriber + OnnxTTSSpeaker"
```

---

## Task 8: Update build.sh

**Files:**
- Modify: `build.sh`

- [ ] Read `build.sh` then replace the whisper-build block and add sherpa bundling:

Remove:
```bash
# call tools/build-whisper.sh
# copy model into Resources
```

Add before `swift build`:
```bash
# Ensure sherpa-onnx assets are present
if [[ ! -f "external/sherpa-onnx/lib/libsherpa-onnx-c-api.dylib" ]]; then
  echo "→ running tools/download-sherpa.sh"
  ./tools/download-sherpa.sh
fi
```

After the `swift build` and before app bundling, add:
```bash
# Bundle sherpa-onnx dylibs into Frameworks/
FRAMEWORKS_DIR="build/LiveTranslate.app/Contents/Frameworks"
mkdir -p "${FRAMEWORKS_DIR}"
for dylib in external/sherpa-onnx/lib/*.dylib; do
  cp -f "${dylib}" "${FRAMEWORKS_DIR}/"
done
echo "✓ sherpa-onnx dylibs → Frameworks/"
```

After bundling, add to swift build flags (in the `swift build` invocation):
```bash
-Xlinker -rpath -Xlinker @executable_path/../Frameworks
```

Also add model resources copy:
```bash
# Copy ONNX models into Resources/
RESOURCES_DIR="build/LiveTranslate.app/Contents/Resources"
cp -R build/sherpa-models/* "${RESOURCES_DIR}/"
echo "✓ ONNX models → Resources/"
```

- [ ] Commit:
```bash
git add build.sh
git commit -m "Update build.sh: bundle sherpa dylibs + models, remove whisper"
```

---

## Task 9: First build attempt + fix errors

- [ ] Run the build:
```bash
LIVETRANSLATE_SIGN_IDENTITY=LiveTranslateDev ./build.sh 2>&1 | tee /tmp/build.log | tail -50
```

- [ ] If CSherpaOnnx header issues (symbols not found in c-api.h):
Check that `Sources/CSherpaOnnx/include/sherpa-onnx-c-api.h` was copied correctly and that the module.modulemap references the right filename.

- [ ] If SherpaTranscriber compile errors on C API types:
The sherpa-onnx C struct fields use snake_case. Verify field names against `external/sherpa-onnx/include/sherpa-onnx/c-api/c-api.h` and fix any mismatches.

- [ ] If linker errors (`-lsherpa-onnx-c-api` not found):
Check that `external/sherpa-onnx/lib/` exists and contains the dylib. Verify the lib name with `ls external/sherpa-onnx/lib/`.

- [ ] Commit fixes as they're made.

---

## Task 10: Remove language picker from UI + update docs

**Files:**
- Modify: `Sources/LiveTranslate/TranscriptView.swift`
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] In `TranscriptView.swift`, remove or hide the source language picker (since it's now compile-time). Replace with a static label showing the language pair from `ModelConfig`.

Find the source language Picker in the UI and replace with:
```swift
Text("\(ModelConfig.sourceLanguageCode.uppercased()) → \(ModelConfig.targetLanguageCode.uppercased())")
    .font(.caption)
    .foregroundStyle(.secondary)
```

- [ ] Update `README.md`:
- Remove whisper model section
- Add sherpa-onnx setup: `./tools/download-sherpa.sh` (one-time)
- Document compile-time language change: edit `ModelConfig.swift`
- Remove Premium TTS voice installation instructions (kitten-mini is bundled)

- [ ] Update `CLAUDE.md`:
- Replace WhisperCppTranscriber file description with SherpaTranscriber
- Replace TTSSpeaker with OnnxTTSSpeaker
- Update architecture diagram
- Add lesson about sherpa-onnx C API bridging approach
- Remove whisper-specific lessons that no longer apply

- [ ] Commit:
```bash
git add Sources/LiveTranslate/TranscriptView.swift README.md CLAUDE.md
git commit -m "Remove language picker UI; update docs for sherpa-onnx"
```

---

## Task 11: Smoke test + final polish

- [ ] Open the app:
```bash
open build/LiveTranslate.app
tail -f /tmp/livetranslate.log
```

- [ ] Verify in log:
  - `SherpaTranscriber: models loaded (provider=coreml)`
  - `OnnxTTSSpeaker: kitten-mini loaded`
  - `LiveAudioServer: listening on :8765`

- [ ] Speak German into mic → verify text appears in German → translation appears in English.

- [ ] Connect phone to `http://<lan-ip>:8765/` → verify English TTS audio plays.

- [ ] If CoreML provider fails (model not supported), change `ModelConfig.provider` to `"cpu"` and rebuild.

- [ ] Commit any final fixes:
```bash
git add -A
git commit -m "Fix smoke test issues; working MVP"
```

- [ ] Push branch:
```bash
git push -u origin feature/onnx-streaming
```
