import Foundation
import AVFoundation
import CSherpaOnnx

/// Synthesizes translated text to 24 kHz mono PCM16 LE via
/// `kitten-mini-en-v0_8` (sherpa-onnx offline TTS).
///
/// Drop-in replacement for `TTSSpeaker` from the caller's perspective —
/// same `enqueue(_:)`, `stop()`, and `isAvailable()` API.
///
/// Serial queue: utterances are synthesized one at a time in FIFO order.
/// Back-pressure: if more than `maxQueue` are pending the **oldest** are
/// dropped (same policy as TTSSpeaker — a listener tolerates gaps better
/// than ever-growing lag).
final class OnnxTTSSpeaker: @unchecked Sendable {

    // MARK: — Availability

    /// True when the kitten-mini model.onnx file exists in the app bundle.
    static func isAvailable() -> Bool {
        let path = Bundle.main.bundleURL
            .appendingPathComponent("Contents/Resources")
            .appendingPathComponent(ModelConfig.ttsModel)
            .path
        return FileManager.default.fileExists(atPath: path)
    }

    // MARK: — Private state

    private let onPCM: (Data) -> Void
    private let onActivityChanged: (Bool) -> Void

    private let q = DispatchQueue(label: "OnnxTTSSpeaker.queue")
    private var pending: [String] = []
    private var busy: Bool = false
    private let maxQueue = 5

    private var tts: OpaquePointer?
    private var sampleRate: Int32 = 24_000

    // MARK: — Init

    init(onPCM: @escaping (Data) -> Void,
         onActivityChanged: @escaping (Bool) -> Void = { _ in }) {
        self.onPCM = onPCM
        self.onActivityChanged = onActivityChanged

        q.async { [weak self] in
            self?.loadTTS()
        }
    }

    deinit {
        if let t = tts { SherpaOnnxDestroyOfflineTts(t) }
    }

    // MARK: — Public interface

    func enqueue(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        q.async { [weak self] in
            guard let self else { return }
            pending.append(trimmed)
            if pending.count > maxQueue {
                let drop = pending.count - maxQueue
                pending.removeFirst(drop)
                Log.line("OnnxTTSSpeaker: dropped \(drop) oldest pending (backpressure)")
            }
            pumpLocked()
        }
    }

    func stop() {
        q.async { [weak self] in
            self?.pending.removeAll()
        }
    }

    // MARK: — Private

    private func loadTTS() {
        let modelPath   = resourcePath(ModelConfig.ttsModel)
        let voicesPath  = resourcePath(ModelConfig.ttsVoices)
        let tokensPath  = resourcePath(ModelConfig.ttsTokens)
        let dataDir     = resourcePath(ModelConfig.ttsDataDir)

        var kitten = SherpaOnnxOfflineTtsKittenModelConfig()
        kitten.length_scale = 1.0   // normal speed

        var model = SherpaOnnxOfflineTtsModelConfig()
        model.num_threads = 2
        model.debug       = 0

        var cfg = SherpaOnnxOfflineTtsConfig()
        cfg.max_num_sentences = 1

        guard let t = modelPath.withCString({ mP -> OpaquePointer? in
            kitten.model = mP
            return voicesPath.withCString { vP -> OpaquePointer? in
                kitten.voices = vP
                return tokensPath.withCString { tP -> OpaquePointer? in
                    kitten.tokens = tP
                    return dataDir.withCString { dP -> OpaquePointer? in
                        kitten.data_dir = dP
                        model.kitten = kitten
                        cfg.model = model
                        return SherpaOnnxCreateOfflineTts(&cfg)
                    }
                }
            }
        }) else {
            Log.line("OnnxTTSSpeaker: SherpaOnnxCreateOfflineTts failed — check model bundle")
            return
        }

        tts = t
        sampleRate = SherpaOnnxOfflineTtsSampleRate(t)
        Log.line("OnnxTTSSpeaker: kitten-mini loaded, sampleRate=\(sampleRate) Hz")
    }

    private func pumpLocked() {
        if !busy { onActivityChanged(false) }
        guard !busy, !pending.isEmpty, tts != nil else { return }
        let text = pending.removeFirst()
        busy = true
        speak(text) { [weak self] in
            self?.q.asyncAfter(deadline: .now() + 0.5) {
                self?.busy = false
                self?.pumpLocked()
            }
        }
    }

    /// Synthesize `text` synchronously on the serial queue, convert the
    /// generated Float32 samples to 24 kHz PCM16 LE, forward to `onPCM`.
    private func speak(_ text: String, done: @escaping () -> Void) {
        guard let t = tts else { done(); return }

        onActivityChanged(true)

        var genCfg = SherpaOnnxGenerationConfig()
        genCfg.speed  = 1.0
        genCfg.sid    = 2   // expr-voice-3-m

        guard let audio = text.withCString({ cText -> UnsafePointer<SherpaOnnxGeneratedAudio>? in
            withUnsafeMutablePointer(to: &genCfg) { cfgPtr in
                SherpaOnnxOfflineTtsGenerateWithConfig(t, cText, cfgPtr, nil, nil)
            }
        }) else {
            Log.line("OnnxTTSSpeaker: SherpaOnnxOfflineTtsGenerateWithConfig returned nil for \"\(text.prefix(40))\"")
            done()
            return
        }
        defer { SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio) }

        let n = Int(audio.pointee.n)
        guard n > 0, let rawSamples = audio.pointee.samples else { done(); return }

        let generatedRate = Double(audio.pointee.sample_rate)

        // Convert Float32 at generatedRate → PCM16 LE at 24 kHz.
        if let pcm = resampleToPCM16(
            floats: rawSamples, count: n,
            fromRate: generatedRate, toRate: 24_000
        ) {
            onPCM(pcm)
        }

        done()
    }

    // MARK: — Format conversion

    /// Wrap the raw Float32 samples from sherpa-onnx into an AVAudioPCMBuffer,
    /// convert to 24 kHz mono PCM16 LE, return as Data.
    private func resampleToPCM16(
        floats: UnsafePointer<Float>, count: Int,
        fromRate: Double, toRate: Double
    ) -> Data? {
        guard let srcFmt = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: fromRate, channels: 1, interleaved: false
        ) else { return nil }

        guard let dstFmt = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: toRate, channels: 1, interleaved: true
        ) else { return nil }

        guard let srcBuf = AVAudioPCMBuffer(pcmFormat: srcFmt,
                                             frameCapacity: AVAudioFrameCount(count))
        else { return nil }
        srcBuf.frameLength = AVAudioFrameCount(count)
        if let ptr = srcBuf.floatChannelData?[0] {
            ptr.initialize(from: floats, count: count)
        }

        guard let conv = AVAudioConverter(from: srcFmt, to: dstFmt) else { return nil }

        let ratio = toRate / fromRate
        let outCap = AVAudioFrameCount(Double(count) * ratio) + 1024
        guard let dstBuf = AVAudioPCMBuffer(pcmFormat: dstFmt, frameCapacity: outCap)
        else { return nil }

        var supplied = false
        var err: NSError?
        _ = conv.convert(to: dstBuf, error: &err) { _, status in
            if supplied { status.pointee = .noDataNow; return nil }
            supplied = true; status.pointee = .haveData; return srcBuf
        }

        // Flush resampler tail.
        if let flushBuf = AVAudioPCMBuffer(pcmFormat: dstFmt, frameCapacity: 4096) {
            _ = conv.convert(to: flushBuf, error: nil) { _, status in
                status.pointee = .endOfStream; return nil
            }
            // Append flush output to dstBuf via raw bytes.
            if flushBuf.frameLength > 0, let ch = flushBuf.int16ChannelData?[0] {
                let extra = Int(flushBuf.frameLength) * MemoryLayout<Int16>.size
                var data = dataFromInt16(dstBuf)
                data?.append(Data(bytes: ch, count: extra))
                return data
            }
        }

        return dataFromInt16(dstBuf)
    }

    private func dataFromInt16(_ buf: AVAudioPCMBuffer) -> Data? {
        guard buf.frameLength > 0, let ch = buf.int16ChannelData?[0] else { return nil }
        return Data(bytes: ch, count: Int(buf.frameLength) * MemoryLayout<Int16>.size)
    }

    private func resourcePath(_ relative: String) -> String {
        Bundle.main.bundleURL
            .appendingPathComponent("Contents/Resources")
            .appendingPathComponent(relative)
            .path
    }
}
