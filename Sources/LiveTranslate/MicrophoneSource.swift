import Foundation
import AVFoundation

/// Microphone capture via `AVAudioEngine`. Emits 48 kHz mono Float32
/// — RNNoise's native rate. Downstream consumers (`DenoisingAudioSource`,
/// `SherpaTranscriber`, `AudioRecorder`) resample from this single rate
/// as needed.
///
/// We deliberately do NOT enable AVAudioEngine's `VoiceProcessingIO`
/// unit. It bundles AEC + noise suppression + AGC, but on macOS it also
/// flips the audio session into a voice-chat mode that ducks every
/// other audio source the user can hear — so they can't actually hear
/// the system audio they're capturing while recording.
final class MicrophoneSource: AudioSource {
    private let engine = AVAudioEngine()
    private var tapInstalled = false

    /// 48 kHz mono Float32 — RNNoise's native rate. Downstream consumers
    /// resample to whatever they need (sherpa-onnx works at 16 kHz; the
    /// `AudioRecorder` writes 48 kHz directly).
    private let targetFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 48_000,
        channels: 1,
        interleaved: false
    )!
    private var converter: AVAudioConverter?
    private var sourceFormat: AVAudioFormat?

    private let broadcaster = BufferBroadcaster()
    var buffers: AsyncStream<AVAudioPCMBuffer> { broadcaster.stream }

    func start() async throws {
        guard !engine.isRunning else { return }
        let input = engine.inputNode
        if tapInstalled {
            input.removeTap(onBus: 0)
            tapInstalled = false
        }
        let native = input.outputFormat(forBus: 0)
        sourceFormat = native
        converter = AVAudioConverter(from: native, to: targetFormat)

        // 512 samples = ~10.7 ms at 48 kHz — half the previous 1024 (~21 ms)
        // so partial ASR updates and endpoint detection see audio ~10 ms
        // sooner. AVAudioEngine bills this as a hint, not a guarantee, but
        // on M-series the buffer ends up at or near the requested size.
        input.installTap(onBus: 0, bufferSize: 512, format: native) { [weak self] buf, _ in
            guard let self, let converted = self.convert(buf) else { return }
            self.broadcaster.emit(converted)
        }
        tapInstalled = true
        engine.prepare()
        try engine.start()
        Log.line("Mic: started, native=\(native), target=\(targetFormat)")
    }

    func stop() async {
        if tapInstalled {
            engine.inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }
        if engine.isRunning { engine.stop() }
        // Close every live `buffers` subscription so consumers' for-await
        // loops exit naturally. This is the audio-side signal that drives
        // the Pipeline's graceful drain on Stop.
        broadcaster.finishAll()
        Log.line("Mic: stopped")
    }

    /// Convert one input-format buffer to the target format. Returns nil
    /// on conversion error; very rare in practice.
    private func convert(_ src: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        guard let converter, let sourceFormat else { return nil }
        let outCapacity = AVAudioFrameCount(
            Double(src.frameLength) * targetFormat.sampleRate / sourceFormat.sampleRate
        ) + 1024
        guard let out = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outCapacity) else { return nil }
        var didFeed = false
        let status = converter.convert(to: out, error: nil) { _, outStatus in
            if didFeed { outStatus.pointee = .noDataNow; return nil }
            didFeed = true
            outStatus.pointee = .haveData
            return src
        }
        return status == .error ? nil : out
    }
}
