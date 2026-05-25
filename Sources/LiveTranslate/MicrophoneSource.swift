import Foundation
import AVFoundation

/// Microphone capture via `AVAudioEngine`. Emits 48 kHz mono Float32
/// — the rate RNNoise expects natively (see `RNNoiseProcessor`).
/// Downstream consumers (`SherpaTranscriber`, `AudioRecorder`) resample
/// from this single common rate as needed.
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
        // Enable AVAudioEngine's VoiceProcessingIO unit — built-in AEC,
        // noise suppression, and AGC, all in hardware on Apple Silicon
        // (zero added CPU). This replaces the RNNoise denoise + manual
        // AGC stage that used to live in DenoisingAudioSource. The
        // engine's input format may change after enabling; we re-read it
        // below before installing the tap.
        do {
            try input.setVoiceProcessingEnabled(true)
            // Don't bypass on output (we're recording mic, not playback).
            input.isVoiceProcessingBypassed = false
            // Hardware AGC inside VP unit — leave it on; user controls
            // mic gain via macOS Sound preferences.
            input.isVoiceProcessingAGCEnabled = true
        } catch {
            // Voice processing isn't fatal — fall back to raw input.
            Log.line("Mic: setVoiceProcessingEnabled failed (\(error.localizedDescription)) — using raw input")
        }
        let native = input.outputFormat(forBus: 0)
        sourceFormat = native
        converter = AVAudioConverter(from: native, to: targetFormat)

        input.installTap(onBus: 0, bufferSize: 1024, format: native) { [weak self] buf, _ in
            guard let self, let converted = self.convert(buf) else { return }
            self.broadcaster.emit(converted)
        }
        tapInstalled = true
        engine.prepare()
        try engine.start()
        Log.line("Mic: started (voice-processing), native=\(native), target=\(targetFormat)")
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
