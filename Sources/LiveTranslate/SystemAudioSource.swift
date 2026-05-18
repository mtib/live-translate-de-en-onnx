import Foundation
import AVFoundation
import ScreenCaptureKit
import CoreMedia

/// Captures audio playing through the system speakers via
/// `ScreenCaptureKit`. Bypasses the speaker→mic round-trip, so video
/// or call audio reaches the recognizer clean.
///
/// Requirements:
///   - macOS 13+ (we target 15).
///   - **Screen Recording permission** in System Settings → Privacy &
///     Security → Screen Recording. macOS prompts on first start.
///
/// `ScreenCaptureKit` is screen-capture-first but supports audio-only
/// configurations: we just never subscribe to `.screen` samples.
final class SystemAudioSource: NSObject, AudioSource, SCStreamOutput, SCStreamDelegate {

    private var stream: SCStream?
    private let sampleQueue = DispatchQueue(label: "SystemAudioSource.samples", qos: .userInteractive)

    /// Set to `true` in `stop()` before calling `stopCapture()` so that
    /// `didStopWithError` knows not to attempt a reconnect on a user-initiated stop.
    private var intentionalStop = false

    /// In-flight reconnect task; cancelled on `stop()`.
    private var reconnectTask: Task<Void, Never>?

    private let broadcaster = BufferBroadcaster()
    var buffers: AsyncStream<AVAudioPCMBuffer> { broadcaster.stream }

    /// Lazy converter: SCK source format → 48 kHz mono Float32 to match
    /// MicrophoneSource. The whole pipeline runs at 48 kHz because that
    /// is RNNoise's native rate (see RNNoiseProcessor).
    private var converter: AVAudioConverter?
    private var sourceFormat: AVAudioFormat?
    private let targetFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 48_000,
        channels: 1,
        interleaved: false
    )!

    /// SCK setup is natively async — never wrap with a `DispatchSemaphore`
    /// (deadlocks the MainActor, see CLAUDE.md lesson #9).
    func start() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        guard let display = content.displays.first else { throw SystemAudioError.noDisplay }
        let filter = SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])

        let cfg = SCStreamConfiguration()
        cfg.capturesAudio = true
        cfg.excludesCurrentProcessAudio = true
        cfg.sampleRate = 48_000
        cfg.channelCount = 2
        // SCK requires *some* video config; we keep it tiny and ignore the samples.
        cfg.width = 2
        cfg.height = 2
        cfg.minimumFrameInterval = CMTime(value: 1, timescale: 1)

        let stream = SCStream(filter: filter, configuration: cfg, delegate: self)
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: sampleQueue)
        try await stream.startCapture()
        self.stream = stream
        Log.line("SystemAudio: capture started")
    }

    /// Async so the caller can `await` the SCK teardown — without this,
    /// rapid Stop→Start cycles could race the new stream against the
    /// not-yet-stopped previous one.
    func stop() async {
        intentionalStop = true
        // Cancel any in-flight reconnect so it doesn't race the teardown.
        reconnectTask?.cancel()
        reconnectTask = nil
        let streamToStop = self.stream
        self.stream = nil
        if let streamToStop {
            do { try await streamToStop.stopCapture() }
            catch { Log.line("SystemAudio: stopCapture error: \(error)") }
        }
        // Close every live `buffers` subscription so consumers' for-await
        // loops exit naturally on Stop.
        broadcaster.finishAll()
        intentionalStop = false
        Log.line("SystemAudio: capture stopped")
    }

    // MARK: - SCStreamOutput

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio,
              sampleBuffer.isValid,
              CMSampleBufferDataIsReady(sampleBuffer)
        else { return }
        guard let pcm = makePCMBuffer(from: sampleBuffer) else { return }
        broadcaster.emit(pcm)
    }

    // MARK: - SCStreamDelegate

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        Log.line("SystemAudio: stream stopped with error: \(error.localizedDescription)")
        // Guard against user-initiated stops: `stop()` sets `intentionalStop = true`
        // before calling `stopCapture()`, which is what fires this delegate.
        guard !intentionalStop else { return }
        // System stopped the stream unexpectedly — nil out the dead stream and
        // attempt to restart it so the downstream pipeline keeps receiving audio.
        self.stream = nil
        reconnectTask?.cancel()
        reconnectTask = Task { [weak self] in
            await self?.attemptReconnect()
        }
    }

    // MARK: - Reconnect

    /// Attempts to restart SCK audio capture after a system-initiated stop.
    /// Uses exponential backoff (1 s → 2 s → 4 s → 8 s → 16 s) before giving up.
    /// The `broadcaster` is deliberately NOT finished between attempts so that
    /// downstream consumers (DenoisingAudioSource for-await loops) remain alive
    /// and resume receiving audio as soon as capture restarts. `finishAll()` is
    /// only called after all attempts are exhausted.
    private func attemptReconnect() async {
        let delays: [Duration] = [
            .seconds(1), .seconds(2), .seconds(4), .seconds(8), .seconds(16)
        ]
        for (index, delay) in delays.enumerated() {
            guard !Task.isCancelled else { return }
            Log.line("SystemAudio: reconnect attempt \(index + 1)/\(delays.count), waiting \(delay)…")
            try? await Task.sleep(for: delay)
            guard !Task.isCancelled else { return }
            do {
                try await start()
                Log.line("SystemAudio: reconnected successfully (attempt \(index + 1))")
                return
            } catch {
                Log.line("SystemAudio: reconnect attempt \(index + 1) failed: \(error.localizedDescription)")
            }
        }
        Log.line("SystemAudio: all reconnect attempts exhausted — closing broadcaster")
        broadcaster.finishAll()
    }

    // MARK: - Sample conversion

    /// Convert one `CMSampleBuffer` from SCK into a 16 kHz mono Float32
    /// `AVAudioPCMBuffer`. SCK typically delivers 48 kHz stereo Float32;
    /// `AVAudioConverter` handles downsample + downmix.
    private func makePCMBuffer(from sample: CMSampleBuffer) -> AVAudioPCMBuffer? {
        guard let formatDesc = sample.formatDescription,
              let asbd = formatDesc.audioStreamBasicDescription else { return nil }

        if converter == nil
            || sourceFormat?.sampleRate != asbd.mSampleRate
            || sourceFormat?.channelCount != AVAudioChannelCount(asbd.mChannelsPerFrame) {
            var asbdCopy = asbd
            guard let src = AVAudioFormat(streamDescription: &asbdCopy) else { return nil }
            self.sourceFormat = src
            self.converter = AVAudioConverter(from: src, to: targetFormat)
        }
        guard let converter, let sourceFormat else { return nil }

        guard let srcBuffer = AVAudioPCMBuffer.fromCMSampleBuffer(sample, format: sourceFormat) else {
            return nil
        }

        let targetCapacity = AVAudioFrameCount(
            Double(srcBuffer.frameLength) * targetFormat.sampleRate / sourceFormat.sampleRate
        ) + 1024
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: targetCapacity) else {
            return nil
        }

        var didFeed = false
        let status = converter.convert(to: outBuffer, error: nil) { _, outStatus in
            if didFeed { outStatus.pointee = .noDataNow; return nil }
            didFeed = true
            outStatus.pointee = .haveData
            return srcBuffer
        }
        return status == .error ? nil : outBuffer
    }
}

enum SystemAudioError: LocalizedError {
    case noDisplay
    var errorDescription: String? {
        switch self {
        case .noDisplay: return "No display available for ScreenCaptureKit."
        }
    }
}

// MARK: - Helpers

private extension AVAudioPCMBuffer {
    /// Build an `AVAudioPCMBuffer` from a `CMSampleBuffer`. Copies into a
    /// fresh buffer so the lifetime is independent of CoreMedia (which
    /// may reclaim the sample buffer once the delegate returns).
    ///
    /// Uses `CMSampleBufferCopyPCMDataIntoAudioBufferList`, which writes
    /// directly into the destination's `mutableAudioBufferList` — that's
    /// already correctly sized for the destination format (one buffer for
    /// interleaved, N buffers for non-interleaved with N channels). The
    /// earlier `GetAudioBufferListWithRetainedBlockBuffer` approach with
    /// a fixed-size `AudioBufferList` failed on non-interleaved stereo —
    /// see CLAUDE.md lesson #13.
    static func fromCMSampleBuffer(_ sample: CMSampleBuffer, format: AVAudioFormat) -> AVAudioPCMBuffer? {
        let numSamples = AVAudioFrameCount(CMSampleBufferGetNumSamples(sample))
        guard numSamples > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: numSamples)
        else { return nil }
        buffer.frameLength = numSamples
        let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sample,
            at: 0,
            frameCount: Int32(numSamples),
            into: buffer.mutableAudioBufferList
        )
        return status == noErr ? buffer : nil
    }
}
