import Foundation
import AVFoundation

/// Streams one denoised audio source straight to a `.wav` file on disk.
/// One instance per source per run, written under the temp work-dir
/// (later moved into the per-run zip).
///
/// Stored format is **48 kHz mono signed-16-bit PCM** — RNNoise's native
/// rate. `AVAudioFile` downcasts Float32 buffers to Int16 on the write
/// path. The 48 kHz files are universally playable and ffmpeg muxes them
/// straight into the MKV without resampling.
///
/// Writes are serialized on a private queue so the MainActor — which is
/// where ingest runs — never blocks on disk IO.
final class AudioRecorder {
    let url: URL
    private var file: AVAudioFile?
    private let queue = DispatchQueue(label: "AudioRecorder.write", qos: .utility)

    /// Open a file for writing. Throws if the directory isn't writable or
    /// the audio format isn't supported by Core Audio (extremely rare).
    init(at url: URL) throws {
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 48_000,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ]
        self.file = try AVAudioFile(forWriting: url, settings: settings)
        self.url = url
    }

    /// Append one PCM buffer. Returns immediately; the actual disk write
    /// happens on `queue`. `AVAudioFile.write(from:)` transparently
    /// converts the buffer's Float32 frames to the file's Int16 format.
    func append(_ buffer: AVAudioPCMBuffer) {
        queue.async { [weak self] in
            guard let file = self?.file else { return }
            do {
                try file.write(from: buffer)
            } catch {
                Log.line("AudioRecorder: write failed: \(error.localizedDescription)")
            }
        }
    }

    /// Block until every queued write has hit disk, then close the
    /// underlying `AVAudioFile` so the WAV header's data-chunk length
    /// gets finalized. Without the close, `AVAudioFile(forReading:)`
    /// (used downstream by `MKVExporter` to probe duration) sees a
    /// stale header — the writer only finalizes on deinit. That
    /// stale duration was making lavfi produce 0.5s of video for a
    /// 10s audio file.
    func flush() {
        queue.sync {
            self.file = nil
        }
    }
}
