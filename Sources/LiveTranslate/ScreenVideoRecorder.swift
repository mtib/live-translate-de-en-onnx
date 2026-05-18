import Foundation
import AVFoundation
import ScreenCaptureKit
import CoreMedia
import CoreVideo

/// Optional screen-recording sidecar.
///
/// Runs a dedicated `SCStream` (separate from `SystemAudioSource` so audio
/// stays display-wide while video targets whatever the user picked) and
/// pipes the `.screen` sample buffers into an `AVAssetWriter` writing an
/// H.264 `.mov` at 10 fps / 1280×720 / 500 kbps inside the per-session
/// work directory.
///
/// Time alignment: SCK / `AVAssetWriter` track *wall-clock host-time*
/// PTS. The recorder may start later than the audio (SCK startup, user
/// reaction) or end earlier (target window closed mid-session) — we
/// record `firstFrameWallClock - runStartedAt` to a sidecar text file
/// alongside the `.mov`. `MKVExporter` reads that offset and inserts it
/// with `-itsoffset` so the screen video lines up with the audio
/// timeline even when the recording doesn't span the whole session.
///
/// Failure modes (filter went stale, screen-rec TCC revoked, writer
/// hiccup) all leave `screenVideo` absent from the work dir — the rest
/// of the session output is unchanged and `MKVExporter` falls back to
/// the lavfi black-frame video track. This is by design: nothing about
/// the screen recorder is allowed to break the rest of the pipeline.
final class ScreenVideoRecorder: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {

    /// Hard cap; the actual writer dimensions are picked per segment
    /// to match the chosen target's source aspect (see `pickDimensions`).
    /// Keeping the writer at the source aspect means SCK never has to
    /// letterbox — any padding is done by ffmpeg at compose time,
    /// where it can be centered. SCK's own letterbox origin-aligns
    /// the content, which is why we avoid it.
    static let maxWidth = 1280
    static let maxHeight = 720
    static let fps: Int32 = 10
    static let bitrate = 500_000

    private let filter: SCContentFilter
    private let outputURL: URL
    private let offsetURL: URL
    private let runStartedAt: Date

    /// Per-segment writer/capture dimensions, chosen at `start()`.
    private var width: Int = ScreenVideoRecorder.maxWidth
    private var height: Int = ScreenVideoRecorder.maxHeight

    private var stream: SCStream?
    private let sampleQueue = DispatchQueue(label: "ScreenVideoRecorder.samples", qos: .userInteractive)

    /// Set to `true` in `stop()` before calling `stopCapture()` so that
    /// `didStopWithError` knows not to attempt a reconnect on a user-initiated stop.
    private var intentionalStop = false

    /// Called after the writer is finalized when the SCK stream stops
    /// system-initiated (i.e. NOT via `stop()`). Pipeline wires this to
    /// open a new segment so screen recording auto-resumes.
    var onSystemStop: (() -> Void)?

    private var writer: AVAssetWriter?
    private var writerInput: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    /// Writer-state lock — `didOutputSampleBuffer` runs on `sampleQueue`
    /// and `stop()` runs on the MainActor; both touch the writer.
    private let writerLock = NSLock()
    private var startedSession = false
    private var finished = false
    private var firstFramePTS: CMTime = .invalid

    init(filter: SCContentFilter, outputURL: URL, offsetURL: URL, runStartedAt: Date) {
        self.filter = filter
        self.outputURL = outputURL
        self.offsetURL = offsetURL
        self.runStartedAt = runStartedAt
        super.init()
    }

    /// Spin up the writer + SCStream. Throws if AVAssetWriter setup or
    /// stream start fails — the caller logs and continues without
    /// screen recording.
    func start() async throws {
        // Pick capture dimensions at the source's aspect ratio (capped
        // to 1280×720). This avoids SCK's letterbox, which is origin-
        // aligned — so a 16:9 buffer holding a 4:3 source had the
        // content jammed top-left with black bars on right + bottom.
        // With source-aspect dims, SCK scales the source 1:1 into the
        // buffer (no letterbox needed) and ffmpeg's pad-and-center
        // step composes it onto the final 1280×720 canvas centered,
        // filling either horizontally or vertically as appropriate.
        //
        // If the target window then resizes mid-segment, SCK
        // letterboxes the new content into our locked dims —
        // origin-aligned, but that's the rare case and the simplest
        // recovery is to stop/start the segment from the popover.
        let (w, h) = Self.pickDimensions(for: filter)
        self.width = w
        self.height = h
        try setupWriter()

        let cfg = SCStreamConfiguration()
        cfg.width = w
        cfg.height = h
        cfg.minimumFrameInterval = CMTime(value: 1, timescale: Self.fps)
        cfg.pixelFormat = kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        cfg.queueDepth = 5
        cfg.capturesAudio = false
        cfg.scalesToFit = true
        cfg.preservesAspectRatio = true
        cfg.showsCursor = true

        let stream = SCStream(filter: filter, configuration: cfg, delegate: self)
        try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: sampleQueue)
        try await stream.startCapture()
        self.stream = stream
        Log.line("ScreenVideoRecorder: capture started \(w)×\(h) → \(outputURL.lastPathComponent)")
    }

    /// Choose writer/capture dimensions for this segment: scale the
    /// source's content rect down to fit within `maxWidth`×`maxHeight`
    /// while preserving aspect, snapped to even integers (h.264
    /// wants even dims). Falls back to the cap if the filter exposes
    /// no content rect.
    private static func pickDimensions(for filter: SCContentFilter) -> (Int, Int) {
        let rect = filter.contentRect
        let scale = max(1, CGFloat(filter.pointPixelScale))
        var srcW = Double(rect.width * scale)
        var srcH = Double(rect.height * scale)
        if !srcW.isFinite || srcW <= 0 || !srcH.isFinite || srcH <= 0 {
            srcW = Double(maxWidth)
            srcH = Double(maxHeight)
        }
        let factor = min(Double(maxWidth) / srcW, Double(maxHeight) / srcH, 1.0)
        var w = Int((srcW * factor).rounded())
        var h = Int((srcH * factor).rounded())
        w -= w % 2
        h -= h % 2
        w = max(w, 16)
        h = max(h, 16)
        return (w, h)
    }

    /// Stop the SCK stream, finalize the writer, write the offset
    /// sidecar (so MKVExporter can shift the video to the right spot
    /// on the audio timeline). Idempotent; safe to call from `defer`.
    func stop() async {
        guard let stream else { return }
        // Mark as intentional BEFORE nilling stream and calling stopCapture
        // so that didStopWithError (if it fires) knows not to reconnect.
        intentionalStop = true
        self.stream = nil
        do { try await stream.stopCapture() }
        catch { Log.line("ScreenVideoRecorder: stopCapture error: \(error)") }
        await finalizeWriter()
        intentionalStop = false
    }

    // MARK: - Writer setup

    private func setupWriter() throws {
        // Wipe any zero-byte file left from a previous failed attempt
        // — AVAssetWriter init fails on an existing path.
        try? FileManager.default.removeItem(at: outputURL)

        let writer = try AVAssetWriter(url: outputURL, fileType: .mov)
        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: Self.bitrate,
                AVVideoMaxKeyFrameIntervalKey: Int(Self.fps) * 5,  // 5 s keyframe interval
                AVVideoProfileLevelKey: AVVideoProfileLevelH264MainAutoLevel,
                AVVideoAllowFrameReorderingKey: false,
            ] as [String: Any]
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        input.expectsMediaDataInRealTime = true

        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
            ]
        )
        guard writer.canAdd(input) else {
            throw NSError(domain: "ScreenVideoRecorder", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "Cannot add video input to writer"])
        }
        writer.add(input)
        self.writer = writer
        self.writerInput = input
        self.adaptor = adaptor
    }

    // MARK: - SCStreamOutput

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen,
              sampleBuffer.isValid,
              CMSampleBufferDataIsReady(sampleBuffer),
              let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
        else { return }

        // Drop frames marked as "idle" / non-content by SCK (status
        // attachment). Without this the writer sees every screen
        // refresh tick at 10 fps regardless of whether the content
        // actually changed — fine for our use case, but also fine
        // to skip the no-op frames.
        if let attachments = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
           let info = attachments.first,
           let statusRaw = info[.status] as? Int,
           let status = SCFrameStatus(rawValue: statusRaw),
           status != .complete && status != .started {
            return
        }

        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        guard pts.isValid else { return }

        writerLock.lock()
        defer { writerLock.unlock() }
        guard !finished, let writer, let input = writerInput, let adaptor else { return }

        if !startedSession {
            // First frame: start the writer + session at this PTS, record
            // wall-clock offset to runStartedAt for MKVExporter to use.
            guard writer.startWriting() else {
                Log.line("ScreenVideoRecorder: startWriting failed: \(writer.error?.localizedDescription ?? "?")")
                finished = true
                return
            }
            writer.startSession(atSourceTime: pts)
            firstFramePTS = pts
            startedSession = true
            // Wall-clock offset of this first frame relative to
            // runStartedAt. Clamp negative to 0 — pipeline starts audio
            // first so this should always be ≥ 0, but defensive.
            let offset = max(0, Date().timeIntervalSince(runStartedAt))
            writeOffset(offset)
            Log.line(String(format: "ScreenVideoRecorder: first frame, offset=%.3fs", offset))
        }

        guard input.isReadyForMoreMediaData else { return }
        if !adaptor.append(imageBuffer, withPresentationTime: pts) {
            Log.line("ScreenVideoRecorder: append failed: \(writer.error?.localizedDescription ?? "?")")
        }
    }

    // MARK: - SCStreamDelegate

    /// Mid-session stop — either the window/target disappeared, or macOS
    /// terminated the stream (e.g. during a Space transition on macOS 26).
    /// Finalize whatever we've written so the partial `.mov` is valid up to
    /// the last frame, then fire `onSystemStop` if this wasn't user-initiated
    /// so Pipeline can open a new segment and resume recording.
    func stream(_ stream: SCStream, didStopWithError error: Error) {
        Log.line("ScreenVideoRecorder: stream stopped: \(error.localizedDescription)")
        // Guard against user-initiated stops: `stop()` sets `intentionalStop = true`
        // before calling `stopCapture()`, which is what fires this delegate.
        guard !intentionalStop else { return }
        self.stream = nil
        let callback = onSystemStop
        Task {
            await finalizeWriter()
            callback?()
        }
    }

    // MARK: - Finalize

    private func finalizeWriter() async {
        // Mark + extract under the writer lock from the sample queue.
        // Doing the lock acquisition on the sample queue (rather than
        // the awaiting caller) keeps NSLock out of an async context —
        // Swift 6 forbids it.
        let snapshot: (AVAssetWriter, AVAssetWriterInput)? = await withCheckedContinuation { cont in
            sampleQueue.async {
                self.writerLock.lock()
                defer { self.writerLock.unlock() }
                guard !self.finished, let w = self.writer, let i = self.writerInput else {
                    cont.resume(returning: nil); return
                }
                self.finished = true
                guard self.startedSession else {
                    cont.resume(returning: nil); return
                }
                i.markAsFinished()
                cont.resume(returning: (w, i))
            }
        }
        guard let (writer, _) = snapshot else {
            if writer == nil {
                // Nothing to log
            } else {
                Log.line("ScreenVideoRecorder: no frames captured; no .mov produced")
            }
            return
        }
        await writer.finishWriting()
        if writer.status == .completed {
            Log.line("ScreenVideoRecorder: finalized \(outputURL.lastPathComponent)")
        } else {
            Log.line("ScreenVideoRecorder: writer ended with status=\(writer.status.rawValue) error=\(writer.error?.localizedDescription ?? "nil")")
        }
    }

    private func writeOffset(_ seconds: Double) {
        let payload = String(format: "%.6f\n", seconds)
        try? payload.data(using: .utf8)?.write(to: offsetURL, options: .atomic)
    }
}
