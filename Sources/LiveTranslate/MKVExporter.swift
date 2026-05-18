import Foundation
import AVFoundation

/// Shell out to ffmpeg to build the per-session MKV from the work dir
/// contents (per-source WAVs + already-merged SRTs). No SRT merging
/// here — `MergedSubtitleArchive` maintains those files live during
/// the session, so `ffmpeg` just embeds the existing files.
///
/// 640×360 black background, both WAVs `amix`'d into one audio track,
/// per-language merged SRT embedded with `deu`/`eng`/etc. metadata.
/// First subtitle track is marked default so VLC picks it up
/// automatically.
///
/// No-op (logs a hint) if ffmpeg isn't installed; the rest of the
/// session output (WAVs, SRTs, JSONL) still ends up in the zip.
enum MKVExporter {

    /// Build the MKV at `outputs.mkvOutput`. `langs` is the list of
    /// language codes (e.g. `["de", "en"]`) whose **merged** SRT
    /// files we expect to embed. Per-source SRTs
    /// (`<stamp>.mic.<lang>.srt`, `<stamp>.system.<lang>.srt`) are
    /// deliberately NOT passed to ffmpeg — they're for debugging /
    /// post-hoc inspection only.
    static func export(outputs: Paths.Outputs, langs: [String]) async {
        guard let ffmpeg = locateFFmpeg() else {
            Log.line("MKVExporter: ffmpeg not found (looked in /opt/homebrew/bin, /usr/local/bin, /usr/bin); skipping MKV. Install via `brew install ffmpeg`.")
            return
        }

        let fm = FileManager.default
        let micWAV = outputs.recording(.mic)
        let sysWAV = outputs.recording(.system)
        let haveMic = fm.fileExists(atPath: micWAV.path)
        let haveSys = fm.fileExists(atPath: sysWAV.path)
        guard haveMic || haveSys else {
            Log.line("MKVExporter: no .wav files in work dir; skipping MKV")
            return
        }
        let srtFiles: [(URL, String)] = langs.compactMap { lang in
            let url = outputs.mergedSubtitle(lang)
            return fm.fileExists(atPath: url.path) ? (url, lang) : nil
        }

        // Compute the longest audio duration so we can bound the
        // lavfi video — otherwise `-shortest` would trim the output
        // to the last SRT cue's end time (since subtitle streams
        // count toward "shortest" too).
        let micDur = haveMic ? duration(of: micWAV) : 0
        let sysDur = haveSys ? duration(of: sysWAV) : 0
        let videoDuration = max(micDur, sysDur, 0.5)

        // Optional screen-recording segments. A session may have zero
        // or more — start/stop/change-target each closes a segment
        // and (where applicable) opens the next. We probe each .mov
        // for its actual duration so the composer knows how long to
        // overlay it for. Invalid (zero-duration / probe failed)
        // segments are skipped.
        var segments: [(url: URL, offset: Double, duration: Double)] = []
        for seg in outputs.screenSegments() {
            let off = readScreenOffset(seg.offset)
            let dur = await Self.movDuration(of: seg.mov)
            if dur > 0.05 {
                segments.append((seg.mov, off, dur))
            } else {
                Log.line("MKVExporter: skipping segment \(seg.mov.lastPathComponent) (duration=\(dur))")
            }
        }

        let args = buildArgs(
            segments: segments,
            micWAV: haveMic ? micWAV : nil,
            sysWAV: haveSys ? sysWAV : nil,
            srts: srtFiles,
            output: outputs.mkvOutput,
            videoDuration: videoDuration
        )
        Log.line("MKVExporter: running ffmpeg (langs=\(srtFiles.map(\.1)))")
        do {
            try await runProcess(executable: ffmpeg, args: args)
            Log.line("MKVExporter: wrote \(outputs.mkvOutput.lastPathComponent)")
        } catch {
            Log.line("MKVExporter: ffmpeg failed: \(error.localizedDescription)")
        }
    }

    // MARK: - ffmpeg argv

    private static func buildArgs(
        segments: [(url: URL, offset: Double, duration: Double)],
        micWAV: URL?, sysWAV: URL?, srts: [(URL, String)], output: URL,
        videoDuration: Double
    ) -> [String] {
        // Input ordering: segment .mov files first (indices 0..S-1),
        // then optional mic+system WAVs, then SRT files. The video
        // composer runs in `-filter_complex` and emits `[vout]`; the
        // audio mixer emits `[aout]`. Subtitle streams are mapped
        // directly from their input indices.
        var args: [String] = ["-y", "-loglevel", "warning"]

        // Inputs in fixed order.
        for seg in segments {
            args.append(contentsOf: ["-i", seg.url.path])
        }
        var audioInputs: [Int] = []
        var nextIdx = segments.count
        if let m = micWAV {
            args.append(contentsOf: ["-i", m.path])
            audioInputs.append(nextIdx)
            nextIdx += 1
        }
        if let s = sysWAV {
            args.append(contentsOf: ["-i", s.path])
            audioInputs.append(nextIdx)
            nextIdx += 1
        }
        let firstSRTIdx = nextIdx
        for (path, _) in srts {
            args.append(contentsOf: ["-i", path.path])
            nextIdx += 1
        }

        // Build the single filter_complex graph: video composer
        // (base canvas + per-segment scale-pad + overlays) and audio
        // mixer.
        var filterParts: [String] = []
        let videoOutLabel: String

        if segments.isEmpty {
            // No segments → flat black canvas for the whole audio
            // length. Matches today's behavior 1:1.
            filterParts.append(
                "color=c=black:s=1280x720:r=10:d=\(String(format: "%.3f", videoDuration))[vout]"
            )
            videoOutLabel = "[vout]"
        } else {
            // Base layer: black 1280×720 at 10 fps for the full
            // audio duration. Per-segment streams are letterboxed
            // (scale to fit + pad with black), PTS-shifted to their
            // offset on the run timeline, then overlaid in order.
            // `overlay=eof_action=pass` leaves the previous frame
            // (or base black, between segments) when the overlay
            // input has no current frame — that's how a 5 s clip
            // starting at t=20 ends up showing black for [0,20)
            // and [25,end) without trimming the output.
            // `format=yuv420p` on the base ensures every overlay
            // stage agrees on pixel format (libx264 wants yuv420p
            // anyway).
            filterParts.append(
                "color=c=black:s=1280x720:r=10:d=\(String(format: "%.3f", videoDuration)),format=yuv420p[base]"
            )
            for (i, seg) in segments.enumerated() {
                // Letterbox: keep aspect ratio, pad to 1280×720,
                // center the content. Then shift PTS so the first
                // frame of this segment lands at `offset` seconds
                // on the composed timeline.
                let offsetMs = String(format: "%.3f", max(0, seg.offset))
                filterParts.append(
                    "[\(i):v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,setpts=PTS-STARTPTS+\(offsetMs)/TB,format=yuv420p[s\(i)]"
                )
            }
            // Chain overlays: [base][s0]overlay→[v0]; [v0][s1]→[v1]; …
            var prev = "[base]"
            for i in 0..<segments.count {
                let out = (i == segments.count - 1) ? "[vout]" : "[v\(i)]"
                filterParts.append("\(prev)[s\(i)]overlay=eof_action=pass:shortest=0\(out)")
                prev = out
            }
            videoOutLabel = "[vout]"
        }

        // Audio chain into [aout].
        if audioInputs.count == 2 {
            filterParts.append(
                "[\(audioInputs[0]):a][\(audioInputs[1]):a]amix=inputs=2:normalize=0[aout]"
            )
        } else if audioInputs.count == 1 {
            filterParts.append("[\(audioInputs[0]):a]anull[aout]")
        }

        args.append(contentsOf: ["-filter_complex", filterParts.joined(separator: ";")])

        // Maps. Video always comes from the filter graph now (even
        // the no-segments path goes through filter so the rest of
        // the command stays uniform).
        args.append(contentsOf: ["-map", videoOutLabel])
        if !audioInputs.isEmpty {
            args.append(contentsOf: ["-map", "[aout]"])
        }
        for i in 0..<srts.count {
            args.append(contentsOf: ["-map", "\(firstSRTIdx + i)"])
        }

        // Always re-encode video: filter_complex output isn't
        // copyable. ultrafast keeps export quick; the composite is
        // already low-fps low-bitrate so quality loss is moot.
        args.append(contentsOf: [
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-r", "10",
            "-c:a", "aac",
            "-c:s", "srt",
        ])
        for (i, (_, lang)) in srts.enumerated() {
            args.append(contentsOf: ["-metadata:s:s:\(i)", "language=\(iso639_3(lang))"])
        }
        // Explicitly mark video + audio + first subtitle as default.
        // Without this, the MKV mux leaves all dispositions cleared
        // (probably because lavfi-color and amix outputs don't get
        // any default tag), and VLC interprets a video stream with
        // `default=0` as "auto-deselect" — showing the file as
        // audio-only.
        args.append(contentsOf: ["-disposition:v:0", "default"])
        args.append(contentsOf: ["-disposition:a:0", "default"])
        if !srts.isEmpty {
            // `default+forced` — `default` picks the track on open,
            // `forced` makes VLC actually render its cues without
            // the user manually enabling subtitles via the menu.
            args.append(contentsOf: ["-disposition:s:0", "default+forced"])
        }
        // No `-shortest` — the lavfi `-t` above caps the video, and
        // the audio runs as long as its WAV. Subtitle streams (whose
        // duration is "last cue end") would otherwise force an early
        // cut-off when speech ends before the recording does.
        args.append(output.path)
        return args
    }

    /// Parse the `<stamp>.screen.offset` sidecar — one decimal number
    /// of seconds, written by `ScreenVideoRecorder` on its first
    /// frame. Missing / malformed → 0 (treat as "first frame coincided
    /// with run start"; safer than guessing).
    private static func readScreenOffset(_ url: URL) -> Double {
        guard let data = try? Data(contentsOf: url),
              let text = String(data: data, encoding: .utf8) else { return 0 }
        return Double(text.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0
    }

    /// AVFoundation-based WAV duration probe. `length / sampleRate` is
    /// exact for PCM and avoids the ffprobe round-trip.
    private static func duration(of url: URL) -> Double {
        guard let file = try? AVAudioFile(forReading: url) else { return 0 }
        let sr = file.processingFormat.sampleRate
        return sr > 0 ? Double(file.length) / sr : 0
    }

    /// AVURLAsset-based video duration probe. Returns 0 on failure
    /// so a corrupt/zero-length segment .mov is simply skipped by
    /// the composer rather than blowing up the whole export.
    private static func movDuration(of url: URL) async -> Double {
        let asset = AVURLAsset(url: url)
        do {
            let dur = try await asset.load(.duration)
            let seconds = CMTimeGetSeconds(dur)
            return seconds.isFinite ? seconds : 0
        } catch {
            return 0
        }
    }

    private static func locateFFmpeg() -> String? {
        let candidates = ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"]
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0) }
    }

    /// ISO 639-1 → 639-3 for the languages the app surfaces. Unknown
    /// codes pass through so the MKV stays valid.
    private static func iso639_3(_ code: String) -> String {
        switch code {
        case "en": return "eng"; case "de": return "deu"
        case "fr": return "fra"; case "es": return "spa"
        case "it": return "ita"; case "pt": return "por"
        case "nl": return "nld"; case "da": return "dan"
        case "sv": return "swe"; case "no": return "nor"
        case "fi": return "fin"; case "pl": return "pol"
        case "cs": return "ces"; case "uk": return "ukr"
        case "ru": return "rus"; case "tr": return "tur"
        case "el": return "ell"; case "he": return "heb"
        case "ar": return "ara"; case "hi": return "hin"
        case "th": return "tha"; case "vi": return "vie"
        case "ja": return "jpn"; case "ko": return "kor"
        case "zh": return "zho"
        default: return code
        }
    }

    /// Run ffmpeg asynchronously; throws on non-zero exit.
    private static func runProcess(executable: String, args: [String]) async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: executable)
            proc.arguments = args
            proc.standardOutput = FileHandle(forWritingAtPath: "/dev/null")
            proc.standardError = FileHandle(forWritingAtPath: "/dev/null")
            proc.terminationHandler = { p in
                if p.terminationStatus == 0 {
                    cont.resume()
                } else {
                    cont.resume(throwing: NSError(
                        domain: "MKVExporter", code: Int(p.terminationStatus),
                        userInfo: [NSLocalizedDescriptionKey: "ffmpeg exited with status \(p.terminationStatus)"]
                    ))
                }
            }
            do { try proc.run() }
            catch { cont.resume(throwing: error) }
        }
    }
}

/// Pack a specific set of files into a flat zip (no directory
/// structure) and delete the entire work directory on success. Uses
/// macOS-bundled `/usr/bin/zip` — no external dependency. Caller is
/// responsible for ensuring the parent directory of `destination`
/// exists.
enum ZipArchiver {

    /// Zip exactly `files` into `destination`, flat (no paths), then
    /// delete `workDir`. The work dir holds all the intermediate
    /// per-source artifacts that aren't worth shipping; cleanup is
    /// gated on the zip succeeding so we don't lose data on a
    /// failed pack.
    static func zipFilesAndCleanup(_ files: [URL], into destination: URL, workDir: URL) async {
        // Skip files that aren't actually present (e.g. ffmpeg
        // missing → no MKV). Zip can't include a missing file and
        // we'd rather end up with a partial zip than a hard failure.
        let existing = files.filter { FileManager.default.fileExists(atPath: $0.path) }
        guard !existing.isEmpty else {
            Log.line("ZipArchiver: no shippable files; leaving work dir for inspection: \(workDir.path)")
            return
        }
        do {
            try await runZip(files: existing, into: destination)
            do {
                try FileManager.default.removeItem(at: workDir)
                Log.line("ZipArchiver: wrote \(destination.path), cleaned \(workDir.path)")
            } catch {
                Log.line("ZipArchiver: zip ok but cleanup failed: \(error.localizedDescription)")
            }
        } catch {
            Log.line("ZipArchiver: zip failed (work dir kept for inspection): \(error.localizedDescription)")
        }
    }

    private static func runZip(files: [URL], into destination: URL) async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/zip")
            // `-j` junks paths so files land flat at the zip root.
            // `-q` quiet, `-X` strip extra file attrs.
            var args: [String] = ["-j", "-q", "-X", destination.path]
            args.append(contentsOf: files.map(\.path))
            proc.arguments = args
            proc.terminationHandler = { p in
                if p.terminationStatus == 0 {
                    cont.resume()
                } else {
                    cont.resume(throwing: NSError(
                        domain: "ZipArchiver", code: Int(p.terminationStatus),
                        userInfo: [NSLocalizedDescriptionKey: "zip exited with status \(p.terminationStatus)"]
                    ))
                }
            }
            do { try proc.run() }
            catch { cont.resume(throwing: error) }
        }
    }
}
