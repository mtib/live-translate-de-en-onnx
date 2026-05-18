import Foundation

/// All session artifacts live in a per-run **temp work directory**
/// while a session is in progress (audio buffers, JSONL log, per-source
/// SRTs, merged SRTs maintained live). When the session ends and the
/// MKV is built, the directory is zipped into
///
///     ~/Documents/LiveTranslate/<stamp>.zip
///
/// — one self-contained artifact per session — and the work directory
/// is deleted. Restarting a session opens a fresh work directory with
/// a new timestamp.
///
/// Centralising the layout here means changing it is one edit, and the
/// rest of the app uses `Paths.Outputs` accessors rather than building
/// paths inline.
enum Paths {
    static let runFilenameFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        return f
    }()

    /// `~/Documents/LiveTranslate/` — the only directory the user
    /// sees. Only finished zips land here.
    static func documentsRoot() throws -> URL {
        let fm = FileManager.default
        guard let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first else {
            throw PathError.noDocumentsDirectory
        }
        let url = docs.appendingPathComponent("LiveTranslate", isDirectory: true)
        try fm.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// All output paths for one in-progress session. Flat layout
    /// inside the work directory — only intermediates that ffmpeg
    /// needs (per-source WAVs, per-language merged SRTs) plus the
    /// two shipped artifacts (JSONL + MKV) live there. Everything
    /// outside `shippedFiles` is deleted when the work dir is
    /// removed after the zip is written.
    struct Outputs {
        /// `<stamp>` — e.g. `2026-05-17_15-42-10`.
        let timestamp: String
        /// Temp directory holding all session artifacts. Removed
        /// after the zip is written.
        let workDir: URL
        /// Final per-session zip in `~/Documents/LiveTranslate/`.
        let zipDestination: URL

        /// `<workDir>/<stamp>.jsonl` — one JSON object per emitted
        /// sentence, with `source` field. The shipped text record.
        var transcript: URL {
            workDir.appendingPathComponent("\(timestamp).jsonl")
        }

        /// `<workDir>/<stamp>.<source>.wav` — per-source post-denoise
        /// + AGC audio. Intermediate only; consumed by ffmpeg's
        /// `amix` filter at session end.
        func recording(_ source: SourceTag) -> URL {
            workDir.appendingPathComponent("\(timestamp).\(source.rawValue).wav")
        }

        /// `<workDir>/<stamp>.<lang>.srt` — live-merged SRT for one
        /// language, both sources interleaved (no source prefix).
        /// Intermediate only; embedded in the MKV.
        func mergedSubtitle(_ langCode: String) -> URL {
            workDir.appendingPathComponent("\(timestamp).\(langCode).srt")
        }

        /// `<workDir>/<stamp>.mkv` — final per-session video bundle:
        /// merged audio (amix'd from per-source WAVs) + both
        /// language SRTs embedded over either the screen recording
        /// (if `screenVideo` is present) or a 1280×720 black frame.
        var mkvOutput: URL {
            workDir.appendingPathComponent("\(timestamp).mkv")
        }

        /// `<workDir>/<stamp>.screen.NNN.mov` — one segment of the
        /// optional screen recording (10 fps / 1280×720 / 500 kbps
        /// H.264). A session can produce zero or more segments — the
        /// user can start, stop, and change targets mid-session, each
        /// of which closes the current segment and (where applicable)
        /// opens a new one. Intermediates only; composed into the
        /// MKV's video track via ffmpeg filter graph.
        func screenSegmentMov(_ index: Int) -> URL {
            workDir.appendingPathComponent(String(format: "%@.screen.%03d.mov", timestamp, index))
        }

        /// Sidecar text file holding the wall-clock offset of the
        /// segment's first frame relative to `runStartedAt`, in
        /// seconds. Read by `MKVExporter` to place the segment at
        /// the correct slot on the composed timeline.
        func screenSegmentOffset(_ index: Int) -> URL {
            workDir.appendingPathComponent(String(format: "%@.screen.%03d.offset", timestamp, index))
        }

        /// Enumerate all `.screen.NNN.mov` files in the work dir,
        /// ordered by segment index. Used by `MKVExporter` and
        /// `CrashRecovery` so the composer doesn't need to know how
        /// many segments the session produced.
        func screenSegments() -> [(index: Int, mov: URL, offset: URL)] {
            let fm = FileManager.default
            guard let entries = try? fm.contentsOfDirectory(at: workDir, includingPropertiesForKeys: nil) else { return [] }
            let prefix = "\(timestamp).screen."
            let suffix = ".mov"
            var hits: [(Int, URL, URL)] = []
            for url in entries {
                let name = url.lastPathComponent
                guard name.hasPrefix(prefix), name.hasSuffix(suffix) else { continue }
                let mid = name.dropFirst(prefix.count).dropLast(suffix.count)
                guard let idx = Int(mid) else { continue }
                hits.append((idx, url, screenSegmentOffset(idx)))
            }
            return hits.sorted { $0.0 < $1.0 }
        }

        /// The files that go into the user-facing zip: just the
        /// transcript and the MKV. Everything else in the work dir
        /// is intermediate.
        var shippedFiles: [URL] {
            [transcript, mkvOutput]
        }
    }

    /// Allocate a fresh work directory for a new session.
    static func newRunOutputs(now: Date = Date()) throws -> Outputs {
        let stamp = runFilenameFormatter.string(from: now)
        let workURL = URL(
            fileURLWithPath: NSTemporaryDirectory().appendingPathComponent("livetranslate-\(stamp)", isDirectory: true),
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: workURL, withIntermediateDirectories: true)
        return Outputs(
            timestamp: stamp,
            workDir: workURL,
            zipDestination: try documentsRoot().appendingPathComponent("\(stamp).zip")
        )
    }
}

enum PathError: LocalizedError {
    case noDocumentsDirectory
    var errorDescription: String? {
        switch self {
        case .noDocumentsDirectory: return "No Documents directory available."
        }
    }
}

// Small `String` helper that appendingPathComponent doesn't exist on.
private extension String {
    func appendingPathComponent(_ s: String, isDirectory: Bool = false) -> String {
        let base = hasSuffix("/") ? self : self + "/"
        return base + s + (isDirectory && !s.hasSuffix("/") ? "/" : "")
    }
}
