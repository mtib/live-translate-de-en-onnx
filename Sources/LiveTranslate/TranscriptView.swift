import SwiftUI
import Translation
import AppKit
import CoreImage
import CoreImage.CIFilterBuiltins

/// Main UI surface. Renders ONLY the rolling sentence list — Start/Stop,
/// screen picker, stream share, and the AI toggle live in the menu-bar
/// popover (`MenuBarView`) and the Settings window (`OptionsView`).
///
/// The list shows completed `Sentence` rows followed by any `InflightChunk`
/// rows currently in flight (listening / partial / translating). When a
/// chunk graduates, its inflight row is replaced by the matching sentence
/// with the same UUID, so SwiftUI animates the transition in place.
///
/// Layout (mixed / side-by-side / compact) is read from `AppSettings`.
struct TranscriptView: View {
    @ObservedObject var pipeline: Pipeline
    @EnvironmentObject var settings: AppSettings

    private var translationConfig: TranslationSession.Configuration {
        TranslationSession.Configuration(
            source: Locale.Language(identifier: String(pipeline.source.identifier.prefix(2))),
            target: Locale.Language(identifier: pipeline.target.code)
        )
    }

    var body: some View {
        ZStack {
            // Translucent flat color (no blur). Theme-aware via
            // `NSColor.textBackgroundColor` (white in light, near-black
            // in dark) — more contrast against the primary text than
            // `windowBackgroundColor` would give. 0.7 opacity keeps
            // the overlay see-through over content behind it.
            overlayBackgroundColor
                .opacity(settings.windowOpacity)
                .ignoresSafeArea()
            content
        }
        // Extend into the (now-hidden) title-bar area so we don't leave
        // a dead band above our controls.
        .ignoresSafeArea()
        // Park the translation session for the lifetime of this config.
        // SwiftUI cancels the closure on config change or view disappear;
        // `defer` then clears the session before the next one is installed.
        // We park by iterating an AsyncStream that's never written to —
        // cancellation wakes the iterator. `Task.sleep` with anything
        // close to Duration's range trips a precondition on macOS 15.
        .translationTask(translationConfig) { session in
            pipeline.installTranslationSession(session)
            defer { pipeline.installTranslationSession(nil) }
            do {
                try await session.prepareTranslation()
                Log.line("Translation prepared")
            } catch {
                Log.line("prepareTranslation failed: \(error.localizedDescription)")
            }
            let (parked, holder) = AsyncStream<Never>.makeStream()
            defer { holder.finish() }
            for await _ in parked { }
        }
        .translationTask(
            TranslationSession.Configuration(
                source: Locale.Language(identifier: "en"),
                target: Locale.Language(identifier: "de")
            )
        ) { session in
            pipeline.installOBSTranslationSession(session)
            defer { pipeline.installOBSTranslationSession(nil) }
            do {
                try await session.prepareTranslation()
                Log.line("OBS en→de translation prepared")
            } catch {
                Log.line("OBS prepareTranslation failed: \(error.localizedDescription)")
            }
            let (parked, holder) = AsyncStream<Never>.makeStream()
            defer { holder.finish() }
            for await _ in parked { }
        }
    }

    private var content: some View {
        sentenceList()
            .padding(14)
    }

    // MARK: - Sentence list

    /// Auto-scrolling list. Sentences and inflight chunks share one
    /// `ForEach` keyed on UUID. When a chunk graduates, its UUID is
    /// inherited by the new `Sentence` (see `Pipeline.graduate`), so
    /// SwiftUI sees an in-place content update on the same row — no
    /// remove+insert, no flicker. `.transition(.opacity)` still fires
    /// for real adds (new chunk) and real removes (pruned sentence).
    private func sentenceList() -> some View {
        let isCompactLayout = settings.layoutMode == .compact
        return ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: isCompactLayout ? 6 : 8) {
                    ForEach(displayRows) { row in
                        Group {
                            switch settings.layoutMode {
                            case .sideBySide: SideBySideRow(row: row)
                            case .compact:    TranscriptRow(row: row, compact: true)
                            case .mixed:      TranscriptRow(row: row, compact: false)
                            }
                        }
                        .id(row.id)
                        .transition(.opacity)
                    }
                    Color.clear.frame(height: 1).id("BOTTOM")
                }
                .padding(.vertical, 2)
                // Row add/remove (a new chunk arrives, a sentence is
                // pruned) gets the .transition(.opacity) treatment via
                // ID changes.
                .animation(.easeInOut(duration: 0.09), value: displayRows.map(\.id))
                // Any visible content change on an existing row —
                // partial-text growth, partial translation refining,
                // graduation — runs through this animation context.
                // Combined with `.contentTransition(.opacity)` on the
                // Text views inside `TranscriptRow`, each change
                // cross-fades smoothly.
                .animation(.easeInOut(duration: 0.09), value: displayRows.map(\.bodyKey))
            }
            .frame(minHeight: isCompactLayout ? 80 : 140)
            .scrollIndicators(.hidden)
            .onChange(of: displayRows.last?.id) { _, _ in
                withAnimation(.easeOut(duration: 0.12)) {
                    proxy.scrollTo("BOTTOM", anchor: .bottom)
                }
            }
            // Partial text grows the last inflight row — scroll to keep
            // the bottom visible as height changes.
            .onChange(of: pipeline.inflightChunks) { _, _ in
                proxy.scrollTo("BOTTOM", anchor: .bottom)
            }
        }
    }

    /// Sentences (top) followed by in-flight chunks (bottom), in one
    /// list so a graduating row keeps its identity across the swap.
    private var displayRows: [DisplayRow] {
        pipeline.sentences.map(DisplayRow.sentence)
            + pipeline.inflightChunks.map(DisplayRow.inflight)
    }
}

/// One entry in the unified transcript list. Both variants carry the
/// same UUID across graduation, so SwiftUI's diffing sees an in-place
/// content update rather than a remove + insert.
enum DisplayRow: Identifiable, Equatable {
    case sentence(Sentence)
    case inflight(InflightChunk)

    var id: UUID {
        switch self {
        case .sentence(let s):  return s.id
        case .inflight(let c):  return c.id
        }
    }

    /// A key derived from everything `TranscriptRow` actually renders,
    /// so any change — kind transitions, partial-text growth, partial
    /// translation refinement, graduation — fires the surrounding
    /// `.animation(_, value:)` context. Combined with
    /// `.contentTransition(.opacity)` on the Text views, every change
    /// cross-fades smoothly. Includes a leading discriminator so two
    /// states that happen to stringify to the same content (e.g. an
    /// inflight `.partial("foo", nil)` and a `.translating("foo")`)
    /// still register as distinct values.
    var bodyKey: String {
        switch self {
        case .sentence(let s):
            return "S\u{1F}\(s.translation)\u{1F}\(s.text)"
        case .inflight(let c):
            switch c.state {
            case .listening:                              return "L"
            case .partial(let t, let trans):              return "P\u{1F}\(t)\u{1F}\(trans ?? "")"
            case .translating(let t):                     return "T\u{1F}\(t)"
            }
        }
    }
}

/// One completed-sentence row. Source icon (mic/speaker) on the left
/// keeps the layout aligned with in-flight rows; translation is the
/// One row in the transcript list. Renders either a graduated
/// `Sentence` or an in-flight chunk; using a single view type means
/// SwiftUI keeps the underlying view instance when a chunk's UUID
/// transitions from `.inflight(...)` to `.sentence(...)` — content
/// updates in place, no fade-out/fade-in flicker.
///
/// Visual states:
///   `.listening`                    → italic "listening" placeholder
///   `.partial(text, nil)`           → italic raw ASR text (no translation yet)
///   `.partial(text, translation)`   → primary translation + secondary caption
///   `.translating(text)`            → italic "translating" + caption
///   `.sentence`                     → identical layout to `.partial(text, translation)`
///                                      so the graduation swap is invisible.
struct TranscriptRow: View {
    let row: DisplayRow
    let compact: Bool
    var fontSizeCap: Double? = nil
    @EnvironmentObject var settings: AppSettings

    private func cap(_ v: Double) -> Double {
        fontSizeCap.map { min(v, $0) } ?? v
    }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            if settings.showSource {
                Image(systemName: source.iconSystemName)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
                    .frame(width: 14, alignment: .center)
            }
            VStack(alignment: .leading, spacing: 1) {
                Text(primaryText)
                    .font(.system(size: cap(isPlaceholder ? settings.transcriptFontSize : settings.translationFontSize)))
                    .italic(isPlaceholder)
                    .foregroundStyle(isPlaceholder
                        ? AnyShapeStyle(settings.transcriptColor.opacity(0.6))
                        : AnyShapeStyle(settings.translationColor))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .contentTransition(.opacity)
                if !compact, let cap = captionText {
                    Text(cap)
                        .font(.system(size: self.cap(settings.transcriptFontSize)))
                        .foregroundStyle(settings.transcriptColor)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .contentTransition(.opacity)
                }
            }
        }
    }

    private var source: SourceTag {
        switch row {
        case .sentence(let s):  return s.source
        case .inflight(let c):  return c.source
        }
    }

    /// True when `primaryText` is a placeholder word ("listening",
    /// "translating", or a raw partial hypothesis with no translation
    /// yet) — render italic + secondary so it reads as transient.
    private var isPlaceholder: Bool {
        switch row {
        case .sentence:                             return false
        case .inflight(let c):
            switch c.state {
            case .partial(_, let translation):      return translation == nil
            default:                                return true
            }
        }
    }

    private var primaryText: String {
        switch row {
        case .sentence(let s):
            return s.translation.isEmpty ? s.text : s.translation
        case .inflight(let c):
            switch c.state {
            case .listening:                            return "listening"
            case .partial(let text, let translation):  return translation ?? text
            case .translating:                          return "translating"
            }
        }
    }

    private var captionText: String? {
        switch row {
        case .sentence(let s):
            return s.translation.isEmpty ? nil : s.text
        case .inflight(let c):
            switch c.state {
            case .partial(let text, let translation):   return translation != nil ? text : nil
            case .translating(let text):                return text
            default:                                    return nil
            }
        }
    }
}

/// Side-by-side layout for all rows: transcript on the left, translation on the right.
/// Handles both completed sentences and inflight chunks so the layout stays stable
/// through the entire lifecycle — no jump from mixed to side-by-side on finalization.
struct SideBySideRow: View {
    let row: DisplayRow
    @EnvironmentObject var settings: AppSettings

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            if settings.showSource {
                Image(systemName: source.iconSystemName)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
                    .frame(width: 14, alignment: .center)
                    .padding(.top, 4)
            }
            Text(leftText)
                .font(.system(size: settings.transcriptFontSize))
                .foregroundStyle(isPlaceholder ? AnyShapeStyle(settings.transcriptColor.opacity(0.5)) : AnyShapeStyle(settings.transcriptColor))
                .italic(isPlaceholder)
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
                .contentTransition(.opacity)
            Text(rightText)
                .font(.system(size: settings.translationFontSize))
                .foregroundStyle(isPlaceholder ? AnyShapeStyle(settings.translationColor.opacity(0.5)) : AnyShapeStyle(settings.translationColor))
                .italic(isPlaceholder)
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
                .contentTransition(.opacity)
        }
        .padding(.vertical, 4)
    }

    private var isPlaceholder: Bool {
        switch row {
        case .sentence: return false
        case .inflight(let c):
            switch c.state {
            case .partial(_, let t): return t == nil
            default: return true
            }
        }
    }

    private var source: SourceTag {
        switch row {
        case .sentence(let s): return s.source
        case .inflight(let c): return c.source
        }
    }

    private var leftText: String {
        switch row {
        case .sentence(let s): return s.text
        case .inflight(let c):
            switch c.state {
            case .listening:                           return "listening…"
            case .partial(let text, _):               return text
            case .translating(let text):              return text
            }
        }
    }

    private var rightText: String {
        switch row {
        case .sentence(let s): return s.translation.isEmpty ? s.text : s.translation
        case .inflight(let c):
            switch c.state {
            case .listening:                           return "…"
            case .partial(_, let translation):        return translation ?? "…"
            case .translating:                        return "translating…"
            }
        }
    }
}

/// Popover content for the stream share icon. Renders the audio stream URL
/// (with copy button and QR code) and, if available, an OBS Browser Source URL.
struct StreamShareView: View {
    let url: String
    var obsURL: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Live translated audio")
                .font(.headline)
            Text("Open on a phone with headphones to hear translations in near-real time.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            urlRow(url)
            if let img = qrImage(for: url) {
                Image(nsImage: img).interpolation(.none).resizable().scaledToFit()
                    .frame(width: 200, height: 200).frame(maxWidth: .infinity).padding(.top, 2)
            }
            if let obsURL {
                Divider()
                Text("OBS Browser Source")
                    .font(.headline)
                Text("Add as Browser Source in OBS. Set dimensions to your overlay size (e.g. 1920×1080).")
                    .font(.caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                urlRow(obsURL)
            }
        }
    }

    @ViewBuilder
    private func urlRow(_ u: String) -> some View {
        HStack(spacing: 6) {
            Text(u).font(.system(.caption, design: .monospaced))
                .textSelection(.enabled).lineLimit(1).truncationMode(.middle)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button {
                let pb = NSPasteboard.general; pb.clearContents()
                pb.setString(u, forType: .string)
            } label: { Image(systemName: "doc.on.doc").font(.system(size: 11)) }
            .buttonStyle(.borderless).help("Copy URL")
        }
    }

    /// Pure CoreImage QR generator. Scales up 8× so the matrix
    /// renders sharp at 200×200 instead of being interpolated from
    /// the module-sized native output.
    private func qrImage(for string: String) -> NSImage? {
        let filter = CIFilter.qrCodeGenerator()
        filter.message = Data(string.utf8)
        filter.correctionLevel = "M"
        guard let out = filter.outputImage?.transformed(by: CGAffineTransform(scaleX: 8, y: 8)) else {
            return nil
        }
        let ctx = CIContext()
        guard let cg = ctx.createCGImage(out, from: out.extent) else { return nil }
        return NSImage(cgImage: cg, size: NSSize(width: out.extent.width, height: out.extent.height))
    }
}
