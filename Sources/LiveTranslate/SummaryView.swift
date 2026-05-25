import SwiftUI
import AppKit

/// Standalone window showing the LLM-generated topic + 2-sentence summary.
/// Auto-opened/closed by `LiveTranslateApp` based on `pipeline.aiAnalysisEnabled`.
struct SummaryView: View {
    @ObservedObject var pipeline: Pipeline
    @EnvironmentObject var settings: AppSettings

    var body: some View {
        ZStack {
            overlayBackgroundColor
                .opacity(settings.windowOpacity)
                .ignoresSafeArea()
            content
                .padding(16)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .ignoresSafeArea()
        .frame(minWidth: 280, idealWidth: 380, minHeight: 120, idealHeight: 220)
        .background(WindowConfigurer())
        .animation(.easeInOut(duration: 0.25), value: pipeline.transcriptSummary?.summary)
    }

    @ViewBuilder
    private var content: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let s = pipeline.transcriptSummary {
                if !s.topic.isEmpty {
                    Text(s.topic)
                        .font(.system(size: settings.translationFontSize, weight: .semibold))
                        .foregroundStyle(settings.translationColor)
                        .lineLimit(2)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if !s.summary.isEmpty {
                    Text(s.summary)
                        .font(.system(size: settings.translationFontSize))
                        .foregroundStyle(settings.translationColor)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
                if s.topic.isEmpty && s.summary.isEmpty {
                    placeholder
                }
            } else {
                placeholder
            }
            Spacer(minLength: 0)
        }
    }

    private var placeholder: some View {
        Text("Waiting for first summary…")
            .font(.system(size: settings.transcriptFontSize))
            .foregroundStyle(settings.transcriptColor.opacity(0.7))
            .italic()
    }
}

/// Strips the title bar and makes the AI Summary window translucent so it
/// matches the floating-overlay aesthetic of the main window.
private struct WindowConfigurer: NSViewRepresentable {
    func makeNSView(context: Context) -> NSView {
        let v = NSView()
        DispatchQueue.main.async { [weak v] in
            guard let w = v?.window else { return }
            w.titlebarAppearsTransparent = true
            w.isMovableByWindowBackground = true
            w.backgroundColor = .clear
            w.isOpaque = false
            w.hasShadow = true
            w.styleMask.insert(.fullSizeContentView)
            w.titleVisibility = .hidden
            w.level = .statusBar
            w.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
            w.standardWindowButton(.closeButton)?.isHidden = true
            w.standardWindowButton(.miniaturizeButton)?.isHidden = true
            w.standardWindowButton(.zoomButton)?.isHidden = true
        }
        return v
    }
    func updateNSView(_ nsView: NSView, context: Context) {}
}

/// Hidden view embedded in the main window. Observes `aiAnalysisEnabled`
/// and opens / dismisses the "summary" window scene to match.
struct SummaryWindowController: View {
    @ObservedObject var pipeline: Pipeline
    @Environment(\.openWindow) private var openWindow
    @Environment(\.dismissWindow) private var dismissWindow

    var body: some View {
        Color.clear
            .frame(width: 0, height: 0)
            .onAppear { sync(pipeline.aiAnalysisEnabled) }
            .onChange(of: pipeline.aiAnalysisEnabled) { _, on in sync(on) }
    }

    private func sync(_ on: Bool) {
        if on { openWindow(id: "summary") }
        else  { dismissWindow(id: "summary") }
    }
}
