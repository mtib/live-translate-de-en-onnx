import SwiftUI
import ScreenCaptureKit

/// Compact transcript UI shown inside the MenuBarExtra `.window` popover.
/// Mirrors the compact bar layout from `TranscriptView` and reuses
/// `TranscriptRow` / `DisplayRow` for the sentence list so rendering
/// is identical between the floating overlay and the popover.
struct MenuBarView: View {
    @ObservedObject var pipeline: Pipeline
    @ObservedObject var settings: AppSettings
    @Binding var isWindowVisible: Bool
    let mainWindow: NSWindow?
    @Environment(\.openSettings) private var openSettings

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            compactBar
            if let s = pipeline.transcriptSummary, !s.summary.isEmpty {
                Text(s.summary)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
            sentenceList
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(width: 340)
        .animation(.easeInOut(duration: 0.25), value: pipeline.transcriptSummary?.summary)
    }

    // MARK: - Bar

    private var compactBar: some View {
        HStack(spacing: 6) {
            primaryButton
            if let topic = pipeline.transcriptSummary?.topic {
                Text(topic)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Spacer(minLength: 4)
            ScreenPickButton(pipeline: pipeline)
            streamShareButton
            settingsButton
            overlayToggleButton
        }
        .animation(.easeInOut(duration: 0.2), value: pipeline.transcriptSummary?.topic)
    }

    private var primaryButton: some View {
        let finalizing = pipeline.status.isFinalizing
        return Button {
            pipeline.toggle()
        } label: {
            if finalizing {
                ProgressView().controlSize(.small).frame(width: 14, height: 14)
            } else {
                Image(systemName: pipeline.isRunning ? "stop.fill" : "play.fill")
                    .font(.system(size: 11, weight: .semibold))
                    .frame(width: 14, height: 14)
            }
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .tint(finalizing ? .secondary : (pipeline.isRunning ? .red : .accentColor))
        .disabled(finalizing)
        .keyboardShortcut(.return, modifiers: [])
    }

    @State private var streamShareShown = false
    @ViewBuilder
    private var streamShareButton: some View {
        if pipeline.liveStreamURL != nil || pipeline.liveOBSURL != nil {
            Button { streamShareShown.toggle() } label: {
                Image(systemName: "dot.radiowaves.left.and.right")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(
                        pipeline.ttsActive ? AnyShapeStyle(.green) : AnyShapeStyle(.secondary)
                    )
                    .animation(.easeInOut(duration: 0.25), value: pipeline.ttsActive)
            }
            .buttonStyle(.plain)
            .help(pipeline.ttsActive ? "Live audio stream — listener connected" : "Live translated-audio stream")
            .popover(isPresented: $streamShareShown, arrowEdge: .bottom) {
                StreamShareView(
                    url: pipeline.liveStreamURL ?? pipeline.liveOBSURL ?? "",
                    obsURL: pipeline.liveOBSURL
                )
                .padding(16).frame(width: 240)
            }
        }
    }

    private var settingsButton: some View {
        Button { openSettings() } label: {
            Image(systemName: "gearshape")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .buttonStyle(.plain)
        .help("Open settings (⌘,)")
    }

    private var overlayToggleButton: some View {
        Button {
            if isWindowVisible {
                mainWindow?.orderOut(nil)
            } else {
                mainWindow?.orderFrontRegardless()
            }
            isWindowVisible.toggle()
        } label: {
            Image(systemName: isWindowVisible ? "pip.exit" : "pip.enter")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .buttonStyle(.plain)
        .help(isWindowVisible ? "Hide floating overlay" : "Show floating overlay")
        .keyboardShortcut("l", modifiers: [.command, .shift])
    }

    // MARK: - Sentence list

    private var displayRows: [DisplayRow] {
        pipeline.sentences.map(DisplayRow.sentence)
            + pipeline.inflightChunks.map(DisplayRow.inflight)
    }

    private var sentenceList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 6) {
                    ForEach(displayRows) { row in
                        TranscriptRow(row: row, compact: true, fontSizeCap: 15)
                            .id(row.id)
                            .transition(.opacity)
                    }
                    Color.clear.frame(height: 1).id("BOTTOM")
                }
                .padding(.vertical, 2)
                .animation(.easeInOut(duration: 0.09), value: displayRows.map(\.id))
            }
            // ~6 compact rows: callout font (~16pt) + 6pt spacing = ~22pt/row
            .frame(minHeight: 120, maxHeight: 240)
            .scrollIndicators(.hidden)
            // Summary sits here (not in parent VStack) so its appearance never
            // changes the ScrollView frame — safeAreaInset adjusts content
            // offset instead, keeping the bottom anchor stable.
            .onChange(of: displayRows.last?.id) { _, _ in
                withAnimation(.easeOut(duration: 0.12)) {
                    proxy.scrollTo("BOTTOM", anchor: .bottom)
                }
            }
            .onChange(of: pipeline.inflightChunks) { _, _ in
                proxy.scrollTo("BOTTOM", anchor: .bottom)
            }
        }
    }
}
