import SwiftUI
import ScreenCaptureKit

/// Compact transcript UI shown inside the MenuBarExtra `.window` popover.
/// Mirrors the compact bar layout from `TranscriptView` and reuses
/// `TranscriptRow` / `DisplayRow` for the sentence list so rendering
/// is identical between the floating overlay and the popover.
struct MenuBarView: View {
    @ObservedObject var pipeline: Pipeline
    @Binding var isWindowVisible: Bool
    let mainWindow: NSWindow?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            compactBar
            if let s = pipeline.transcriptSummary {
                VStack(alignment: .leading, spacing: 2) {
                    Text(s.topic)
                        .font(.caption.bold())
                        .lineLimit(1)
                    Text(s.summary)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(3)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.vertical, 4)
                .transition(.opacity)
            }
            sentenceList
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(width: 340)
        .animation(.easeInOut(duration: 0.25), value: pipeline.transcriptSummary != nil)
    }

    // MARK: - Bar

    private var compactBar: some View {
        HStack(spacing: 6) {
            primaryButton
            Text("\(ModelConfig.sourceLanguage) → \(ModelConfig.targetLanguage)")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
            Spacer(minLength: 4)
            ScreenPickButton(pipeline: pipeline)
            streamShareButton
            overlayToggleButton
        }
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
        if let url = pipeline.liveStreamURL {
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
                StreamShareView(url: url).padding(16).frame(width: 240)
            }
        }
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
                        TranscriptRow(row: row, compact: true)
                            .id(row.id)
                            .transition(.opacity)
                    }
                    Color.clear.frame(height: 1).id("BOTTOM")
                }
                .padding(.vertical, 2)
                .animation(.easeInOut(duration: 0.18), value: displayRows.map(\.id))
                .animation(.easeInOut(duration: 0.18), value: displayRows.map(\.bodyKey))
            }
            // ~6 compact rows: callout font (~16pt) + 6pt spacing = ~22pt/row
            .frame(minHeight: 50, maxHeight: 132)
            .scrollIndicators(.hidden)
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
