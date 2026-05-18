import SwiftUI
import ScreenCaptureKit

/// Shared chip for arming / starting / stopping / changing the
/// screen-recording segment. Used by the main `TranscriptView` bars
/// and the `MenuBarExtra` popover so the UI is consistent.
///
/// State semantics (see `Pipeline`):
/// * `screenFilter == nil` → no target selected. MKV uses black.
/// * `screenFilter != nil`, not running → armed; will roll on Start.
/// * `screenFilter != nil`, running, not recording → can start a
///   segment mid-session.
/// * `isScreenRecording == true` → actively writing a segment;
///   change target or stop available.
struct ScreenPickButton: View {
    @ObservedObject var pipeline: Pipeline

    @State private var shown = false
    /// What the popover is showing right now. The picker is rendered
    /// inline (instead of via SCContentSharingPicker) — see the file
    /// header on ScreenTargetChooser for why.
    @State private var mode: Mode = .controls

    private enum Mode {
        /// Status + buttons (Pick / Start / Stop / Change).
        case controls
        /// Source list — sets `pendingAction` once a filter is picked.
        case choosing(pendingAction: PickAction)
    }

    private enum PickAction {
        /// Just arm the filter (idle, pre-session).
        case arm
        /// Start a fresh segment with the picked filter.
        case start
        /// Swap segments — stop current, start new.
        case change
    }

    var body: some View {
        Button {
            shown.toggle()
            if !shown { mode = .controls }   // reset on next open
        } label: {
            Image(systemName: iconName)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(tint)
        }
        .buttonStyle(.plain)
        .help(helpText)
        .popover(isPresented: $shown, arrowEdge: .bottom) {
            switch mode {
            case .controls:
                popoverBody.padding(14).frame(width: 280)
            case .choosing(let action):
                ScreenTargetChooser { filter in
                    handlePicked(filter, action: action)
                }
            }
        }
    }

    private func handlePicked(_ filter: SCContentFilter?, action: PickAction) {
        defer { shown = false; mode = .controls }
        guard let filter else { return }
        switch action {
        case .arm:    pipeline.setScreenFilter(filter)
        case .start:  pipeline.setScreenFilter(filter); pipeline.startScreenRecording()
        case .change: pipeline.changeScreenRecording(to: filter)
        }
    }

    private var iconName: String {
        if pipeline.isScreenRecording { return "rectangle.dashed.badge.record" }
        if pipeline.screenFilter != nil { return "rectangle.dashed.badge.record" }
        return "rectangle.dashed"
    }

    private var tint: AnyShapeStyle {
        if pipeline.isScreenRecording { return AnyShapeStyle(.red) }
        if pipeline.screenFilter != nil { return AnyShapeStyle(.orange) }
        return AnyShapeStyle(.secondary)
    }

    private var helpText: String {
        if pipeline.isScreenRecording { return "Recording — click to stop or change target" }
        if pipeline.screenFilter != nil { return "Recording armed — will roll on Start" }
        return "Record screen into the .mkv (optional)"
    }

    @ViewBuilder
    private var popoverBody: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Screen recording").font(.headline)
            Text("Captured at 10 fps / 720p / ~500 kbps. Composed onto a 720p canvas — multiple segments line up on the audio timeline, letterboxed (centered) if the source aspect differs.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            statusLine

            HStack(spacing: 8) {
                if pipeline.isRunning {
                    runningControls
                } else {
                    idleControls
                }
            }
        }
    }

    @ViewBuilder
    private var statusLine: some View {
        if pipeline.isScreenRecording {
            Label("Recording a segment now.", systemImage: "record.circle.fill")
                .font(.caption).foregroundStyle(.red)
        } else if pipeline.screenFilter != nil {
            Label(pipeline.isRunning
                  ? "Target armed — not currently recording."
                  : "Target armed — will roll on Start.",
                  systemImage: "checkmark.circle")
                .font(.caption).foregroundStyle(.orange)
        } else {
            Label("No target — MKV uses a black background.", systemImage: "circle")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var idleControls: some View {
        Button(pipeline.screenFilter == nil ? "Pick…" : "Change…") {
            mode = .choosing(pendingAction: .arm)
        }
        if pipeline.screenFilter != nil {
            Button("Clear", role: .destructive) {
                pipeline.setScreenFilter(nil)
            }
        }
    }

    /// Picking a new target while recording swaps segments; picking
    /// while idle mid-session starts a fresh segment.
    @ViewBuilder
    private var runningControls: some View {
        Button(pipeline.isScreenRecording ? "Change target…" : "Pick & start…") {
            mode = .choosing(pendingAction: pipeline.isScreenRecording ? .change : .start)
        }
        if pipeline.isScreenRecording {
            Button("Stop recording", role: .destructive) {
                pipeline.stopScreenRecording()
            }
        } else if pipeline.screenFilter != nil {
            Button("Start") { pipeline.startScreenRecording() }
        }
    }
}
