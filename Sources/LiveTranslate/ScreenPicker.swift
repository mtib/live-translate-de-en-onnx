import SwiftUI
import ScreenCaptureKit

/// In-app picker for an `SCContentFilter`. Built on
/// `SCShareableContent` instead of `SCContentSharingPicker` —
/// the system-wide picker has the side effect of stopping any
/// SCStream not registered with it, which kills our
/// `SystemAudioSource` mid-session ("Stream was stopped by the
/// system", SCK -3808). A self-rolled list of displays + windows
/// avoids that entirely.
///
/// Skips windows owned by our own app so the floating overlay
/// can't be picked (would create a mirror feedback loop).
struct ScreenTargetChooser: View {

    let onPicked: (SCContentFilter?) -> Void

    @State private var content: SCShareableContent?
    @State private var loadError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Pick what to record").font(.headline)
                Spacer()
                Button("Cancel") { onPicked(nil) }
                    .controlSize(.small)
            }
            Divider()
            content_body
        }
        .padding(12)
        .frame(width: 340, height: 420)
        .task { await load() }
    }

    @ViewBuilder
    private var content_body: some View {
        if let content {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    displaySection(content.displays)
                    windowSection(content.windows)
                }
                .padding(.trailing, 4)
            }
        } else if let loadError {
            Text(loadError)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        } else {
            HStack { Spacer(); ProgressView(); Spacer() }
                .frame(maxHeight: .infinity)
        }
    }

    private func load() async {
        do {
            content = try await SCShareableContent.excludingDesktopWindows(
                false, onScreenWindowsOnly: true
            )
        } catch {
            loadError = "Couldn't load shareable content: \(error.localizedDescription)"
        }
    }

    // MARK: - Displays

    @ViewBuilder
    private func displaySection(_ displays: [SCDisplay]) -> some View {
        if !displays.isEmpty {
            Text("Displays")
                .font(.caption.bold())
                .foregroundStyle(.secondary)
            ForEach(displays, id: \.displayID) { display in
                Button {
                    let filter = SCContentFilter(
                        display: display,
                        excludingApplications: ownAppExclusion(),
                        exceptingWindows: []
                    )
                    onPicked(filter)
                } label: {
                    row(
                        icon: "display",
                        title: "Display \(display.displayID)",
                        subtitle: "\(Int(display.width))×\(Int(display.height))"
                    )
                }
                .buttonStyle(.plain)
            }
        }
    }

    // MARK: - Windows

    @ViewBuilder
    private func windowSection(_ windows: [SCWindow]) -> some View {
        let ownBundleID = Bundle.main.bundleIdentifier
        let pickable = windows.filter { w in
            // Drop windows from our own process (floating overlay
            // would otherwise be a feedback loop), windows with no
            // title (system chrome, menu shadows), and zero-sized.
            guard let app = w.owningApplication else { return false }
            if app.bundleIdentifier == ownBundleID { return false }
            guard let title = w.title, !title.isEmpty else { return false }
            return w.frame.width > 32 && w.frame.height > 32
        }
        let grouped = Dictionary(grouping: pickable) {
            $0.owningApplication?.applicationName ?? "Other"
        }
        let appNames = grouped.keys.sorted { $0.localizedCaseInsensitiveCompare($1) == .orderedAscending }

        if !appNames.isEmpty {
            Text("Windows")
                .font(.caption.bold())
                .foregroundStyle(.secondary)
                .padding(.top, 4)
            ForEach(appNames, id: \.self) { appName in
                Text(appName)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .padding(.top, 2)
                ForEach(grouped[appName] ?? [], id: \.windowID) { window in
                    Button {
                        let filter = SCContentFilter(desktopIndependentWindow: window)
                        onPicked(filter)
                    } label: {
                        row(
                            icon: "macwindow",
                            title: window.title ?? "(untitled)",
                            subtitle: "\(Int(window.frame.width))×\(Int(window.frame.height))"
                        )
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private func ownAppExclusion() -> [SCRunningApplication] {
        guard let content,
              let bid = Bundle.main.bundleIdentifier else { return [] }
        return content.applications.filter { $0.bundleIdentifier == bid }
    }

    private func row(icon: String, title: String, subtitle: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .foregroundStyle(.secondary)
                .frame(width: 16)
            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(subtitle)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(.vertical, 3)
        .padding(.horizontal, 4)
        .contentShape(Rectangle())
        .background(
            RoundedRectangle(cornerRadius: 4)
                .fill(.secondary.opacity(0.001))  // hit-testable, visually transparent
        )
    }
}
