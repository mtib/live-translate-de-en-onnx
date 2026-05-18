import SwiftUI

@main
struct LiveTranslateApp: App {
    @StateObject private var pipeline = Pipeline()
    /// Captured once by `WindowAccessor` so the menu-bar Show/Hide action
    /// can order the window in/out without going through NSApp.windows.
    @State private var mainWindow: NSWindow?
    /// Tracked explicitly because NSWindow.isVisible is not observable —
    /// SwiftUI can't re-render the menu label based on it changing.
    /// The close button is hidden so this can only change through our
    /// own Show/Hide action, making manual tracking reliable.
    @State private var isWindowVisible = true

    init() {
        Log.startup()  // truncates the log if it's grown past the cap
        // Recover any sessions whose previous app instance died before
        // finalize completed. Runs in the background; doesn't block
        // the UI or interfere with new sessions.
        Task.detached(priority: .background) {
            await CrashRecovery.recoverPendingSessions()
        }
    }

    var body: some Scene {
        Window("LiveTranslate", id: "main") {
            TranscriptView(pipeline: pipeline)
                .frame(minWidth: 260, minHeight: 80)
                .background(WindowAccessor { window in
                    mainWindow = window
                    configure(window)
                })
                .onAppear { installTerminateHook(pipeline: pipeline) }
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentMinSize)
        .defaultSize(width: 520, height: 480)
        .commands {
            // Native macOS menu-bar entries. The Debug menu is small
            // but exists so screenshots can be captured in a known UI
            // state without recording real audio.
            CommandMenu("Debug") {
                Button("Load fixture sentences") {
                    pipeline.loadDebugFixtures()
                }
                .keyboardShortcut("d", modifiers: [.command, .shift])
                Button("Clear sentences") {
                    pipeline.clear()
                }
                .keyboardShortcut("k", modifiers: [.command, .shift])
            }
        }

        // Status-bar icon. Clicking shows the compact transcript popover.
        // Icon fills while recording so state is visible at a glance.
        MenuBarExtra {
            MenuBarView(
                pipeline: pipeline,
                isWindowVisible: $isWindowVisible,
                mainWindow: mainWindow
            )
        } label: {
            Image(systemName: pipeline.isRunning ? "waveform.circle.fill" : "waveform.circle")
        }
        .menuBarExtraStyle(.window)
    }

    /// Registers (once) for `NSApplication.willTerminateNotification` so
    /// any sentences still visible in the rolling list get archived to
    /// the JSONL file before the process exits. Without this, Cmd+Q
    /// would drop everything that hadn't aged into the prune path yet.
    private func installTerminateHook(pipeline: Pipeline) {
        if Self.terminateHookInstalled { return }
        Self.terminateHookInstalled = true
        NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification,
            object: nil,
            queue: .main
        ) { _ in
            // `queue: .main` guarantees this runs on the main thread, so
            // we can safely assume MainActor isolation. A `Task { @MainActor }`
            // would be async and might not finish before the process exits.
            MainActor.assumeIsolated { pipeline.flushPendingSentences() }
        }
    }
    private static var terminateHookInstalled = false

    /// Tweaks the host NSWindow once SwiftUI hands it to us: translucent,
    /// movable from any point in the window, floats above other apps,
    /// stays across Spaces.
    private func configure(_ window: NSWindow) {
        window.titlebarAppearsTransparent = true
        window.isMovableByWindowBackground = true
        window.backgroundColor = .clear
        window.isOpaque = false
        window.hasShadow = true
        // .statusBar (level 25) is above full-screen app content nominally,
        // but macOS does not re-assert z-order on Space transitions — the
        // window exists in the new Space but renders behind the full-screen
        // app until explicitly ordered front. The activeSpaceDidChange
        // observer below handles that. Level stays at .statusBar rather than
        // something extreme like .screenSaver so system UI still renders
        // above us where expected.
        window.level = .statusBar
        // Extend our content into the title-bar area so the hidden
        // traffic-light strip doesn't leave a dead band of background
        // above the controls. The View ignores the safe area to match.
        window.styleMask.insert(.fullSizeContentView)
        window.titleVisibility = .hidden
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        // Hide all traffic lights — the app is meant to be a small floating
        // overlay and the title-bar chrome eats vertical real estate. Use
        // Cmd+Q (or the app menu) to quit; the window drags from anywhere
        // thanks to isMovableByWindowBackground.
        window.standardWindowButton(.closeButton)?.isHidden = true
        window.standardWindowButton(.miniaturizeButton)?.isHidden = true
        window.standardWindowButton(.zoomButton)?.isHidden = true

        // Re-assert z-order on every Space transition. canJoinAllSpaces
        // puts the window into the new Space automatically, but macOS
        // does not re-raise it above full-screen app content — it just
        // sits there invisible behind Discord/YouTube/etc. Calling
        // orderFrontRegardless() after the transition fixes that.
        // We skip the call when the user has explicitly hidden the overlay
        // (isVisible == false after orderOut).
        NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.activeSpaceDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak window] _ in
            guard window?.isVisible == true else { return }
            window?.orderFrontRegardless()
        }
    }
}

/// Bridge to grab the underlying NSWindow so we can apply non-SwiftUI
/// properties (translucency, floating level, click-through behaviour).
private struct WindowAccessor: NSViewRepresentable {
    let onWindow: (NSWindow) -> Void

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        DispatchQueue.main.async { [weak view] in
            if let w = view?.window { onWindow(w) }
        }
        return view
    }
    func updateNSView(_ nsView: NSView, context: Context) {}
}
