import SwiftUI
import Combine

/// Persisted display preferences. Read by `TranscriptView`, `SummaryView`,
/// `MenuBarView`, `OptionsView`, and `Pipeline` (which forwards changes to
/// the live web target).
///
/// Backed by UserDefaults with explicit @Published wrappers — using
/// @AppStorage inside an ObservableObject doesn't fire objectWillChange,
/// so views observing the object would never re-render on writes.
final class AppSettings: ObservableObject {

    @Published var transcriptFontSize:   Double { didSet { persist(transcriptFontSize,  for: Self.kTranscriptFontSize)  } }
    @Published var translationFontSize:  Double { didSet { persist(translationFontSize, for: Self.kTranslationFontSize) } }
    @Published var windowOpacity:        Double { didSet { persist(windowOpacity,       for: Self.kWindowOpacity)       } }
    @Published var layoutModeRaw:        String { didSet { persist(layoutModeRaw,       for: Self.kLayoutModeRaw)       } }
    @Published var transcriptColorHex:   String { didSet { persist(transcriptColorHex,  for: Self.kTranscriptColorHex)  } }
    @Published var translationColorHex:  String { didSet { persist(translationColorHex, for: Self.kTranslationColorHex) } }
    @Published var showSource:           Bool   { didSet { persist(showSource,          for: Self.kShowSource)          } }

    init() {
        let d = UserDefaults.standard
        transcriptFontSize  = d.object(forKey: Self.kTranscriptFontSize)  as? Double ?? 13
        translationFontSize = d.object(forKey: Self.kTranslationFontSize) as? Double ?? 16
        windowOpacity       = d.object(forKey: Self.kWindowOpacity)       as? Double ?? 0.7
        layoutModeRaw       = d.string(forKey: Self.kLayoutModeRaw)             ?? LayoutMode.mixed.rawValue
        transcriptColorHex  = d.string(forKey: Self.kTranscriptColorHex)        ?? "#8a8a8a"
        translationColorHex = d.string(forKey: Self.kTranslationColorHex)       ?? "#eeeeee"
        showSource          = (d.object(forKey: Self.kShowSource) as? Bool) ?? true
    }

    private static let kTranscriptFontSize  = "settings.transcriptFontSize"
    private static let kTranslationFontSize = "settings.translationFontSize"
    private static let kWindowOpacity       = "settings.windowOpacity"
    private static let kLayoutModeRaw       = "settings.layoutModeRaw"
    private static let kTranscriptColorHex  = "settings.transcriptColorHex"
    private static let kTranslationColorHex = "settings.translationColorHex"
    private static let kShowSource          = "settings.showSource"

    private func persist<T>(_ value: T, for key: String) {
        UserDefaults.standard.set(value, forKey: key)
    }

    var layoutMode: LayoutMode {
        get { LayoutMode(rawValue: layoutModeRaw) ?? .mixed }
        set { layoutModeRaw = newValue.rawValue }
    }

    var transcriptColor: Color {
        get { Color(hex: transcriptColorHex) ?? Color(nsColor: .secondaryLabelColor) }
        set { transcriptColorHex = newValue.hexString ?? transcriptColorHex }
    }

    var translationColor: Color {
        get { Color(hex: translationColorHex) ?? Color(nsColor: .labelColor) }
        set { translationColorHex = newValue.hexString ?? translationColorHex }
    }

    /// JSON payload broadcast to web subscribers so they mirror the
    /// app's colors and font sizes. Updated whenever any visible
    /// setting changes.
    func webPayload() -> String {
        let bg = overlayBackgroundHex()
        return """
        {"backgroundHex":"\(bg)","translationHex":"\(translationColorHex)","transcriptHex":"\(transcriptColorHex)","translationFontSize":\(translationFontSize),"transcriptFontSize":\(transcriptFontSize)}
        """
    }

    enum LayoutMode: String, CaseIterable, Identifiable {
        case mixed      = "mixed"
        case sideBySide = "sideBySide"
        case compact    = "compact"
        var id: String { rawValue }
        var label: String {
            switch self {
            case .mixed:      return "Mixed"
            case .sideBySide: return "Side by Side"
            case .compact:    return "Compact"
            }
        }
    }
}

// MARK: - Color ↔ hex string

/// Hex color string of the current overlay background, resolved against the
/// app's current appearance. Used by the web target so it visually matches
/// the macOS window.
func overlayBackgroundHex() -> String {
    let appearance = NSApp?.effectiveAppearance ?? NSAppearance(named: .darkAqua)!
    let isDark = appearance.bestMatch(from: [.darkAqua, .vibrantDark, .accessibilityHighContrastDarkAqua, .accessibilityHighContrastVibrantDark]) != nil
    let ns = isDark ? NSColor(white: 0.06, alpha: 1) : NSColor.textBackgroundColor
    guard let c = ns.usingColorSpace(.sRGB) else { return isDark ? "#0f0f0f" : "#ffffff" }
    return String(format: "#%02x%02x%02x",
                  Int(c.redComponent * 255),
                  Int(c.greenComponent * 255),
                  Int(c.blueComponent * 255))
}

/// Background color used by the floating overlay and AI summary windows.
/// In light mode: the system text-background (white). In dark mode: a
/// darker-than-system grey (~#0F0F0F) — pure black would crush the
/// translucency layering, but the default `.textBackgroundColor` (~#1E1E1E)
/// reads as too light against full-screen video.
let overlayBackgroundColor: Color = Color(nsColor: NSColor(name: nil) { appearance in
    let isDark = appearance.bestMatch(from: [.darkAqua, .vibrantDark, .accessibilityHighContrastDarkAqua, .accessibilityHighContrastVibrantDark]) != nil
    return isDark ? NSColor(white: 0.06, alpha: 1) : .textBackgroundColor
})

extension Color {
    /// Parse `#rrggbb` (6-digit hex, leading `#` required).
    init?(hex: String) {
        guard hex.hasPrefix("#") else { return nil }
        let h = String(hex.dropFirst())
        guard h.count == 6, let rgb = UInt64(h, radix: 16) else { return nil }
        self.init(
            red:   Double((rgb >> 16) & 0xFF) / 255,
            green: Double((rgb >>  8) & 0xFF) / 255,
            blue:  Double( rgb        & 0xFF) / 255
        )
    }

    /// Render as `#rrggbb`. Returns nil if color space conversion fails.
    var hexString: String? {
        guard let c = NSColor(self).usingColorSpace(.sRGB) else { return nil }
        return String(format: "#%02x%02x%02x",
                      Int(c.redComponent * 255),
                      Int(c.greenComponent * 255),
                      Int(c.blueComponent * 255))
    }
}
