import SwiftUI

/// Persisted display preferences. Inject as `.environmentObject(settings)` at the
/// app root so `TranscriptView`, `MenuBarView`, and row views all read from the
/// same source without prop-drilling.
final class AppSettings: ObservableObject {

    @AppStorage("settings.transcriptFontSize")   var transcriptFontSize: Double  = 13
    @AppStorage("settings.translationFontSize")  var translationFontSize: Double  = 16
    @AppStorage("settings.windowOpacity")         var windowOpacity: Double        = 0.7
    @AppStorage("settings.layoutModeRaw")         var layoutModeRaw: String        = LayoutMode.mixed.rawValue
    @AppStorage("settings.transcriptColorHex")    var transcriptColorHex: String   = "#8a8a8a"
    @AppStorage("settings.translationColorHex")   var translationColorHex: String  = "#eeeeee"
    @AppStorage("settings.showSource")            var showSource: Bool             = true

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

    /// JSON payload broadcast to web subscribers so they can mirror
    /// the app's colors and font sizes. Updated whenever any visible
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
/// darker-than-system grey (#101010) — pure black would crush the
/// translucency layering, but the default `.textBackgroundColor` (~#1e1e1e)
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
