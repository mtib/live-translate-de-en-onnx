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

    enum LayoutMode: String, CaseIterable, Identifiable {
        case mixed      = "mixed"
        case sideBySide = "sideBySide"
        var id: String { rawValue }
        var label: String { self == .mixed ? "Mixed" : "Side by Side" }
    }
}

// MARK: - Color ↔ hex string

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
