import SwiftUI

struct OptionsView: View {
    @ObservedObject var settings: AppSettings
    @ObservedObject var pipeline: Pipeline
    @State private var transcriptColor: Color  = .secondary
    @State private var translationColor: Color = .primary

    var body: some View {
        Form {
            Section("Typography") {
                LabeledContent("Transcript size") {
                    HStack {
                        Slider(value: $settings.transcriptFontSize, in: 10...28, step: 1)
                            .frame(width: 160)
                        Text("\(Int(settings.transcriptFontSize))pt")
                            .frame(width: 32, alignment: .trailing).monospacedDigit()
                    }
                }
                LabeledContent("Translation size") {
                    HStack {
                        Slider(value: $settings.translationFontSize, in: 10...32, step: 1)
                            .frame(width: 160)
                        Text("\(Int(settings.translationFontSize))pt")
                            .frame(width: 32, alignment: .trailing).monospacedDigit()
                    }
                }
            }
            Section("Colors") {
                ColorPicker("Transcript text",  selection: $transcriptColor,  supportsOpacity: false)
                    .onChange(of: transcriptColor)  { _, c in settings.transcriptColor  = c }
                ColorPicker("Translation text", selection: $translationColor, supportsOpacity: false)
                    .onChange(of: translationColor) { _, c in settings.translationColor = c }
            }
            Section("Layout") {
                Picker("Sentence layout", selection: $settings.layoutModeRaw) {
                    ForEach(AppSettings.LayoutMode.allCases) { m in
                        Text(m.label).tag(m.rawValue)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                Group {
                    switch settings.layoutMode {
                    case .mixed:      mixedPreview
                    case .sideBySide: sideBySidePreview
                    case .compact:    compactPreview
                    }
                }
                Toggle("Show source (mic / system icon)", isOn: $settings.showSource)
            }
            if pipeline.aiAnalysisAvailable {
                Section("AI") {
                    Toggle("AI analysis (topic + summary)", isOn: $pipeline.aiAnalysisEnabled)
                }
            }
            Section("Window") {
                LabeledContent("Background opacity") {
                    HStack {
                        Slider(value: $settings.windowOpacity, in: 0.2...1.0, step: 0.05)
                            .frame(width: 160)
                        Text("\(Int(settings.windowOpacity * 100))%")
                            .frame(width: 36, alignment: .trailing).monospacedDigit()
                    }
                }
            }
        }
        .formStyle(.grouped)
        .frame(minWidth: 380, idealWidth: 420, minHeight: 420, idealHeight: 520)
        .padding()
        .onAppear {
            transcriptColor  = settings.transcriptColor
            translationColor = settings.translationColor
        }
    }

    private var mixedPreview: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Translation appears here")
                .font(.system(size: settings.translationFontSize))
                .foregroundStyle(settings.translationColor)
            Text("Transcription sits below as a caption")
                .font(.system(size: settings.transcriptFontSize))
                .foregroundStyle(settings.transcriptColor)
        }
        .padding(.top, 4)
    }

    private var compactPreview: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Translation only — no transcript caption")
                .font(.system(size: settings.translationFontSize))
                .foregroundStyle(settings.translationColor)
        }
        .padding(.top, 4)
    }

    private var sideBySidePreview: some View {
        HStack(alignment: .top, spacing: 12) {
            Text("Transcription on the left")
                .font(.system(size: settings.transcriptFontSize))
                .foregroundStyle(settings.transcriptColor)
                .frame(maxWidth: .infinity, alignment: .leading)
            Divider()
            Text("Translation on the right")
                .font(.system(size: settings.translationFontSize))
                .foregroundStyle(settings.translationColor)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.top, 4)
    }
}
