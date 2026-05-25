/// Compile-time constants for ASR / VAD / TTS models.
///
/// Language pair is fixed at build time — default: German → English.
/// Model file names are relative to the app bundle's Resources/ directory
/// (copied there by build.sh from build/sherpa-models/).
enum ModelConfig {

    // MARK: — Language pair

    /// BCP-47 source language fed to the ASR decoder.
    static let sourceLanguage = "de"

    /// BCP-47 target language for the Apple Translation API and TTS voice.
    static let targetLanguage = "en"

    // MARK: — Runtime provider

    /// ONNX execution provider.  "coreml" enables the Apple CoreML / ANE
    /// acceleration path; falls back to "cpu" automatically when unavailable.
    static let provider = "coreml"

    // MARK: — ASR  (sherpa-onnx streaming zipformer RNN-T)

    static let asrModelDir = "sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06"

    /// Paths inside the model directory.
    static var asrEncoder: String { "\(asrModelDir)/encoder.onnx" }
    static var asrDecoder: String { "\(asrModelDir)/decoder.onnx" }
    static var asrJoiner:  String { "\(asrModelDir)/joiner.onnx" }
    static var asrTokens:  String { "\(asrModelDir)/tokens.txt" }

    // MARK: — TTS  (kitten-mini-en-v0_8)

    static let ttsModelDir = "kitten-mini-en-v0_8"

    static var ttsModel:   String { "\(ttsModelDir)/model.onnx" }
    static var ttsVoices:  String { "\(ttsModelDir)/voices.bin" }
    static var ttsTokens:  String { "\(ttsModelDir)/tokens.txt" }
    /// espeak-ng-data/ directory (required by kitten TTS at runtime).
    static var ttsDataDir: String { "\(ttsModelDir)/espeak-ng-data" }
}
