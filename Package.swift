// swift-tools-version:6.2
import PackageDescription

let package = Package(
    name: "LiveTranslate",
    platforms: [.macOS(.v26)],
    targets: [
        // RNNoise (xiph, BSD 3-clause), pinned to v0.1.1 where the model
        // weights are embedded in the C sources (no runtime download).
        // See Sources/CRNNoise/LICENSE.
        .target(
            name: "CRNNoise",
            path: "Sources/CRNNoise",
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("."),
                // The vendored C uses some warnings-prone older idioms.
                .unsafeFlags([
                    "-Wno-implicit-function-declaration",
                    // rnn.c has intentional null-dereference patterns (upstream
                    // xiph code uses NULL deref as a compile-time assert trick).
                    "-Wno-null-dereference",
                ]),
            ]
        ),
        // Thin bridge target around the sherpa-onnx shared dylib.
        // The dylib lives under external/sherpa-onnx/lib/ and is
        // downloaded by tools/download-sherpa.sh. build.sh copies it
        // into the app bundle's Frameworks/ directory and sets RPATH.
        .target(
            name: "CSherpaOnnx",
            path: "Sources/CSherpaOnnx",
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("include"),
            ],
            linkerSettings: [
                .unsafeFlags([
                    "-L./external/sherpa-onnx/lib",
                    "-lsherpa-onnx-c-api",
                    // onnxruntime is loaded by the sherpa dylib at runtime
                    // from Frameworks/; we only need to link against it so
                    // the dynamic loader can locate it.
                    "-lonnxruntime",
                ]),
            ]
        ),
        .executableTarget(
            name: "LiveTranslate",
            dependencies: ["CRNNoise", "CSherpaOnnx"],
            path: "Sources/LiveTranslate",
            swiftSettings: [
                // Swift 5 mode keeps the data-flow code (AsyncStream pumping
                // a non-Sendable AVAudioPCMBuffer into SFSpeech) tractable
                // without scattering @unchecked Sendable everywhere.
                .swiftLanguageMode(.v5)
            ]
        )
    ]
)
