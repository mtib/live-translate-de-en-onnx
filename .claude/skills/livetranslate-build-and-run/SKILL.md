---
name: livetranslate-build-and-run
description: Use whenever building, signing, launching, or shelling-out for LiveTranslate — including running the build script, downloading sherpa-onnx assets, TCC permission resets, log tailing, and bundle layout. After any change to build.sh/download-sherpa.sh/Info.plist/permissions/signing, update this skill.
---

# Build, run, debug

## Build & launch (canonical)

```sh
LIVETRANSLATE_SIGN_IDENTITY=LiveTranslateDev ./build.sh
open build/LiveTranslate.app
```

- The env var is required EVERY invocation — non-interactive bash does NOT source `~/.zshrc`. Without it the binary is ad-hoc-signed (fresh `cdhash`) and TCC re-prompts for mic+screen recording on every launch.
- Always launch via `open` (TCC keys on bundle ID, not exec path).

## SwiftPM constraints

- swift-tools-version: 6.0
- `LiveTranslate` target pinned to `.swiftLanguageMode(.v5)` (Translation APIs awkward under Swift 6 strict concurrency).
- macOS deployment: `.macOS(.v26)` (enables FoundationModels).
- No `.xcodeproj`. No CMake.

## tools/download-sherpa.sh (idempotent)

Downloads (skips existing):

| Artifact | Source | Destination |
|---|---|---|
| `libsherpa-onnx-c-api.dylib` + `libonnxruntime.1.24.4.dylib` | `sherpa-onnx-v1.13.2-osx-arm64-shared.tar.bz2` | `external/sherpa-onnx/lib/` |
| `sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06/` | k2-fsa asr-models | `build/sherpa-models/<dir>/` |
| `kitten-mini-en-v0_8/` | k2-fsa tts-models | `build/sherpa-models/<dir>/` |

kitten-mini contains: `model.onnx`, `voices.bin`, `tokens.txt`, `espeak-ng-data/` — **no `lexicon.txt`** (see bug history #25). Silero VAD onnx is still downloaded for external scripts but not bundled — Swift uses energy+ZCR VAD.

## build.sh steps

1. `./tools/download-sherpa.sh`
2. `LIBRARY_PATH=external/sherpa-onnx/lib swift build -c release`
3. Wipe & create `build/LiveTranslate.app/Contents/{MacOS,Resources,Frameworks}`
4. Copy binary → `Contents/MacOS/LiveTranslate`
5. Copy `Info.plist`
6. `install_name_tool -add_rpath @executable_path/../Frameworks`
7. Copy both dylibs to `Frameworks/`; create unversioned `libonnxruntime.dylib` symlink
8. Copy `sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06/` → `Resources/`
9. Copy `kitten-mini-en-v0_8/` → `Resources/`
10. `./tools/make-icon.sh build/icon` + copy `icon.icns` → `Resources/`
11. `codesign --force --deep --sign "${SIGN_IDENTITY:--}"`
12. Print `✓ built build/LiveTranslate.app`

## Common operational commands

```sh
tail -f /tmp/livetranslate.log         # live log
pkill -f LiveTranslate                 # kill all instances

# Reset TCC if grants go stale
tccutil reset Microphone local.mtib.livetranslate
tccutil reset ScreenCapture local.mtib.livetranslate

./tools/download-sherpa.sh             # assets without rebuild
```

## Permissions

- `NSMicrophoneUsageDescription` — `AVCaptureDevice.requestAccess` at run start.
- `NSScreenCaptureUsageDescription` — first `SCStream.startCapture()`.
- No `NSSpeechRecognitionUsageDescription` (sherpa-onnx is fully local).

To persist grants across rebuilds: `LIVETRANSLATE_SIGN_IDENTITY=<self-signed-cert-name>` keys TCC by certificate identity, not binary hash.

## Icon generation

`tools/make-icon.sh` → calls `tools/make-icon.swift` to render master 1024×1024 PNG (SF Symbol `bubble.left.and.text.bubble.right.fill` on blue gradient squircle), then `sips`+`iconutil` for the rest. The OBS overlay reuses an SVG approximation of this glyph as a top-right "connected" indicator (`LiveAudioServer.obsPageHTML` `#indicator`).

## Bundle layout

```
LiveTranslate.app/Contents/
├── Info.plist
├── MacOS/LiveTranslate
├── Frameworks/
│   ├── libsherpa-onnx-c-api.dylib
│   ├── libonnxruntime.1.24.4.dylib
│   └── libonnxruntime.dylib → 1.24.4
└── Resources/
    ├── icon.icns
    ├── sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06/
    └── kitten-mini-en-v0_8/
```

## Build verification

After release build, watch `/tmp/livetranslate.log` startup line:
`SherpaTranscriber: recognizer loaded (provider=coreml)` — `provider=cpu` means CoreML EP didn't link (~2× decode latency).
