# LiveTranslate — agent context

macOS app. Live German→English captions over fullscreen video. Sherpa-onnx streaming zipformer ASR + Apple on-device Translation + kitten-mini ONNX TTS + LAN HTTP server (SSE + WAV + OBS overlay) + FoundationModels topic/summary loop. Per-stream pipelines (mic + SCK system audio). Branch `feature/onnx-streaming` — different codebase from `main` of `transcrybe-diy`.

## Rules

1. **Update docs in the same commit as meaningful code changes.** "Meaningful" = data flow, type shapes, lifecycle state machine, sentence splitting, file layout, build steps, permissions, model paths, or behaviors described in skills. Stale docs cause incorrect diffs and re-introduce fixed bugs.
2. **Persist learnings.** Every new non-obvious bug → append a numbered entry to `livetranslate-bug-history` skill. Never delete entries.
3. **Read all `Sources/LiveTranslate/*.swift` before editing.** Data flow crosses many files; boundaries are subtle. Skimming for one symbol misses patterns.
4. **Always build with the signing identity** (see `livetranslate-build-and-run` skill). `LIVETRANSLATE_SIGN_IDENTITY=LiveTranslateDev ./build.sh`. Non-interactive bash doesn't source `~/.zshrc`.

## Skills (`.claude/skills/`)

- **livetranslate-architecture** — data flow, files table, state machines, UUID continuity, sentence splitting, crosstalk, latency knobs. Read before any source edit.
- **livetranslate-bug-history** — numbered catalogue of bugs that have bitten us. Read before audio/sherpa/lifecycle/SwiftUI-translation/SCK changes. Append-only.
- **livetranslate-build-and-run** — build, sign, launch, TCC, sherpa downloads, bundle layout.

Each skill instructs: after use, update with new repo state and any new learnings, in the same commit.

## Compile-time configuration

`Sources/LiveTranslate/ModelConfig.swift` — `sourceLanguage`/`targetLanguage`, ASR/TTS model paths, ONNX EP (`coreml`). Retarget = change constants, swap ASR model dir, rebuild. No runtime pickers, no UserDefaults for language.
