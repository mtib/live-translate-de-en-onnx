# ONNX Topic Summarizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every 60 seconds, feed the last 5 minutes of translated sentences to a small on-device LLM (Qwen2.5-0.5B int4, via onnxruntime-genai) and display the resulting topic label + 2-sentence summary in both the floating overlay and menu-bar popover.

**Architecture:** A new `TopicSummarizer` Swift actor wraps the onnxruntime-genai C API; it is called from `Pipeline` on a background 60-second timer. A new `CGenAI` SwiftPM target mirrors the pattern used by the existing `CSherpaOnnx` target: headers in `Sources/CGenAI/include/`, dylib in `external/onnxruntime-genai/lib/`, bundled into `Contents/Frameworks/` by `build.sh`. The Qwen model (~862 MB, int4 ONNX format) is downloaded to `build/llm-models/` by `tools/download-llm.sh` and copied to `Contents/Resources/llm-model/` at bundle time.

**Tech Stack:** onnxruntime-genai C API (`ort_genai_c.h`), Qwen2.5-0.5B-Instruct int4 ONNX, Swift actor, existing SwiftPM `.target` + `linkerSettings .unsafeFlags` pattern.

---

## File Map

| Action | Path | What changes |
|---|---|---|
| Create | `Sources/CGenAI/include/ort_genai_c.h` | Copied from onnxruntime-genai release tarball by `tools/download-llm.sh` |
| Create | `Sources/CGenAI/include/module.modulemap` | Makes the C header importable as `CGenAI` in Swift |
| Create | `tools/download-llm.sh` | Idempotent script: downloads genai dylib + header + Qwen model |
| Create | `Sources/LiveTranslate/TopicSummarizer.swift` | Swift actor: load/unload model, build prompt, generate, parse |
| Modify | `Sources/LiveTranslate/Types.swift` | Add `TranscriptSummary` struct |
| Modify | `Sources/LiveTranslate/Pipeline.swift` | Add `@Published var transcriptSummary`, 60 s timer, call summarizer |
| Modify | `Sources/LiveTranslate/TranscriptView.swift` | Show topic + summary below the language bar |
| Modify | `Sources/LiveTranslate/MenuBarView.swift` | Show topic + summary in the compact popover |
| Modify | `Package.swift` | Add `CGenAI` target; add it to `LiveTranslate` dependencies |
| Modify | `build.sh` | Call `download-llm.sh`; add genai dylib to `Frameworks/`; copy model to `Resources/` |
| Modify | `CLAUDE.md` | Files table, Roadmap, tools list |

---

## Task 1: Download onnxruntime-genai and create the CGenAI bridge

**Files:**
- Create: `tools/download-llm.sh`
- Create: `Sources/CGenAI/include/ort_genai_c.h` (populated by the script)
- Create: `Sources/CGenAI/include/module.modulemap`

- [ ] **Step 1: Create `tools/download-llm.sh`**

```bash
#!/usr/bin/env bash
# Downloads onnxruntime-genai dylib + C header and the Qwen2.5-0.5B int4 ONNX model.
# Idempotent: skips files that are already present.
set -euo pipefail

cd "$(dirname "$0")/.."

GENAI_VERSION="0.13.1"
GENAI_DIR="external/onnxruntime-genai"
GENAI_LIB="${GENAI_DIR}/lib"
GENAI_INCLUDE="${GENAI_DIR}/include"
MODEL_DIR="build/llm-models/qwen-0.5b"

# ── onnxruntime-genai dylib + header ────────────────────────────────────────
if [[ ! -f "${GENAI_LIB}/libonnxruntime-genai.dylib" ]]; then
    echo "→ Downloading onnxruntime-genai v${GENAI_VERSION} for osx-arm64..."
    TMP=$(mktemp -d)
    trap 'rm -rf "${TMP}"' EXIT
    curl -fsSL \
        "https://github.com/microsoft/onnxruntime-genai/releases/download/v${GENAI_VERSION}/onnxruntime-genai-${GENAI_VERSION}-osx-arm64.tar.gz" \
        -o "${TMP}/genai.tar.gz"
    mkdir -p "${TMP}/extracted"
    tar -xzf "${TMP}/genai.tar.gz" -C "${TMP}/extracted" --strip-components=1
    mkdir -p "${GENAI_LIB}" "${GENAI_INCLUDE}"
    # Copy dylib(s)
    find "${TMP}/extracted" -name "*.dylib" -exec cp {} "${GENAI_LIB}/" \;
    # Copy C header
    if [[ -f "${TMP}/extracted/include/ort_genai_c.h" ]]; then
        cp "${TMP}/extracted/include/ort_genai_c.h" "${GENAI_INCLUDE}/"
    elif [[ -f "${TMP}/extracted/include/onnxruntime_genai_c.h" ]]; then
        cp "${TMP}/extracted/include/onnxruntime_genai_c.h" "${GENAI_INCLUDE}/ort_genai_c.h"
    else
        echo "ERROR: Could not find genai C header in release tarball."
        echo "       Contents of ${TMP}/extracted/include/:"
        ls "${TMP}/extracted/include/" 2>/dev/null || true
        exit 1
    fi
    echo "  genai dylib and header extracted."
else
    echo "  onnxruntime-genai already present, skipping."
fi

# Copy the genai header to the SwiftPM bridge target (idempotent)
mkdir -p "Sources/CGenAI/include"
cp "${GENAI_INCLUDE}/ort_genai_c.h" "Sources/CGenAI/include/ort_genai_c.h"

# ── Inspect onnxruntime dependency ──────────────────────────────────────────
echo "→ Checking onnxruntime dependency of libonnxruntime-genai.dylib..."
ORT_DEP=$(otool -L "${GENAI_LIB}/libonnxruntime-genai.dylib" 2>/dev/null \
    | grep -o 'libonnxruntime[^ ]*' | head -1 || true)
echo "  genai links against: ${ORT_DEP:-<could not determine>}"
echo "  sherpa uses:         libonnxruntime.1.24.4.dylib"
echo "  (If major versions differ and you get symbol errors, set ORT_FOR_GENAI"
echo "   to a matching onnxruntime dylib path and re-run this script.)"

# If the tarball included a bundled onnxruntime, keep it in GENAI_LIB.
# build.sh will handle picking the right one for Frameworks/.

# ── Qwen2.5-0.5B int4 ONNX model ────────────────────────────────────────────
MODEL_REPO="hazemmabbas/Qwen2.5-0.5B-int4-block-32-acc-3-Instruct-onnx-cpu"
HF_BASE="https://huggingface.co/${MODEL_REPO}/resolve/main"

mkdir -p "${MODEL_DIR}"

for FNAME in \
    "genai_config.json" \
    "model.onnx" \
    "model.onnx.data" \
    "tokenizer.json" \
    "tokenizer_config.json" \
    "special_tokens_map.json"
do
    if [[ ! -f "${MODEL_DIR}/${FNAME}" ]]; then
        echo "→ Downloading ${FNAME}..."
        curl -fL \
            --progress-bar \
            "${HF_BASE}/${FNAME}" \
            -o "${MODEL_DIR}/${FNAME}"
    else
        echo "  ${FNAME} already present."
    fi
done

echo "✓ LLM assets ready."
echo "  genai dylib: ${GENAI_LIB}/"
echo "  model:       ${MODEL_DIR}/"
```

Make it executable:
```bash
chmod +x tools/download-llm.sh
```

- [ ] **Step 2: Create `Sources/CGenAI/include/module.modulemap`**

```
module CGenAI {
    header "ort_genai_c.h"
    export *
}
```

- [ ] **Step 3: Run the download script to populate the header and dylib**

```bash
cd /Users/mtib/transcrybe-diy
./tools/download-llm.sh
```

Expected output ends with:
```
✓ LLM assets ready.
  genai dylib: external/onnxruntime-genai/lib/
  model:       build/llm-models/qwen-0.5b/
```

Verify the header exists:
```bash
ls Sources/CGenAI/include/
# ort_genai_c.h  module.modulemap
```

Verify at least one dylib in the lib dir:
```bash
ls external/onnxruntime-genai/lib/
# libonnxruntime-genai.dylib  (and possibly libonnxruntime.dylib)
```

Note the onnxruntime dep line printed by the script. If it says `libonnxruntime.dylib` (unversioned), the existing symlink created by `build.sh` satisfies it. If it says `libonnxruntime.1.X.Y.dylib` with X < 24, you'll need to download that ort version into `external/onnxruntime-genai/lib/` as well (but try without it first — the symbol set is stable across patch versions).

- [ ] **Step 4: Update `Package.swift` to add the `CGenAI` target**

Replace the entire file with:

```swift
// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "LiveTranslate",
    platforms: [.macOS(.v15)],
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
                .unsafeFlags([
                    "-Wno-implicit-function-declaration",
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
                    "-lonnxruntime",
                ]),
            ]
        ),
        // Thin bridge target around the onnxruntime-genai shared dylib.
        // The dylib lives under external/onnxruntime-genai/lib/ and is
        // downloaded by tools/download-llm.sh. build.sh copies it into
        // the app bundle's Frameworks/ directory alongside the sherpa dylibs.
        .target(
            name: "CGenAI",
            path: "Sources/CGenAI",
            publicHeadersPath: "include",
            linkerSettings: [
                .unsafeFlags([
                    "-L./external/onnxruntime-genai/lib",
                    "-lonnxruntime-genai",
                ]),
            ]
        ),
        .executableTarget(
            name: "LiveTranslate",
            dependencies: ["CRNNoise", "CSherpaOnnx", "CGenAI"],
            path: "Sources/LiveTranslate",
            swiftSettings: [
                .swiftLanguageMode(.v5)
            ]
        )
    ]
)
```

- [ ] **Step 5: Verify the package resolves and compiles**

```bash
cd /Users/mtib/transcrybe-diy
LIBRARY_PATH="external/sherpa-onnx/lib:external/onnxruntime-genai/lib" \
    swift build -c debug 2>&1 | tail -20
```

Expected: build succeeds (or fails only on Swift source errors, not linker errors). If you see `library not found for -lonnxruntime-genai`, check that `external/onnxruntime-genai/lib/libonnxruntime-genai.dylib` exists.

- [ ] **Step 6: Commit**

```bash
git add \
    tools/download-llm.sh \
    Sources/CGenAI/include/module.modulemap \
    Sources/CGenAI/include/ort_genai_c.h \
    Package.swift
git commit -m "feat: add CGenAI SwiftPM bridge target for onnxruntime-genai

Mirrors the CSherpaOnnx pattern: header in Sources/CGenAI/include/,
dylib in external/onnxruntime-genai/lib/ (downloaded by download-llm.sh),
bundled into Frameworks/ at build time."
```

---

## Task 2: Add `TranscriptSummary` to Types.swift

**Files:**
- Modify: `Sources/LiveTranslate/Types.swift`

- [ ] **Step 1: Read the current end of `Types.swift` to find the right insertion point**

```bash
grep -n "^struct\|^enum\|^protocol\|^class" Sources/LiveTranslate/Types.swift
```

- [ ] **Step 2: Add `TranscriptSummary` to `Types.swift`**

Append after the last type definition in `Types.swift`. Find the last `}` in the file and add after it:

```swift
/// Result produced by TopicSummarizer: a short topic label and a two-sentence
/// summary of the current conversation. Shown in both the floating overlay
/// and the menu-bar popover while a session is running.
struct TranscriptSummary: Equatable {
    /// Short phrase, e.g. "Software architecture discussion".
    let topic: String
    /// One or two sentences summarising the most recent exchange.
    let summary: String
}
```

- [ ] **Step 3: Build to confirm no regressions**

```bash
LIBRARY_PATH="external/sherpa-onnx/lib:external/onnxruntime-genai/lib" \
    swift build -c debug 2>&1 | grep -E "error:|warning:|Build complete"
```

Expected: `Build complete!`

- [ ] **Step 4: Commit**

```bash
git add Sources/LiveTranslate/Types.swift
git commit -m "feat: add TranscriptSummary type for LLM topic/summary output"
```

---

## Task 3: Implement `TopicSummarizer`

**Files:**
- Create: `Sources/LiveTranslate/TopicSummarizer.swift`

This is the Swift actor that wraps the onnxruntime-genai C API. It is lazily loaded on first use and stays resident for the session duration.

- [ ] **Step 1: Create `Sources/LiveTranslate/TopicSummarizer.swift`**

```swift
import Foundation
import CGenAI

// MARK: - Errors

enum TopicSummarizerError: Error {
    case modelNotFound(String)
    case loadFailed(String)
    case generateFailed(String)
    case parseFailed(String)
}

// MARK: - Actor

/// Wraps onnxruntime-genai's C API to run Qwen2.5-0.5B-Instruct locally.
/// Lazy-loads the model on first `summarize()` call; stays resident until
/// `unload()` is called (typically on pipeline stop). All methods are
/// isolated to this actor — callers must `await`.
actor TopicSummarizer {

    // Opaque C pointers — never escape the actor.
    private var model: OpaquePointer?     // OgaModel*
    private var tokenizer: OpaquePointer? // OgaTokenizer*

    private let modelDirPath: String

    /// - Parameter modelDirPath: Path to the directory containing
    ///   `genai_config.json`, `model.onnx`, `tokenizer.json`, etc.
    init(modelDirPath: String) {
        self.modelDirPath = modelDirPath
    }

    // MARK: - Lifecycle

    /// Load the ONNX model and tokenizer. No-op if already loaded.
    /// Throws `TopicSummarizerError.loadFailed` on failure.
    func load() throws {
        guard model == nil else { return }
        guard FileManager.default.fileExists(atPath: modelDirPath) else {
            throw TopicSummarizerError.modelNotFound(modelDirPath)
        }

        var m: OpaquePointer?
        if let err = OgaCreateModel(modelDirPath, &m) {
            let msg = String(cString: OgaResultGetError(err))
            OgaDestroyResult(err)
            throw TopicSummarizerError.loadFailed("OgaCreateModel: \(msg)")
        }
        model = m

        var t: OpaquePointer?
        if let err = OgaCreateTokenizer(m, &t) {
            let msg = String(cString: OgaResultGetError(err))
            OgaDestroyResult(err)
            OgaDestroyModel(m)
            model = nil
            throw TopicSummarizerError.loadFailed("OgaCreateTokenizer: \(msg)")
        }
        tokenizer = t
        Log.line("TopicSummarizer: model loaded from \(modelDirPath)")
    }

    /// Release the model and tokenizer from memory.
    func unload() {
        if let t = tokenizer { OgaDestroyTokenizer(t); tokenizer = nil }
        if let m = model { OgaDestroyModel(m); model = nil }
        Log.line("TopicSummarizer: model unloaded")
    }

    // MARK: - Inference

    /// Build a prompt, run the LLM, return parsed topic + summary.
    ///
    /// - Parameters:
    ///   - translations: Translated sentences from the last 5 minutes.
    ///   - previous: The result from the previous cycle, fed back as context.
    func summarize(
        translations: [String],
        previous: TranscriptSummary?
    ) throws -> TranscriptSummary {
        guard let model = model, let tokenizer = tokenizer else {
            throw TopicSummarizerError.loadFailed("Model not loaded — call load() first.")
        }

        let prompt = buildPrompt(translations: translations, previous: previous)

        // ── Tokenise prompt ────────────────────────────────────────────────
        var promptSeqs: OpaquePointer?
        if let err = OgaCreateSequences(&promptSeqs) {
            defer { OgaDestroyResult(err) }
            throw TopicSummarizerError.generateFailed("OgaCreateSequences: \(String(cString: OgaResultGetError(err)))")
        }
        defer { OgaDestroySequences(promptSeqs) }

        if let err = OgaTokenizerEncode(tokenizer, prompt, promptSeqs) {
            defer { OgaDestroyResult(err) }
            throw TopicSummarizerError.generateFailed("OgaTokenizerEncode: \(String(cString: OgaResultGetError(err)))")
        }

        // ── Generator params ────────────────────────────────────────────────
        var params: OpaquePointer?
        if let err = OgaCreateGeneratorParams(model, &params) {
            defer { OgaDestroyResult(err) }
            throw TopicSummarizerError.generateFailed("OgaCreateGeneratorParams: \(String(cString: OgaResultGetError(err)))")
        }
        defer { OgaDestroyGeneratorParams(params) }

        // max_new_tokens = 200  (topic + 2 sentences ≈ 60–120 tokens)
        // temperature = 0  → greedy (deterministic, stable results)
        OgaGeneratorParamsSetSearchNumber(params, "max_new_tokens", 200)
        OgaGeneratorParamsSetSearchNumber(params, "temperature", 0)
        if let err = OgaGeneratorParamsSetInputSequences(params, promptSeqs) {
            defer { OgaDestroyResult(err) }
            throw TopicSummarizerError.generateFailed("SetInputSequences: \(String(cString: OgaResultGetError(err)))")
        }

        // ── Generate ───────────────────────────────────────────────────────
        var outputSeqs: OpaquePointer?
        if let err = OgaGenerate(model, params, &outputSeqs) {
            defer { OgaDestroyResult(err) }
            throw TopicSummarizerError.generateFailed("OgaGenerate: \(String(cString: OgaResultGetError(err)))")
        }
        defer { OgaDestroySequences(outputSeqs) }

        // The output sequence contains prompt tokens + generated tokens.
        // Decode the full sequence; `parse` will find "Topic:" and "Summary:".
        let seqLen = OgaSequencesGetSequenceCount(outputSeqs, 0)
        guard let tokenPtr = OgaSequencesGetSequenceData(outputSeqs, 0) else {
            throw TopicSummarizerError.generateFailed("OgaSequencesGetSequenceData returned nil")
        }

        // ── Decode ─────────────────────────────────────────────────────────
        var outCStr: UnsafePointer<CChar>?
        if let err = OgaTokenizerDecode(tokenizer, tokenPtr, seqLen, &outCStr) {
            defer { OgaDestroyResult(err) }
            throw TopicSummarizerError.generateFailed("OgaTokenizerDecode: \(String(cString: OgaResultGetError(err)))")
        }
        let generated = outCStr.map { String(cString: $0) } ?? ""
        Log.line("TopicSummarizer raw output: \(generated.prefix(200))")

        return try parse(generated)
    }

    // MARK: - Prompt builder

    private func buildPrompt(
        translations: [String],
        previous: TranscriptSummary?
    ) -> String {
        var p = """
        <|im_start|>system
        You analyze conversation transcripts. Reply with EXACTLY two lines:
        Topic: <one short phrase>
        Summary: <two sentences max>
        No other text.<|im_end|>
        <|im_start|>user
        """

        if let prev = previous {
            p += "\nPrevious topic: \(prev.topic)"
            p += "\nPrevious summary: \(prev.summary)\n"
        }

        p += "\nRecent transcript (last ~5 minutes):\n"
        p += translations.joined(separator: "\n")
        p += "\n<|im_end|>\n<|im_start|>assistant\n"
        return p
    }

    // MARK: - Output parser

    /// Scans `text` for "Topic: ..." and "Summary: ..." lines.
    /// Throws if neither is found (model didn't follow the format).
    private func parse(_ text: String) throws -> TranscriptSummary {
        var topic = ""
        var summary = ""

        for line in text.components(separatedBy: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("Topic:") && topic.isEmpty {
                topic = String(trimmed.dropFirst(6)).trimmingCharacters(in: .whitespaces)
            } else if trimmed.hasPrefix("Summary:") && summary.isEmpty {
                summary = String(trimmed.dropFirst(8)).trimmingCharacters(in: .whitespaces)
            }
        }

        // Graceful fallback: if the model ignored the format but produced text,
        // use the first non-empty line as topic and the rest as summary.
        if topic.isEmpty {
            let assistantMarker = "<|im_start|>assistant"
            let responseStart: String
            if let range = text.range(of: assistantMarker) {
                responseStart = String(text[range.upperBound...])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            } else {
                responseStart = text.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            let lines = responseStart.components(separatedBy: "\n")
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty && !$0.hasPrefix("<|") }
            if lines.isEmpty {
                throw TopicSummarizerError.parseFailed("Empty model output")
            }
            topic = lines[0]
            summary = lines.dropFirst().joined(separator: " ")
        }

        return TranscriptSummary(
            topic: topic.isEmpty ? "Unknown" : topic,
            summary: summary.isEmpty ? "No summary available." : summary
        )
    }
}
```

- [ ] **Step 2: Build to catch any compilation errors**

```bash
LIBRARY_PATH="external/sherpa-onnx/lib:external/onnxruntime-genai/lib" \
    swift build -c debug 2>&1 | grep -E "error:|Build complete"
```

**If you see errors about missing symbols in `CGenAI`** (e.g., `OgaCreateModel` undeclared), open `Sources/CGenAI/include/ort_genai_c.h` and search for the actual function name. The onnxruntime-genai C API has been stable but function names have `Oga` prefix — check for minor spelling differences and update `TopicSummarizer.swift` accordingly.

**If you see `OgaDestroyResult` not found**, the function might be called `OgaResultDestroy` — check the header and fix.

Expected when everything is right: `Build complete!`

- [ ] **Step 3: Commit**

```bash
git add Sources/LiveTranslate/TopicSummarizer.swift
git commit -m "feat: implement TopicSummarizer actor using onnxruntime-genai C API

Qwen2.5-0.5B int4 ONNX model. Lazy-loads on first summarize() call.
Prompt follows the <|im_start|>system Qwen2.5 chat template.
Parses 'Topic:' and 'Summary:' lines from generated output."
```

---

## Task 4: Wire `TopicSummarizer` into `Pipeline`

**Files:**
- Modify: `Sources/LiveTranslate/Pipeline.swift`

Read the current Pipeline.swift before editing:
```bash
wc -l Sources/LiveTranslate/Pipeline.swift
grep -n "@Published\|private var\|func run\|func stop\|func toggle" Sources/LiveTranslate/Pipeline.swift | head -30
```

- [ ] **Step 1: Add `@Published var transcriptSummary` and summarizer fields**

In `Pipeline.swift`, find the block of `@Published` properties and add:

```swift
/// Current LLM-generated topic and summary. `nil` until the first
/// 60-second cycle completes (or if no LLM model is bundled).
@Published var transcriptSummary: TranscriptSummary? = nil
```

After the `@Published` block, add private fields (near the other private vars):

```swift
/// Runs the 60-second topic+summary loop while a session is active.
private var summaryLoopTask: Task<Void, Never>? = nil
/// Shared across summary cycles so the LLM can reference its previous output.
private var lastSummary: TranscriptSummary? = nil
```

- [ ] **Step 2: Create a `makeSummarizer()` helper that returns the actor or nil**

Add this private method to `Pipeline`:

```swift
/// Returns a `TopicSummarizer` pointed at the bundled model directory,
/// or `nil` if the model is not present (e.g. dev build without download-llm.sh).
private func makeSummarizer() -> TopicSummarizer? {
    let modelDir = Bundle.main.bundleURL
        .appendingPathComponent("Contents/Resources/llm-model")
        .path
    guard FileManager.default.fileExists(atPath: modelDir + "/genai_config.json") else {
        Log.line("TopicSummarizer: model not found at \(modelDir) — skipping summary loop")
        return nil
    }
    return TopicSummarizer(modelDirPath: modelDir)
}
```

- [ ] **Step 3: Add `startSummaryLoop` and `stopSummaryLoop` methods**

```swift
private func startSummaryLoop() {
    guard summaryLoopTask == nil else { return }
    guard let summarizer = makeSummarizer() else { return }

    summaryLoopTask = Task.detached(priority: .background) { [weak self] in
        // Load the model once per session; unload when the task is cancelled.
        do {
            try await summarizer.load()
        } catch {
            Log.line("TopicSummarizer load error: \(error)")
            return
        }
        defer { Task { await summarizer.unload() } }

        // Wait 15 s before the first attempt so the user has said something.
        try? await Task.sleep(for: .seconds(15))

        while !Task.isCancelled {
            await self?.runOneSummaryCycle(summarizer: summarizer)
            // Sleep 60 s between cycles. Break early if cancelled.
            try? await Task.sleep(for: .seconds(60))
        }
    }
}

private func stopSummaryLoop() {
    summaryLoopTask?.cancel()
    summaryLoopTask = nil
    lastSummary = nil
    transcriptSummary = nil
}

/// Pulls the last 5 minutes of sentences, calls the LLM, updates @Published.
/// Called from a background Task — hops to @MainActor only to update state.
private func runOneSummaryCycle(summarizer: TopicSummarizer) async {
    // Capture sentences on MainActor.
    let recentTexts: [String] = await MainActor.run { [weak self] in
        guard let self else { return [] }
        let cutoff = Date().addingTimeInterval(-5 * 60)
        return self.sentences
            .filter { $0.createdAt >= cutoff }
            .compactMap { $0.translation ?? $0.transcription }
    }

    guard recentTexts.count >= 3 else {
        Log.line("TopicSummarizer: only \(recentTexts.count) sentences — skipping cycle")
        return
    }

    let previous = await MainActor.run { [weak self] in self?.lastSummary }

    do {
        let result = try await summarizer.summarize(
            translations: recentTexts,
            previous: previous
        )
        await MainActor.run { [weak self] in
            self?.transcriptSummary = result
            self?.lastSummary = result
        }
        Log.line("TopicSummarizer: topic=\(result.topic)")
    } catch {
        Log.line("TopicSummarizer error: \(error)")
    }
}
```

- [ ] **Step 4: Call `startSummaryLoop()` from `run()` and `stopSummaryLoop()` from `stop()`**

Find the `run()` method — it starts the audio pipelines. After the pipelines are started (look for where `runTask` is set or after the `SourcePipeline` start calls), add:

```swift
startSummaryLoop()
```

Find the `stop()` or `toggle()` method where the session ends. After `runTask?.cancel()` or wherever the pipeline is torn down, add:

```swift
stopSummaryLoop()
```

- [ ] **Step 5: Build**

```bash
LIBRARY_PATH="external/sherpa-onnx/lib:external/onnxruntime-genai/lib" \
    swift build -c debug 2>&1 | grep -E "error:|Build complete"
```

Expected: `Build complete!`

- [ ] **Step 6: Commit**

```bash
git add Sources/LiveTranslate/Pipeline.swift
git commit -m "feat: add 60 s topic/summary loop to Pipeline

Lazy-loads TopicSummarizer on session start; first attempt after 15 s,
then every 60 s. Uses last 5 min of sentences (≥ 3 required). Publishes
TranscriptSummary on @MainActor; resets on session stop."
```

---

## Task 5: Display `transcriptSummary` in `TranscriptView`

**Files:**
- Modify: `Sources/LiveTranslate/TranscriptView.swift`

First, read the current fullBar and compactBar sections:
```bash
grep -n "fullBar\|compactBar\|private var" Sources/LiveTranslate/TranscriptView.swift | head -30
```

- [ ] **Step 1: Add a `summaryBar` view to `TranscriptView`**

Find the `body` or the outermost `VStack` in `TranscriptView`. Locate where `compactBar` or `fullBar` ends and the sentence list begins. Add `summaryBar` between the bar and the sentence list:

```swift
/// Shows the LLM-generated topic and summary when available.
@ViewBuilder
private var summaryBar: some View {
    if let s = pipeline.transcriptSummary {
        VStack(alignment: .leading, spacing: 2) {
            Text(s.topic)
                .font(.caption.bold())
                .foregroundStyle(.primary)
                .lineLimit(1)
            Text(s.summary)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(3)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 4)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial.opacity(0.6))
        .cornerRadius(6)
        .padding(.horizontal, 8)
        .transition(.opacity.combined(with: .move(edge: .top)))
    }
}
```

In the body `VStack`, add `summaryBar` between the control bar and the `ScrollView`:

```swift
// (existing bar code)
compactBar   // or fullBar — whichever is currently at the top
summaryBar   // ← add this line
// (existing ScrollView / sentence list)
```

Wrap the insertion in `withAnimation` by adding `.animation(.easeInOut(duration: 0.25), value: pipeline.transcriptSummary != nil)` on the `VStack`.

- [ ] **Step 2: Build and visually verify with a fixture**

```bash
LIBRARY_PATH="external/sherpa-onnx/lib:external/onnxruntime-genai/lib" \
    swift build -c debug 2>&1 | grep -E "error:|Build complete"
```

To test the layout without the LLM model: open `App.swift` → `loadDebugFixtures()`, temporarily set `pipeline.transcriptSummary = TranscriptSummary(topic: "Software architecture", summary: "The speakers discussed microservice design patterns. They agreed on a hexagonal approach.")` in `loadDebugFixtures()`, build and run `open build/LiveTranslate.app`, then press Cmd+Shift+D. You should see the summary bar appear below the language label. Remove the temporary line before committing.

- [ ] **Step 3: Commit**

```bash
git add Sources/LiveTranslate/TranscriptView.swift
git commit -m "feat: show topic/summary bar in TranscriptView

Appears below the control bar when transcriptSummary is non-nil.
Fades in/out; shows topic in bold caption and summary in caption2."
```

---

## Task 6: Display `transcriptSummary` in `MenuBarView`

**Files:**
- Modify: `Sources/LiveTranslate/MenuBarView.swift`

- [ ] **Step 1: Add `summaryRow` to `MenuBarView`**

In `MenuBarView.swift`, find the main `VStack` in `body`. After `compactBar` and before `sentenceList`, add:

```swift
if let s = pipeline.transcriptSummary {
    VStack(alignment: .leading, spacing: 2) {
        Text(s.topic)
            .font(.caption.bold())
            .lineLimit(1)
        Text(s.summary)
            .font(.caption2)
            .foregroundStyle(.secondary)
            .lineLimit(3)
            .fixedSize(horizontal: false, vertical: true)
    }
    .frame(maxWidth: .infinity, alignment: .leading)
    .padding(.vertical, 4)
    .transition(.opacity)
}
```

Add `.animation(.easeInOut(duration: 0.25), value: pipeline.transcriptSummary != nil)` on the surrounding `VStack`.

- [ ] **Step 2: Build**

```bash
LIBRARY_PATH="external/sherpa-onnx/lib:external/onnxruntime-genai/lib" \
    swift build -c debug 2>&1 | grep -E "error:|Build complete"
```

Expected: `Build complete!`

- [ ] **Step 3: Commit**

```bash
git add Sources/LiveTranslate/MenuBarView.swift
git commit -m "feat: show topic/summary in MenuBarView popover"
```

---

## Task 7: Update `build.sh` to bundle genai dylib and LLM model

**Files:**
- Modify: `build.sh`

- [ ] **Step 1: Add `download-llm.sh` call and genai vars to `build.sh`**

After the existing `MODELS_DIR="build/sherpa-models"` line, add:

```bash
GENAI_LIB="external/onnxruntime-genai/lib"
LLM_MODEL_DIR="build/llm-models/qwen-0.5b"
```

After the line `./tools/download-sherpa.sh`, add:

```bash
# Download genai dylib + LLM model (idempotent).
./tools/download-llm.sh
```

- [ ] **Step 2: Update the swift build command to include genai lib in LIBRARY_PATH**

Change:
```bash
LIBRARY_PATH="${SHERPA_LIB}" swift build -c "${CONFIG}"
```
to:
```bash
LIBRARY_PATH="${SHERPA_LIB}:${GENAI_LIB}" swift build -c "${CONFIG}"
```

Also update the `--show-bin-path` call:
```bash
BIN_PATH="$(LIBRARY_PATH="${SHERPA_LIB}:${GENAI_LIB}" swift build -c "${CONFIG}" --show-bin-path)/${APP_NAME}"
```

- [ ] **Step 3: Add the genai dylib to the Frameworks/ copy loop**

In the `for DYLIB in ...` loop that copies sherpa dylibs, add the genai dylib entry. The block currently looks like:

```bash
for DYLIB in \
    "${SHERPA_LIB}/libsherpa-onnx-c-api.dylib" \
    "${SHERPA_LIB}/libonnxruntime.1.24.4.dylib"
```

Change it to:

```bash
for DYLIB in \
    "${SHERPA_LIB}/libsherpa-onnx-c-api.dylib" \
    "${SHERPA_LIB}/libonnxruntime.1.24.4.dylib" \
    "${GENAI_LIB}/libonnxruntime-genai.dylib"
```

Also, after the existing `ln -sf` symlink creation, add symlinks for any additional onnxruntime dylib that genai may have bundled:

```bash
# If the genai tarball included its own libonnxruntime (different version),
# copy and symlink it so genai's dylib finds it at runtime.
for EXTRA_ORT in "${GENAI_LIB}/libonnxruntime."*.dylib; do
    if [[ -f "${EXTRA_ORT}" ]]; then
        BNAME=$(basename "${EXTRA_ORT}")
        if [[ ! -f "${APP_DIR}/Contents/Frameworks/${BNAME}" ]]; then
            cp "${EXTRA_ORT}" "${APP_DIR}/Contents/Frameworks/"
            echo "  bundled extra ort: ${BNAME}"
        fi
    fi
done
```

- [ ] **Step 4: Add LLM model copy to Resources/**

After the TTS model copy block, add:

```bash
# LLM model for topic/summary (TopicSummarizer)
if [[ -d "${LLM_MODEL_DIR}" ]]; then
    rm -rf "${APP_DIR}/Contents/Resources/llm-model"
    cp -R "${LLM_MODEL_DIR}" "${APP_DIR}/Contents/Resources/llm-model"
    echo "  bundled LLM model ($(du -sh "${LLM_MODEL_DIR}" | cut -f1))"
else
    echo "WARNING: LLM model not found at ${LLM_MODEL_DIR} — run tools/download-llm.sh"
    echo "         App will launch without topic/summary feature."
fi
```

- [ ] **Step 5: Run a full build**

```bash
LIVETRANSLATE_SIGN_IDENTITY=LiveTranslateDev ./build.sh 2>&1 | tail -30
```

Expected final lines:
```
  bundled libonnxruntime-genai.dylib
  bundled LLM model (862M)
✓ built build/LiveTranslate.app
  run with: open build/LiveTranslate.app
```

If you see `WARNING: LLM model not found`, run `./tools/download-llm.sh` separately and then rebuild.

- [ ] **Step 6: Launch and verify the model loads**

```bash
open build/LiveTranslate.app
tail -f /tmp/livetranslate.log
```

Start a recording session. After ~15 seconds you should see in the log:
```
TopicSummarizer: model loaded from .../Contents/Resources/llm-model
```

After the first summary cycle (~15 s + inference time):
```
TopicSummarizer raw output: <|im_start|>assistant
Topic: ...
Summary: ...
TopicSummarizer: topic=...
```

The summary bar should appear in the overlay and popover.

- [ ] **Step 7: Commit**

```bash
git add build.sh
git commit -m "build: bundle onnxruntime-genai dylib and Qwen LLM model

download-llm.sh is called as part of the build. The genai dylib goes
into Frameworks/ (alongside sherpa). The Qwen model directory is copied
to Contents/Resources/llm-model/. Build succeeds silently if the model
is absent (feature is disabled at runtime)."
```

---

## Task 8: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add `TopicSummarizer.swift` to the Files table**

Find the table row for `TranscriptView.swift`. Add a new row after it:

```markdown
| `TopicSummarizer.swift` | `actor` wrapping the onnxruntime-genai C API. Lazy-loads Qwen2.5-0.5B int4 on first `summarize()` call. Builds a Qwen chat-template prompt from the last 5 min of translations + previous summary; runs greedy decoding (temperature=0, max_new_tokens=200); parses `Topic:` and `Summary:` lines from output. Unloaded on session stop. |
```

- [ ] **Step 2: Add `CGenAI` to the Tools/SDKs section**

Find the line `- SwiftUI` at the bottom of the Tools/SDKs list and add before it:

```markdown
- `onnxruntime-genai` (C API via `CGenAI` bridge target) — LLM text generation using Qwen2.5-0.5B int4 ONNX model; `OgaCreateModel`, `OgaGenerate`, `OgaTokenizerDecode`
```

- [ ] **Step 3: Update the Roadmap**

Find the `[ ] Speaker diarization` line. Add above it:

```markdown
- [x] On-device LLM topic+summary loop — Qwen2.5-0.5B int4 via onnxruntime-genai, runs every 60 s, shows topic label + 2-sentence summary in overlay and popover
```

- [ ] **Step 4: Add to build.sh step-by-step section**

Find step 1 in the `build.sh — step by step` section (`./tools/download-sherpa.sh`). Add after it:

```markdown
1b. `./tools/download-llm.sh` — fetches onnxruntime-genai dylib + header + Qwen2.5-0.5B int4 ONNX model (idempotent).
```

And in the copy steps, add:
```markdown
10b. Copy `libonnxruntime-genai.dylib` into `Contents/Frameworks/`.
10c. Copy `build/llm-models/qwen-0.5b/` → `Contents/Resources/llm-model/`.
```

- [ ] **Step 5: Commit the CLAUDE.md update**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for TopicSummarizer / onnxruntime-genai

Files table, Tools/SDKs, Roadmap, build.sh steps."
```

---

## Self-Review

**Spec coverage:**

| Requirement | Task |
|---|---|
| Small on-device LLM | Task 1 (onnxruntime-genai + Qwen2.5-0.5B int4) |
| Every ~1 minute | Task 4 (60 s loop with 15 s initial delay) |
| Last ~5 minutes of translations | Task 4 (`runOneSummaryCycle` filters by `createdAt`) |
| Topic label + ~2 sentence summary | Task 3 (`buildPrompt` system instruction), Task 3 (`parse`) |
| Feed previous topic/summary back | Task 3 (`previous` param in `buildPrompt`) |
| Display in overlay | Task 5 (`summaryBar` in `TranscriptView`) |
| Display in popover | Task 6 (`MenuBarView`) |
| Build system integration | Task 7 (`build.sh`) |
| Docs | Task 8 (`CLAUDE.md`) |
| Graceful degradation (no model) | Task 4 (`makeSummarizer()` returns nil if model absent; loop never starts) |

**Placeholder scan:** No TBDs. All code blocks are complete.

**Type consistency:**
- `TranscriptSummary` defined in Task 2 → used in Task 3 (return type of `summarize`), Task 4 (`@Published` + `lastSummary`), Task 5+6 (UI).
- `TopicSummarizer` defined in Task 3 → called in Task 4.
- `makeSummarizer()` → `TopicSummarizer?` → used in `startSummaryLoop()`.
- `runOneSummaryCycle(summarizer:)` takes `TopicSummarizer` (actor, not optional).
- `pipeline.transcriptSummary: TranscriptSummary?` used in Task 5+6.

All names are consistent across tasks.
