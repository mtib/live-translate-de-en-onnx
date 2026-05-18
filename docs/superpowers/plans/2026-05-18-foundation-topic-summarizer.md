# Apple FoundationModels Topic Summarizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every 60 seconds, feed the last 5 minutes of translated sentences to the on-device Apple FoundationModels LLM and display the resulting topic label + 2-sentence summary in both the floating overlay and menu-bar popover.

**Architecture:** A new `TopicSummarizer` actor (macOS 26+) uses `LanguageModelSession` with a `@Generable` output struct for constrained structured generation — no text parsing, the framework guarantees schema-conformant output. `Pipeline` runs a 60 s background loop starting 15 s after session start, publishes `@Published var transcriptSummary: TranscriptSummary?`, and resets it on stop. The deployment target is bumped to macOS 26 to use the framework without `@available` clutter.

**Tech Stack:** `FoundationModels` (`LanguageModelSession`, `@Generable`, `@Guide`), Swift actor, zero new dependencies or downloads.

> **Supersedes:** `docs/superpowers/plans/2026-05-18-onnx-topic-summarizer.md` — the onnxruntime-genai approach was dropped in favour of Apple FoundationModels (Apple Intelligence required, no fallback).

---

## File Map

| Action | Path | What changes |
|---|---|---|
| Modify | `Package.swift` | Bump deployment target `.macOS(.v15)` → `.macOS(.v26)` |
| Create | `Sources/LiveTranslate/TopicSummarizer.swift` | `actor TopicSummarizer` wrapping `LanguageModelSession`; `@Generable TopicSummaryOutput` |
| Modify | `Sources/LiveTranslate/Types.swift` | Add `TranscriptSummary` struct (plain, no FoundationModels dep) |
| Modify | `Sources/LiveTranslate/Pipeline.swift` | `@Published var transcriptSummary`, 60 s timer, `startSummaryLoop` / `stopSummaryLoop` |
| Modify | `Sources/LiveTranslate/TranscriptView.swift` | `summaryBar` view below the control bar |
| Modify | `Sources/LiveTranslate/MenuBarView.swift` | Topic + summary rows in compact popover |
| Modify | `CLAUDE.md` | Files table, Tools/SDKs list, Roadmap |

---

## Task 1: Bump deployment target and verify build

**Files:**
- Modify: `Package.swift`

FoundationModels requires macOS 26. The current target is macOS 15. Bump it now so later tasks don't need `@available` sprinkled everywhere.

- [ ] **Step 1: Edit `Package.swift` — change the platforms line**

Open `Package.swift`. Change:
```swift
    platforms: [.macOS(.v15)],
```
to:
```swift
    platforms: [.macOS(.v26)],
```

No other changes to `Package.swift`.

- [ ] **Step 2: Build to confirm the toolchain accepts the new target**

```bash
cd /Users/mtib/transcrybe-diy
LIBRARY_PATH="external/sherpa-onnx/lib" swift build -c debug 2>&1 | tail -10
```

Expected: `Build complete!`

If you see `error: 'v26' is not a valid version` the swift-tools-version in the file needs to be bumped too. In that case, change the first line from:
```
// swift-tools-version:6.0
```
to:
```
// swift-tools-version:6.1
```
and retry.

- [ ] **Step 3: Commit**

```bash
git add Package.swift
git commit -m "build: bump deployment target to macOS 26 for FoundationModels"
```

---

## Task 2: Add `TranscriptSummary` to `Types.swift`

**Files:**
- Modify: `Sources/LiveTranslate/Types.swift`

`TranscriptSummary` is a plain Swift struct with no FoundationModels import so Pipeline and the UI can reference it without pulling in the framework. The `@Generable TopicSummaryOutput` lives separately in `TopicSummarizer.swift`.

- [ ] **Step 1: Find where to insert in `Types.swift`**

```bash
grep -n "^struct\|^enum\|^protocol\|^class" Sources/LiveTranslate/Types.swift
```

- [ ] **Step 2: Append `TranscriptSummary` at the end of `Types.swift`**

Add after the last closing brace:

```swift
/// LLM-generated topic label and short summary of the current conversation.
/// Produced by `TopicSummarizer` and published on `Pipeline.transcriptSummary`.
/// Shown in both `TranscriptView` and `MenuBarView` while a session is running.
struct TranscriptSummary: Equatable {
    /// Short phrase describing the topic, e.g. "Microservice design trade-offs".
    let topic: String
    /// One or two sentences summarising the most recent exchange.
    let summary: String
}
```

- [ ] **Step 3: Build**

```bash
LIBRARY_PATH="external/sherpa-onnx/lib" swift build -c debug 2>&1 | grep -E "error:|Build complete"
```

Expected: `Build complete!`

- [ ] **Step 4: Commit**

```bash
git add Sources/LiveTranslate/Types.swift
git commit -m "feat: add TranscriptSummary type"
```

---

## Task 3: Implement `TopicSummarizer`

**Files:**
- Create: `Sources/LiveTranslate/TopicSummarizer.swift`

- [ ] **Step 1: Create `Sources/LiveTranslate/TopicSummarizer.swift`**

```swift
import Foundation
import FoundationModels

// MARK: - Structured output type

/// The schema the LLM is constrained to generate. `@Generable` macro emits
/// a JSON schema the FoundationModels runtime uses for constrained decoding —
/// the model is guaranteed to return a value that decodes into this struct.
@Generable
struct TopicSummaryOutput {
    @Guide(description: "A short phrase (3-8 words) that labels the current conversation topic")
    var topic: String

    @Guide(description: "A two-sentence summary of the most recent exchange. Be concise.")
    var summary: String
}

// MARK: - Errors

enum TopicSummarizerError: Error {
    case modelUnavailable(String)
    case generationFailed(String)
}

// MARK: - Actor

/// Uses Apple FoundationModels (`LanguageModelSession`) to produce a rolling
/// topic label + summary from the last ~5 minutes of translated sentences.
/// Requires Apple Intelligence to be enabled on the device (macOS 26+).
actor TopicSummarizer {

    /// Check whether the system model is usable before constructing this actor.
    static func isAvailable() -> Bool {
        if case .available = SystemLanguageModel.default.availability {
            return true
        }
        return false
    }

    // MARK: - Inference

    /// Run one summary cycle. Creates a fresh `LanguageModelSession` per call
    /// (sessions carry conversation history; fresh sessions avoid stale context
    /// leaking between the 60-second ticks).
    ///
    /// - Parameters:
    ///   - translations: The translated (or source) text of sentences from the
    ///     last 5 minutes, in chronological order.
    ///   - previous: The result from the previous cycle, included in the prompt
    ///     so the model can return a stable topic when the subject hasn't changed.
    func summarize(
        translations: [String],
        previous: TranscriptSummary?
    ) async throws -> TranscriptSummary {
        guard case .available = SystemLanguageModel.default.availability else {
            let reason: String
            if case .unavailable(let r) = SystemLanguageModel.default.availability {
                reason = "\(r)"
            } else {
                reason = "unknown"
            }
            throw TopicSummarizerError.modelUnavailable(reason)
        }

        let prompt = buildPrompt(translations: translations, previous: previous)

        let session = LanguageModelSession(
            instructions: """
            You analyze live conversation transcripts and produce a short topic \
            label and a two-sentence summary. Be concise and factual. \
            If the topic hasn't changed from the previous cycle, return a \
            slightly refined version rather than inventing a new one.
            """
        )

        let result = try await session.respond(
            to: prompt,
            generating: TopicSummaryOutput.self
        )

        let output = result.content
        return TranscriptSummary(topic: output.topic, summary: output.summary)
    }

    // MARK: - Prompt

    private func buildPrompt(
        translations: [String],
        previous: TranscriptSummary?
    ) -> String {
        var p = ""

        if let prev = previous {
            p += "Previous topic: \(prev.topic)\n"
            p += "Previous summary: \(prev.summary)\n\n"
        }

        p += "Recent transcript (last ~5 minutes):\n"
        p += translations.joined(separator: "\n")

        return p
    }
}
```

- [ ] **Step 2: Build**

```bash
LIBRARY_PATH="external/sherpa-onnx/lib" swift build -c debug 2>&1 | grep -E "error:|Build complete"
```

**If you see `error: cannot find type 'LanguageModelSession'`:** confirm you're on macOS 26 and the swift toolchain is the one shipped with Xcode 26 / Command Line Tools 26. Run `swift --version` — it should report a 2025/2026 toolchain. FoundationModels is an OS framework, not a Swift package; it's available automatically once the deployment target is macOS 26.

**If you see `error: @Generable is not a known attribute`:** the macro plugin ships with the FoundationModels framework. Ensure Xcode 26 / Command Line Tools 26 are active via `xcode-select -p` (should point to a 2025/2026 tools install).

Expected when everything is right: `Build complete!`

- [ ] **Step 3: Commit**

```bash
git add Sources/LiveTranslate/TopicSummarizer.swift
git commit -m "feat: implement TopicSummarizer using Apple FoundationModels

LanguageModelSession with @Generable TopicSummaryOutput for constrained
structured decoding. No external dependencies, no model download —
uses the on-device Apple Intelligence model (requires macOS 26)."
```

---

## Task 4: Wire `TopicSummarizer` into `Pipeline`

**Files:**
- Modify: `Sources/LiveTranslate/Pipeline.swift`

First, read the current file to find the right locations:

```bash
grep -n "@Published\|private var\|func run\|func stop\|func toggle\|func clear" Sources/LiveTranslate/Pipeline.swift | head -40
```

- [ ] **Step 1: Add `@Published var transcriptSummary` and two private fields**

Find the block of `@Published` properties. Add:

```swift
/// LLM-generated topic and summary. `nil` until the first 60-second cycle
/// completes or when no session is running.
@Published var transcriptSummary: TranscriptSummary? = nil
```

Somewhere after the `@Published` block (near other `private var` fields), add:

```swift
/// Background task running the 60-second topic+summary loop.
private var summaryLoopTask: Task<Void, Never>? = nil
/// Carried across cycles so the LLM can detect topic stability.
private var lastSummary: TranscriptSummary? = nil
```

- [ ] **Step 2: Add `startSummaryLoop`, `stopSummaryLoop`, and `runOneSummaryCycle`**

Add these three methods to `Pipeline`. Place them near the bottom of the class, before the final `}`:

```swift
// MARK: - Topic / summary loop

private func startSummaryLoop() {
    guard summaryLoopTask == nil else { return }
    guard TopicSummarizer.isAvailable() else {
        Log.line("TopicSummarizer: Apple Intelligence not available — skipping summary loop")
        return
    }

    let summarizer = TopicSummarizer()

    summaryLoopTask = Task.detached(priority: .background) { [weak self] in
        // Wait 15 s before the first attempt so the user has said something.
        try? await Task.sleep(for: .seconds(15))

        while !Task.isCancelled {
            await self?.runOneSummaryCycle(summarizer: summarizer)
            try? await Task.sleep(for: .seconds(60))
        }
    }
    Log.line("TopicSummarizer: summary loop started")
}

private func stopSummaryLoop() {
    summaryLoopTask?.cancel()
    summaryLoopTask = nil
    lastSummary = nil
    transcriptSummary = nil
}

/// Captures the last 5 minutes of sentences, calls the LLM on a background
/// task, then hops to MainActor to publish the result.
private func runOneSummaryCycle(summarizer: TopicSummarizer) async {
    // Capture data on MainActor.
    let (recentTexts, previous): ([String], TranscriptSummary?) = await MainActor.run { [weak self] in
        guard let self else { return ([], nil) }
        let cutoff = Date().addingTimeInterval(-5 * 60)
        let texts = self.sentences
            .filter { $0.createdAt >= cutoff }
            .compactMap { $0.translation ?? $0.transcription }
        return (texts, self.lastSummary)
    }

    guard recentTexts.count >= 3 else {
        Log.line("TopicSummarizer: \(recentTexts.count) sentences — skipping (need ≥ 3)")
        return
    }

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

- [ ] **Step 3: Call `startSummaryLoop()` from `run()` and `stopSummaryLoop()` from the stop path**

Find `func run()` (or the `Task { ... }` that starts the audio pipelines). After the pipelines are kicked off, add:

```swift
startSummaryLoop()
```

Find where the session ends — either in `func stop()`, in `toggle()`, or in the `runTask` completion handler. After the pipelines are torn down, add:

```swift
stopSummaryLoop()
```

Also call `stopSummaryLoop()` from `clear()` if that method exists (so manually clearing the list also wipes the summary).

- [ ] **Step 4: Build**

```bash
LIBRARY_PATH="external/sherpa-onnx/lib" swift build -c debug 2>&1 | grep -E "error:|Build complete"
```

**If you see an error about `Sentence.createdAt`:** check `Types.swift` for the exact property name on `Sentence` that holds the wall-clock timestamp. CLAUDE.md says `Pipeline` sets `createdAt` and `endsAt` via `runStartedAt` — the field is `createdAt: Date`. If the field is named differently, update `runOneSummaryCycle` accordingly.

**If you see an actor-isolation error on `self?.lastSummary`:** the `runOneSummaryCycle` captures `lastSummary` in a `MainActor.run` block which is safe. If the compiler complains, add `@MainActor` annotation to both `lastSummary` and `transcriptSummary`, or move the capture into the `MainActor.run` closure.

Expected: `Build complete!`

- [ ] **Step 5: Commit**

```bash
git add Sources/LiveTranslate/Pipeline.swift
git commit -m "feat: add 60 s topic/summary loop to Pipeline using TopicSummarizer

Starts 15 s after session begin, runs every 60 s. Requires ≥ 3 sentences
from the last 5 min. Publishes TranscriptSummary on @MainActor; resets
on stop. Gracefully skips if Apple Intelligence is unavailable."
```

---

## Task 5: Display `transcriptSummary` in `TranscriptView`

**Files:**
- Modify: `Sources/LiveTranslate/TranscriptView.swift`

Read the current file structure first:
```bash
grep -n "var body\|private var\|compactBar\|fullBar\|VStack\|ScrollView" Sources/LiveTranslate/TranscriptView.swift | head -30
```

- [ ] **Step 1: Add `summaryBar` computed property**

Find any other `private var` view property (e.g. `compactBar` or `fullBar`) and add `summaryBar` nearby:

```swift
/// Shown below the control bar when the LLM has produced a summary.
/// Fades in/out via `.transition(.opacity)`.
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
        .padding(.vertical, 5)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial.opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .padding(.horizontal, 8)
        .transition(.opacity.combined(with: .move(edge: .top)))
    }
}
```

- [ ] **Step 2: Insert `summaryBar` into the body layout**

In `TranscriptView.body`, find the `VStack` that contains the control bar and the sentence `ScrollView`. Insert `summaryBar` between the bar and the `ScrollView`:

```swift
VStack(alignment: .leading, spacing: 0) {
    // ... existing bar (compactBar / fullBar) ...
    summaryBar   // ← add this
    // ... existing ScrollView / sentence list ...
}
.animation(.easeInOut(duration: 0.25), value: pipeline.transcriptSummary != nil)
```

If the outer `VStack` already has `.animation(...)`, add a second `.animation(.easeInOut(duration: 0.25), value: pipeline.transcriptSummary != nil)` modifier.

- [ ] **Step 3: Test the layout with a fixture**

In `Pipeline.loadDebugFixtures()`, temporarily add at the end:
```swift
transcriptSummary = TranscriptSummary(
    topic: "Software architecture discussion",
    summary: "The speakers discussed microservice decomposition strategies. They agreed on a domain-driven hexagonal approach for the new auth service."
)
```

Build, run (`open build/LiveTranslate.app`), press Cmd+Shift+D (Debug → Load fixture sentences). Confirm the summary bar appears with correct layout below the control bar. Remove the temporary line before committing.

- [ ] **Step 4: Build (without the test fixture line)**

```bash
LIBRARY_PATH="external/sherpa-onnx/lib" swift build -c debug 2>&1 | grep -E "error:|Build complete"
```

- [ ] **Step 5: Commit**

```bash
git add Sources/LiveTranslate/TranscriptView.swift
git commit -m "feat: show LLM topic/summary bar in TranscriptView

Appears below the control bar when transcriptSummary is non-nil.
Fades in with opacity+slide transition. Topic in bold caption,
summary in secondary caption2, max 3 lines."
```

---

## Task 6: Display `transcriptSummary` in `MenuBarView`

**Files:**
- Modify: `Sources/LiveTranslate/MenuBarView.swift`

- [ ] **Step 1: Add a summary section to `MenuBarView.body`**

In `MenuBarView.swift`, find the main `VStack` in `body`. It currently contains `compactBar` and `sentenceList`. Add the summary block between `compactBar` and `sentenceList`:

```swift
// After compactBar, before sentenceList:
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

Add `.animation(.easeInOut(duration: 0.25), value: pipeline.transcriptSummary != nil)` on the outer `VStack`.

- [ ] **Step 2: Build**

```bash
LIBRARY_PATH="external/sherpa-onnx/lib" swift build -c debug 2>&1 | grep -E "error:|Build complete"
```

- [ ] **Step 3: Commit**

```bash
git add Sources/LiveTranslate/MenuBarView.swift
git commit -m "feat: show topic/summary in MenuBarView popover"
```

---

## Task 7: Full build and smoke test

- [ ] **Step 1: Full release build**

```bash
LIVETRANSLATE_SIGN_IDENTITY=LiveTranslateDev ./build.sh
```

Expected final line: `✓ built build/LiveTranslate.app`

- [ ] **Step 2: Launch and watch the log**

```bash
open build/LiveTranslate.app
tail -f /tmp/livetranslate.log
```

- [ ] **Step 3: Start a recording session and wait**

Press Start (or Return in the menu-bar popover). Speak or play audio for 20+ seconds to generate at least 3 translated sentences.

After ~15 seconds you should see in the log:
```
TopicSummarizer: summary loop started
```

After the first generation completes (~15–30 s for model inference):
```
TopicSummarizer: topic=<whatever the model chose>
```

The summary bar should appear in the overlay and in the menu-bar popover.

- [ ] **Step 4: Commit the smoke test sign-off (no code changes — just push)**

```bash
git push
```

---

## Task 8: Update `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add `TopicSummarizer.swift` to the Files table**

Find the row for `TranscriptView.swift` and add a new row after it:

```markdown
| `TopicSummarizer.swift` | `actor` using Apple FoundationModels. `TopicSummaryOutput` is `@Generable` — constrained JSON decoding guarantees a well-formed `topic + summary` response. Creates a fresh `LanguageModelSession` per 60-second cycle (no stale conversation history). `isAvailable()` checks `SystemLanguageModel.default.availability` before starting. |
```

- [ ] **Step 2: Update the macOS deployment target note in the "How it's built" section**

Find the line: `macOS deployment target: .macOS(.v15)` and change it to:

```markdown
- macOS deployment target: `.macOS(.v26)` — bumped from v15 to enable the `FoundationModels` framework (Apple Intelligence on-device LLM).
```

- [ ] **Step 3: Add FoundationModels to Tools/SDKs**

Add before the `SwiftUI` line:

```markdown
- `FoundationModels` (`LanguageModelSession`, `@Generable`, `@Guide`) — on-device Apple Intelligence LLM for topic+summary generation; zero dependencies, model is built into macOS 26
```

- [ ] **Step 4: Update the Roadmap**

Add above `[ ] Speaker diarization`:

```markdown
- [x] On-device LLM topic+summary loop — Apple FoundationModels, every 60 s, shows topic label + 2-sentence summary in overlay and popover; gracefully skipped if Apple Intelligence unavailable
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for TopicSummarizer / FoundationModels

Deployment target note, Files table, Tools/SDKs, Roadmap."
```

---

## Self-Review

**Spec coverage:**

| Requirement | Task |
|---|---|
| Small on-device LLM | Task 3 — Apple FoundationModels, ~3B 2-bit quantized, built into macOS 26 |
| Every ~1 minute | Task 4 — 60 s sleep loop, 15 s initial delay |
| Last ~5 minutes of translations | Task 4 — `createdAt >= cutoff` filter |
| Topic label | Task 3 — `TopicSummaryOutput.topic` with `@Guide` description |
| ~2 sentence summary | Task 3 — `TopicSummaryOutput.summary` with `@Guide` description |
| Feed previous result back | Task 3 — `previous` in `buildPrompt`; Task 4 — `lastSummary` carries over |
| Display in overlay | Task 5 — `summaryBar` in `TranscriptView` |
| Display in popover | Task 6 — `MenuBarView` summary section |
| No fallback | Task 4 — `isAvailable()` check; logs and skips silently if not available; no llama.cpp or onnxruntime-genai path |
| Build integration | Task 1 — deployment target bump; no new downloads needed |
| Docs | Task 8 — `CLAUDE.md` |

**Placeholder scan:** No TBDs, TODOs, or "add error handling" phrases. All code blocks are complete.

**Type consistency:**
- `TranscriptSummary` defined in Task 2 → returned by `TopicSummarizer.summarize` in Task 3 → stored as `Pipeline.lastSummary` and `Pipeline.transcriptSummary` in Task 4 → read in Task 5+6.
- `TopicSummaryOutput` defined in Task 3 (local to `TopicSummarizer.swift`), never referenced outside.
- `TopicSummarizer.isAvailable()` → `Bool`, called in Task 4 `startSummaryLoop`.
- `runOneSummaryCycle(summarizer:)` takes `TopicSummarizer` (actor, non-optional) — only called after the guard in `startSummaryLoop`.
