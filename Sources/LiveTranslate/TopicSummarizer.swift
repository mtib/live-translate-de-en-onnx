import Foundation
import FoundationModels

// MARK: - Errors

enum TopicSummarizerError: Error {
    case modelUnavailable(String)
    case generationFailed(String)
}

// MARK: - Actor

/// Uses Apple FoundationModels (`LanguageModelSession`) to produce a rolling
/// topic label + summary from the last ~5 minutes of translated sentences.
/// Requires Apple Intelligence to be enabled on the device (macOS 26+).
///
/// Structured output via `@Generable` requires the FoundationModelsMacros
/// compiler plugin which is only available when building through Xcode. This
/// implementation falls back to free-text generation and parses
/// "Topic: ..." / "Summary: ..." lines from the raw response instead.
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
            slightly refined version rather than inventing a new one. \
            Always reply in exactly two lines: \
            "Topic: <short phrase>" and "Summary: <two sentences>". \
            Do not include any other text.
            """
        )

        let result = try await session.respond(to: prompt)
        let rawText = result.content

        guard let parsed = parseResponse(rawText) else {
            Log.line("TopicSummarizer: failed to parse response: \(rawText)")
            throw TopicSummarizerError.generationFailed("Could not parse Topic/Summary lines from: \(rawText)")
        }

        return parsed
    }

    // MARK: - Response parsing

    /// Extracts `topic` and `summary` from lines beginning with "Topic:" and "Summary:".
    /// Returns nil if either field is missing or empty.
    private func parseResponse(_ text: String) -> TranscriptSummary? {
        var topic: String?
        var summary: String?

        for line in text.components(separatedBy: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.lowercased().hasPrefix("topic:") {
                let value = String(trimmed.dropFirst("topic:".count)).trimmingCharacters(in: .whitespaces)
                if !value.isEmpty { topic = value }
            } else if trimmed.lowercased().hasPrefix("summary:") {
                let value = String(trimmed.dropFirst("summary:".count)).trimmingCharacters(in: .whitespaces)
                if !value.isEmpty { summary = value }
            }
        }

        guard let t = topic, let s = summary else { return nil }
        return TranscriptSummary(topic: t, summary: s)
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
