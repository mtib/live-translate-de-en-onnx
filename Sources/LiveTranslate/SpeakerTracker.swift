import Foundation
import CSherpaOnnx

/// Assigns stable "Speaker N" IDs to turns of speech using campplus
/// embedding extraction + cosine-similarity clustering.
///
/// One tracker per audio stream — mic and system audio are independent
/// conversations. Thread-safe (NSLock around mutable state) because the
/// transcriber accumulator and worker both touch it from background tasks.
///
/// Identity model: the first speaker gets ID 1. Subsequent turns get the
/// ID of the closest known speaker whose cosine similarity exceeds
/// `threshold`; if no known speaker is close enough, a new ID is assigned.
/// This is "online" clustering — no retroactive label correction once a
/// turn is assigned.
final class SpeakerTracker {

    /// Cosine-similarity threshold above which a new embedding is
    /// considered the same speaker. 0.5 is conservative (avoids spurious
    /// merges across very different voices); tune up toward 0.65 if the
    /// same speaker keeps getting split.
    static let threshold: Float = 0.50

    private let extractor: OpaquePointer
    private let dim: Int

    // Known speakers: array of (id, prototype embedding). The prototype
    // is the first embedding seen for that speaker — no running average
    // to keep the implementation simple.
    private var speakers: [(id: Int, embedding: [Float])] = []
    private var nextID: Int = 1
    private let lock = NSLock()

    /// Returns nil when the model file is missing from the bundle.
    init?() {
        guard let modelURL = Bundle.main.url(forResource: ModelConfig.speakerModel,
                                              withExtension: nil) else {
            Log.line("SpeakerTracker: model not found in bundle (\(ModelConfig.speakerModel))")
            return nil
        }

        // Build the config with C-string lifetimes pinned for the duration
        // of the SherpaOnnx call by nesting `withCString` blocks. The
        // result is captured into a local before assigning stored properties.
        let modelPath = modelURL.path
        let provider  = ModelConfig.provider

        var cfg = SherpaOnnxSpeakerEmbeddingExtractorConfig()
        cfg.num_threads = 2
        cfg.debug       = 0

        guard let ext = modelPath.withCString({ path -> OpaquePointer? in
            cfg.model = path
            return provider.withCString { prov -> OpaquePointer? in
                cfg.provider = prov
                return SherpaOnnxCreateSpeakerEmbeddingExtractor(&cfg)
            }
        }) else {
            Log.line("SpeakerTracker: SherpaOnnxCreateSpeakerEmbeddingExtractor failed")
            return nil
        }

        extractor = ext
        dim = Int(SherpaOnnxSpeakerEmbeddingExtractorDim(ext))
        Log.line("SpeakerTracker: loaded campplus embedding dim=\(dim)")
    }

    deinit {
        SherpaOnnxDestroySpeakerEmbeddingExtractor(extractor)
    }

    /// Reset state between sessions (e.g. user presses Stop then Start).
    func reset() {
        lock.withLock {
            speakers.removeAll()
            nextID = 1
        }
    }

    /// Given a vector of 16 kHz Float32 PCM samples for a completed
    /// speaker turn, return a "Speaker N" label (or "Speaker ?" on error).
    ///
    /// This is synchronous and may take a few milliseconds. Call from a
    /// background task (the transcriber worker already runs off-MainActor).
    func label(for pcm16k: [Float]) -> String {
        guard !pcm16k.isEmpty else { return "Speaker ?" }

        // Create a fresh stream, push all samples at once, check readiness.
        guard let stream = SherpaOnnxSpeakerEmbeddingExtractorCreateStream(extractor) else {
            Log.line("SpeakerTracker: failed to create embedding stream")
            return "Speaker ?"
        }
        defer { SherpaOnnxDestroyOnlineStream(stream) }

        pcm16k.withUnsafeBufferPointer { buf in
            SherpaOnnxOnlineStreamAcceptWaveform(stream, 16_000,
                                                 buf.baseAddress, Int32(buf.count))
        }
        SherpaOnnxOnlineStreamInputFinished(stream)

        guard SherpaOnnxSpeakerEmbeddingExtractorIsReady(extractor, stream) != 0 else {
            Log.line("SpeakerTracker: embedding extractor not ready")
            return "Speaker ?"
        }

        guard let rawEmb = SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(extractor, stream) else {
            Log.line("SpeakerTracker: embedding computation returned nil")
            return "Speaker ?"
        }
        defer { SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(rawEmb) }

        let embedding = Array(UnsafeBufferPointer(start: rawEmb, count: dim))

        return lock.withLock {
            // Find the closest known speaker.
            var bestID: Int? = nil
            var bestSim: Float = -1
            for (id, proto) in speakers {
                let sim = cosineSimilarity(embedding, proto)
                if sim > bestSim {
                    bestSim = sim
                    bestID = id
                }
            }

            if let id = bestID, bestSim >= Self.threshold {
                return "Speaker \(id)"
            } else {
                let id = nextID
                nextID += 1
                speakers.append((id: id, embedding: embedding))
                Log.line("SpeakerTracker: new speaker \(id) (closest sim=\(String(format: "%.3f", bestSim)))")
                return "Speaker \(id)"
            }
        }
    }

    // MARK: - Cosine similarity

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        for i in 0..<a.count {
            dot  += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        let denom = sqrt(normA) * sqrt(normB)
        guard denom > 0 else { return 0 }
        return dot / denom
    }
}
