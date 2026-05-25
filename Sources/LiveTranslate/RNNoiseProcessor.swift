import Foundation
import AVFoundation
import Accelerate
import CRNNoise

/// Thin Swift wrapper around RNNoise (xiph, v0.1.1). The C API processes
/// **480-sample frames at 48 kHz mono** and takes/returns Float32 samples
/// scaled to the int16 range (±32768), not the normalised ±1 range we
/// use everywhere else. This wrapper handles:
///
///   - Allocation + lifetime of the `DenoiseState`.
///   - Buffering input across calls so partial frames don't cause skips
///     — the user feeds in whatever-sized buffers, this class emits
///     480-sample frames steadily.
///   - The int16-scale ⇄ unit-scale conversion at the boundary, using
///     `vDSP_vsmul` (SIMD).
///
/// Frame size of 480 at 48 kHz = 10 ms of latency. That's the minimum
/// algorithmic latency for RNNoise.
///
/// **Hot-path discipline.** Scratch frames and the input/output queues
/// are pre-allocated and reused across every call. No allocations or
/// O(n) shifts on the per-buffer path; the queues use a head/tail index
/// pair and rebase only when fully drained.
final class RNNoiseProcessor {

    /// Samples per RNNoise frame at 48 kHz (10 ms). Public so callers
    /// can size their own buffers if they want.
    static let frameSize: Int = 480

    /// Scale factor between our normalised Float32 audio and the int16
    /// representation RNNoise expects. 32768 = 1 << 15.
    private static let int16Scale: Float = 32768.0
    private static let int16ScaleInv: Float = 1.0 / 32768.0

    /// Opaque `DenoiseState*` from the C side.
    private var state: OpaquePointer?

    /// Scratch frame buffers, allocated once and reused.
    private var inFrame  = ContiguousArray<Float>(repeating: 0, count: frameSize)
    private var outFrame = ContiguousArray<Float>(repeating: 0, count: frameSize)

    /// Input queue: int16-scaled samples awaiting processing.
    /// Uses a head index instead of `removeFirst` to avoid O(n) shifts.
    /// Rebased to 0 whenever the head catches up to the tail (steady
    /// state: head == tail at the end of each `feed()` call once all
    /// full frames are drained, so the queue stays small).
    private var inputQueue  = ContiguousArray<Float>()
    private var inputHead: Int = 0

    /// Output queue: normalised denoised samples awaiting `drain()`.
    /// Same head-index discipline as the input queue.
    private var outputQueue = ContiguousArray<Float>()
    private var outputHead: Int = 0

    init() {
        state = rnnoise_create(nil)
        // Reserve enough headroom that typical feeds (≤ 1024 samples)
        // never reallocate during steady-state operation.
        inputQueue.reserveCapacity(Self.frameSize * 4)
        outputQueue.reserveCapacity(Self.frameSize * 4)
    }

    deinit {
        if let state {
            rnnoise_destroy(state)
        }
    }

    /// Feed normalised Float32 samples (range ±1) in any quantity. The
    /// processor buffers internally and runs RNNoise on full 480-sample
    /// frames at 48 kHz. Output is appended to the internal queue and
    /// returned via `drain(into:)`.
    func feed(samples: UnsafePointer<Float>, count: Int) {
        guard count > 0, let state else { return }

        // 1. Append scaled (×32768) to the input queue. We do this in a
        //    single vDSP_vsmul pass into reserved tail capacity to avoid
        //    a per-sample multiply loop.
        let oldEnd = inputQueue.count
        inputQueue.append(contentsOf: repeatElement(0, count: count))
        var scale = Self.int16Scale
        inputQueue.withUnsafeMutableBufferPointer { qPtr in
            vDSP_vsmul(samples, 1,
                       &scale,
                       qPtr.baseAddress!.advanced(by: oldEnd), 1,
                       vDSP_Length(count))
        }

        // 2. Process every full frame available, reading from inputHead.
        while inputQueue.count - inputHead >= Self.frameSize {
            inFrame.withUnsafeMutableBufferPointer { inP in
                outFrame.withUnsafeMutableBufferPointer { outP in
                    inputQueue.withUnsafeBufferPointer { qPtr in
                        memcpy(inP.baseAddress!,
                               qPtr.baseAddress!.advanced(by: inputHead),
                               Self.frameSize * MemoryLayout<Float>.size)
                    }
                    rnnoise_process_frame(state, outP.baseAddress, inP.baseAddress)
                }
            }
            inputHead += Self.frameSize

            // 3. Append scaled-back (÷32768) output samples to the queue
            //    with one vDSP_vsmul into reserved tail capacity.
            let outOld = outputQueue.count
            outputQueue.append(contentsOf: repeatElement(0, count: Self.frameSize))
            var inv = Self.int16ScaleInv
            outFrame.withUnsafeBufferPointer { srcPtr in
                outputQueue.withUnsafeMutableBufferPointer { dstPtr in
                    vDSP_vsmul(srcPtr.baseAddress!, 1,
                               &inv,
                               dstPtr.baseAddress!.advanced(by: outOld), 1,
                               vDSP_Length(Self.frameSize))
                }
            }
        }

        // 4. If the input queue has been fully consumed, reset both
        //    head and tail to 0 so we never grow unbounded.
        if inputHead >= inputQueue.count {
            inputQueue.removeAll(keepingCapacity: true)
            inputHead = 0
        }
    }

    /// Pull up to `count` denoised samples into `dst`. Returns how many
    /// were actually written. The remainder of `dst` (if any) is left
    /// untouched; the caller should fill the gap with silence or wait
    /// for more input.
    func drain(into dst: UnsafeMutablePointer<Float>, count: Int) -> Int {
        let available = outputQueue.count - outputHead
        let take = min(count, available)
        guard take > 0 else { return 0 }
        outputQueue.withUnsafeBufferPointer { qPtr in
            memcpy(dst,
                   qPtr.baseAddress!.advanced(by: outputHead),
                   take * MemoryLayout<Float>.size)
        }
        outputHead += take
        if outputHead >= outputQueue.count {
            outputQueue.removeAll(keepingCapacity: true)
            outputHead = 0
        }
        return take
    }

    /// Drop any buffered state. Call between recognition sessions if
    /// you want the denoiser to forget recent context.
    func reset() {
        inputQueue.removeAll(keepingCapacity: true)
        outputQueue.removeAll(keepingCapacity: true)
        inputHead = 0
        outputHead = 0
    }
}
