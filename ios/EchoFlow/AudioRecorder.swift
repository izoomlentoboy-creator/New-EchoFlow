import AVFoundation

/// Captures microphone audio and turns it into the fixed-length, normalized
/// mono waveform the model expects.  The preprocessing here MUST mirror
/// `echoflow/audio/preprocess.py` exactly (trim → RMS-normalize → fix length).
///
/// Usage:
///   let rec = AudioRecorder()
///   try rec.start()
///   ... user sustains the vowel /a/ ...
///   let waveform = rec.stopAndProcess()   // [Float] of length clipSamples
final class AudioRecorder {

    // Keep in sync with echoflow.config.AudioConfig
    static let sampleRate: Double = 16_000
    static let clipSeconds: Double = 3.0
    static var clipSamples: Int { Int(sampleRate * clipSeconds) }   // 48_000

    private let engine = AVAudioEngine()
    private var captured = [Float]()
    private let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                             sampleRate: AudioRecorder.sampleRate,
                                             channels: 1, interleaved: false)!

    // MARK: - Capture

    func start() throws {
        captured.removeAll()
        let input = engine.inputNode
        let inFormat = input.outputFormat(forBus: 0)
        guard let converter = AVAudioConverter(from: inFormat, to: targetFormat) else {
            throw NSError(domain: "EchoFlow", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "converter failed"])
        }
        input.installTap(onBus: 0, bufferSize: 4096, format: inFormat) { [weak self] buf, _ in
            guard let self else { return }
            let outCap = AVAudioFrameCount(
                Double(buf.frameLength) * AudioRecorder.sampleRate / inFormat.sampleRate + 16)
            guard let out = AVAudioPCMBuffer(pcmFormat: self.targetFormat,
                                             frameCapacity: outCap) else { return }
            var err: NSError?
            converter.convert(to: out, error: &err) { _, status in
                status.pointee = .haveData
                return buf
            }
            if let ch = out.floatChannelData {
                self.captured.append(contentsOf:
                    UnsafeBufferPointer(start: ch[0], count: Int(out.frameLength)))
            }
        }
        try engine.start()
    }

    /// Stop capture and return the preprocessed fixed-length waveform.
    func stopAndProcess() -> [Float] {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        return AudioRecorder.preprocess(captured)
    }

    // MARK: - Preprocessing (mirror of preprocess.py)

    static func preprocess(_ wav: [Float]) -> [Float] {
        var x = trimSilence(wav, sr: Int(sampleRate))
        x = rmsNormalize(x)
        x = fixLength(x, clipSamples)
        return x
    }

    /// Energy-based VAD keeping the voiced span (mirror of `trim_silence`).
    static func trimSilence(_ wav: [Float], sr: Int, topDb: Float = 30,
                            frameMs: Float = 25, hopMs: Float = 10) -> [Float] {
        let frame = max(1, Int(Float(sr) * frameMs / 1000))
        let hop = max(1, Int(Float(sr) * hopMs / 1000))
        if wav.count < frame { return wav }
        let n = 1 + (wav.count - frame) / hop
        var energies = [Float](repeating: 0, count: n)
        for i in 0..<n {
            var s: Float = 0
            for j in 0..<frame { let v = wav[i*hop+j]; s += v*v }
            energies[i] = (s / Float(frame) + 1e-10).squareRoot()
        }
        let ref = energies.max() ?? 0
        if ref <= 0 { return wav }
        var voiced = [Bool](repeating: false, count: n)
        for i in 0..<n {
            let db = 20 * log10(energies[i] / ref + 1e-10)
            voiced[i] = db > -topDb
        }
        guard let first = voiced.firstIndex(of: true),
              let last = voiced.lastIndex(of: true) else { return wav }
        let start = first * hop
        let end = min(wav.count, last * hop + frame)
        return Array(wav[start..<end])
    }

    /// RMS loudness normalization to -20 dBFS with peak guard.
    static func rmsNormalize(_ wav: [Float], targetDbfs: Float = -20) -> [Float] {
        if wav.isEmpty { return wav }
        var ss: Float = 0
        for v in wav { ss += v*v }
        let rms = (ss / Float(wav.count) + 1e-8).squareRoot()
        let targetRms = pow(10, targetDbfs / 20)
        let gain = targetRms / (rms + 1e-8)
        var out = wav.map { $0 * gain }
        let peak = out.map { abs($0) }.max() ?? 0
        if peak > 1 { out = out.map { $0 / peak } }
        return out
    }

    /// Center-crop or symmetric zero-pad to `target` samples.
    static func fixLength(_ wav: [Float], _ target: Int) -> [Float] {
        if wav.count == target { return wav }
        if wav.count > target {
            let start = (wav.count - target) / 2
            return Array(wav[start..<start+target])
        }
        let pad = target - wav.count
        let left = pad / 2
        return [Float](repeating: 0, count: left) + wav
             + [Float](repeating: 0, count: pad - left)
    }
}
