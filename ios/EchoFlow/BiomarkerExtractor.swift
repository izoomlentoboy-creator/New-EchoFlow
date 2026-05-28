import Accelerate
import Foundation

/// Native port of `echoflow/features/acoustic.py`. Produces the ordered 39-dim
/// clinical biomarker vector the model expects.
///
/// IMPORTANT — verify parity on device: the spectral/MFCC maths must match the
/// Python reference within tolerance. Ship `artifacts/parity_refs.json` and run
/// `BiomarkerParityTests` (see tests/). The CNN branch is exact by construction
/// (the spectrogram is computed in-graph), so even small biomarker mismatches
/// degrade gracefully; still, aim for < 1e-2 relative error before clinical use.
enum BiomarkerExtractor {

    // Order MUST match ACOUSTIC_NAMES in acoustic.py.
    static let names = [
        "f0_mean","f0_std","f0_min","f0_max","f0_range",
        "jitter_local","jitter_ddp","shimmer_local","shimmer_dda",
        "hnr_mean","hnr_std","cpp_mean","cpp_std","gne","spectral_tilt",
        "voiced_fraction","zcr_mean","zcr_std","centroid_mean","centroid_std",
        "bandwidth_mean","bandwidth_std","rolloff_mean","rolloff_std",
        "flatness_mean","flatness_std",
    ] + (1...13).map { "mfcc\($0)_mean" }

    static var voicedFractionIndex: Int { names.firstIndex(of: "voiced_fraction")! }

    static func compute(waveform: [Float], sampleRate sr: Int,
                        nFFT: Int = 1024, hop: Int = 256) -> [Float] {
        var x = waveform
        let peak = x.map { abs($0) }.max() ?? 0
        if peak > 0 { x = x.map { $0 / peak } }

        // ---- F0 / perturbation / HNR (time-domain autocorrelation) ----
        let (f0, voiced) = estimateF0(x, sr: sr)
        let vf0 = zip(f0, voiced).filter { $0.1 }.map { $0.0 }
        let f0Mean = mean(vf0), f0Std = std(vf0)
        let f0Min = vf0.min() ?? 0, f0Max = vf0.max() ?? 0
        let voicedFraction = voiced.isEmpty ? 0 :
            Float(voiced.filter { $0 }.count) / Float(voiced.count)
        let (jitL, jitDdp, shimL, shimDda) = jitterShimmer(x, sr: sr, f0: f0, voiced: voiced)
        let (hnrMean, hnrStd) = hnr(x, sr: sr)
        let gneVal = gne(x, sr: sr)

        // ---- shared magnitude STFT for spectral descriptors ----
        let spec = stftMagnitude(x, nFFT: nFFT, hop: hop)      // [frames][bins]
        let (cppMean, cppStd) = cpp(spec, sr: sr, nFFT: nFFT)
        let tilt = spectralTilt(spec, sr: sr)
        let centroid = perFrame(spec) { specCentroid($0, sr: sr, nFFT: nFFT) }
        let bandwidth = perFrameWithCentroid(spec, sr: sr, nFFT: nFFT)
        let rolloff = perFrame(spec) { specRolloff($0, sr: sr, nFFT: nFFT) }
        let flatness = perFrame(spec) { specFlatness($0) }
        let mfcc = mfccMeans(spec, sr: sr, nFFT: nFFT, nMels: 40, nMfcc: 13)
        let (zcrMean, zcrStd) = zcr(x, frame: Int(0.040*Float(sr)), hop: Int(0.010*Float(sr)))

        var out: [Float] = [
            f0Mean, f0Std, f0Min, f0Max, f0Max - f0Min,
            jitL, jitDdp, shimL, shimDda,
            hnrMean, hnrStd, cppMean, cppStd, gneVal, tilt,
            voicedFraction, zcrMean, zcrStd,
            mean(centroid), std(centroid), mean(bandwidth), std(bandwidth),
            mean(rolloff), std(rolloff), mean(flatness), std(flatness),
        ]
        out.append(contentsOf: mfcc)
        return out.map { $0.isFinite ? $0 : 0 }
    }

    // MARK: - F0 (autocorrelation)
    static func estimateF0(_ x: [Float], sr: Int, fmin: Float = 60, fmax: Float = 400,
                           frameMs: Float = 40, hopMs: Float = 10) -> ([Float], [Bool]) {
        let frame = Int(Float(sr)*frameMs/1000), hop = Int(Float(sr)*hopMs/1000)
        if x.count < frame { return ([], []) }
        let n = 1 + (x.count - frame)/hop
        let win = hann(frame)
        let minLag = max(1, Int(Float(sr)/fmax)), maxLag = min(frame-1, Int(Float(sr)/fmin))
        var f0 = [Float](repeating: 0, count: n), voiced = [Bool](repeating: false, count: n)
        for i in 0..<n {
            var seg = (0..<frame).map { x[i*hop+$0]*win[$0] }
            let m = mean(seg); seg = seg.map { $0 - m }
            if std(seg) < 1e-4 { continue }
            let ac = autocorr(seg)
            if ac[0] <= 0 { continue }
            let norm = ac.map { $0/ac[0] }
            var best: Float = -2; var lag = 0
            for l in minLag..<min(maxLag, norm.count) where norm[l] > best { best = norm[l]; lag = l }
            if best > 0.45 && lag > 0 { f0[i] = Float(sr)/Float(lag); voiced[i] = true }
        }
        return (f0, voiced)
    }

    static func jitterShimmer(_ x: [Float], sr: Int, f0: [Float], voiced: [Bool])
        -> (Float, Float, Float, Float) {
        let vf0 = zip(f0, voiced).filter { $0.1 }.map { $0.0 }
        if vf0.count < 3 { return (0,0,0,0) }
        let periods = vf0.map { Float(sr) / max($0, 1e-3) }
        let dperiod = zip(periods.dropFirst(), periods).map { abs($0 - $1) }
        let meanP = mean(periods) + 1e-8
        let jitL = mean(dperiod) / meanP
        let dd = zip(dperiod.dropFirst(), dperiod).map { abs($0 - $1) }
        let jitDdp = dd.isEmpty ? 0 : mean(dd)/meanP
        // shimmer from per-cycle peak amplitude
        let hop = Int(0.010 * Float(sr))
        var amps = [Float]()
        var k = 0
        for (i, v) in voiced.enumerated() where v {
            let center = i*hop
            let half = max(1, Int((k < periods.count ? periods[k] : meanP)/2)); k += 1
            let lo = max(0, center-half), hi = min(x.count, center+half)
            if hi > lo { amps.append((lo..<hi).map { abs(x[$0]) }.max() ?? 0) }
        }
        if amps.count < 3 || mean(amps) < 1e-8 { return (jitL, jitDdp, 0, 0) }
        let meanA = mean(amps) + 1e-8
        let da = zip(amps.dropFirst(), amps).map { abs($0 - $1) }
        let shimL = mean(da)/meanA
        let dda = zip(da.dropFirst(), da).map { abs($0 - $1) }
        return (jitL, jitDdp, shimL, dda.isEmpty ? 0 : mean(dda)/meanA)
    }

    static func hnr(_ x: [Float], sr: Int, frameMs: Float = 40, hopMs: Float = 10)
        -> (Float, Float) {
        let frame = Int(Float(sr)*frameMs/1000), hop = Int(Float(sr)*hopMs/1000)
        if x.count < frame { return (0,0) }
        let n = 1 + (x.count - frame)/hop, win = hann(frame)
        let minLag = max(1, Int(Float(sr)/400)), maxLag = min(frame-1, Int(Float(sr)/60))
        var vals = [Float]()
        for i in 0..<n {
            var seg = (0..<frame).map { x[i*hop+$0]*win[$0] }
            let m = mean(seg); seg = seg.map { $0 - m }
            if std(seg) < 1e-4 { continue }
            let ac = autocorr(seg); if ac[0] <= 0 { continue }
            let norm = ac.map { $0/ac[0] }
            var peak: Float = 0
            for l in minLag..<min(maxLag, norm.count) { peak = max(peak, norm[l]) }
            let r = min(max(peak, 1e-6), 0.999999)
            vals.append(10*log10(r/(1-r)))
        }
        return (mean(vals), std(vals))
    }

    // MARK: - CPP (cepstral peak prominence)
    static func cpp(_ spec: [[Float]], sr: Int, nFFT: Int,
                    fmin: Float = 60, fmax: Float = 400) -> (Float, Float) {
        let qmin = 1.0/fmax, qmax = 1.0/fmin
        var vals = [Float]()
        for frameMag in spec {
            let logMag = frameMag.map { logf($0 + 1e-6) }
            let cep = irfft(logMag, n: nFFT)               // real cepstrum
            var qs = [Float](); var cs = [Float]()
            for i in 0..<cep.count {
                let q = Float(i)/Float(sr)
                if q >= qmin && q <= qmax { qs.append(q); cs.append(cep[i]) }
            }
            if cs.count < 2 { continue }
            let (a, b) = linregress(qs, cs)                 // baseline
            var maxDev: Float = -.greatestFiniteMagnitude
            for j in 0..<cs.count { maxDev = max(maxDev, cs[j] - (a*qs[j] + b)) }
            vals.append(maxDev)
        }
        return (mean(vals), std(vals))
    }

    // MARK: - GNE (simplified glottal-to-noise excitation)
    static func gne(_ x: [Float], sr: Int, nBands: Int = 4) -> Float {
        let nyq = Float(sr)/2
        let edges = (0...nBands).map { 300 + (min(5000, nyq*0.95)-300)*Float($0)/Float(nBands) }
        var envs = [[Float]]()
        for i in 0..<nBands {
            let band = bandpassFFT(x, low: edges[i], high: edges[i+1], sr: sr)
            var env = hilbertEnvelope(band)
            let m = mean(env); env = env.map { $0 - m }
            envs.append(env)
        }
        var best: Float = 0
        for i in 0..<envs.count { for j in (i+1)..<envs.count {
            let a = envs[i], b = envs[j]
            var dot: Float = 0; vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(min(a.count,b.count)))
            let denom = std(a)*std(b)*Float(a.count) + 1e-9
            best = max(best, dot/denom)
        }}
        return min(max(best, 0), 1)
    }

    // MARK: - spectral descriptors
    static func spectralTilt(_ spec: [[Float]], sr: Int) -> Float {
        guard let bins = spec.first?.count else { return 0 }
        var powerMean = [Float](repeating: 0, count: bins)
        for f in spec { for b in 0..<bins { powerMean[b] += f[b]*f[b] } }
        for b in 0..<bins { powerMean[b] /= Float(spec.count) }
        var fs = [Float](); var ys = [Float]()
        for b in 0..<bins {
            let freq = Float(b)*Float(sr)/Float((bins-1)*2)
            if freq > 50 { fs.append(freq/1000); ys.append(10*log10(powerMean[b] + 1e-10)) }
        }
        if fs.count < 2 { return 0 }
        return linregress(fs, ys).0
    }

    static func specCentroid(_ mag: [Float], sr: Int, nFFT: Int) -> Float {
        var num: Float = 0, den: Float = 0
        for (b, m) in mag.enumerated() {
            let f = Float(b)*Float(sr)/Float(nFFT); num += f*m; den += m
        }
        return den > 0 ? num/den : 0
    }
    static func perFrameWithCentroid(_ spec: [[Float]], sr: Int, nFFT: Int) -> [Float] {
        spec.map { mag in
            let c = specCentroid(mag, sr: sr, nFFT: nFFT)
            var num: Float = 0, den: Float = 0
            for (b, m) in mag.enumerated() {
                let f = Float(b)*Float(sr)/Float(nFFT); num += (f-c)*(f-c)*m; den += m
            }
            return den > 0 ? (num/den).squareRoot() : 0
        }
    }
    static func specRolloff(_ mag: [Float], sr: Int, nFFT: Int, pct: Float = 0.85) -> Float {
        let total = mag.reduce(0, +); if total <= 0 { return 0 }
        var cum: Float = 0
        for (b, m) in mag.enumerated() { cum += m; if cum >= pct*total {
            return Float(b)*Float(sr)/Float(nFFT) } }
        return Float(mag.count-1)*Float(sr)/Float(nFFT)
    }
    static func specFlatness(_ mag: [Float]) -> Float {
        let p = mag.map { $0*$0 }
        let gm = exp(mean(p.map { logf($0 + 1e-10) }))
        let am = mean(p) + 1e-10
        return gm/am
    }

    static func mfccMeans(_ spec: [[Float]], sr: Int, nFFT: Int, nMels: Int, nMfcc: Int) -> [Float] {
        let fb = melFilterbank(sr: sr, nFFT: nFFT, nMels: nMels)
        var accum = [Float](repeating: 0, count: nMfcc)
        for mag in spec {
            let power = mag.map { $0*$0 }
            var melE = [Float](repeating: 0, count: nMels)
            for m in 0..<nMels { var s: Float = 0
                for b in 0..<power.count { s += fb[m][b]*power[b] }; melE[m] = s }
            let db = melE.map { 10*log10(max($0, 1e-10)) }      // power_to_db, ref=1
            let c = dctII(db, n: nMfcc)
            for i in 0..<nMfcc { accum[i] += c[i] }
        }
        return accum.map { $0/Float(max(spec.count,1)) }
    }

    static func zcr(_ x: [Float], frame: Int, hop: Int) -> (Float, Float) {
        if x.count < frame { return (0,0) }
        let n = 1 + (x.count - frame)/hop
        var vals = [Float]()
        for i in 0..<n {
            var c: Float = 0
            for j in 1..<frame {
                if (x[i*hop+j] >= 0) != (x[i*hop+j-1] >= 0) { c += 1 }
            }
            vals.append(c/Float(frame))
        }
        return (mean(vals), std(vals))
    }
}
