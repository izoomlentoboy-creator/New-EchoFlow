import Accelerate
import Foundation

// MARK: - basic stats
func mean(_ x: [Float]) -> Float { x.isEmpty ? 0 : x.reduce(0,+)/Float(x.count) }
func std(_ x: [Float]) -> Float {           // population std (numpy ddof=0)
    if x.isEmpty { return 0 }
    let m = mean(x); return (x.reduce(0) { $0 + ($1-m)*($1-m) }/Float(x.count)).squareRoot()
}
func hann(_ n: Int) -> [Float] {            // matches numpy.hanning
    if n <= 1 { return [Float](repeating: 1, count: max(n,0)) }
    return (0..<n).map { 0.5 - 0.5*cos(2*Float.pi*Float($0)/Float(n-1)) }
}
func linregress(_ x: [Float], _ y: [Float]) -> (Float, Float) {   // slope, intercept
    let n = Float(x.count); if n < 2 { return (0,0) }
    let sx = x.reduce(0,+), sy = y.reduce(0,+)
    let sxx = zip(x,x).reduce(0) { $0+$1.0*$1.1 }
    let sxy = zip(x,y).reduce(0) { $0+$1.0*$1.1 }
    let denom = n*sxx - sx*sx; if abs(denom) < 1e-12 { return (0, sy/n) }
    let slope = (n*sxy - sx*sy)/denom
    return (slope, (sy - slope*sx)/n)
}
func perFrame(_ spec: [[Float]], _ f: ([Float]) -> Float) -> [Float] { spec.map(f) }

// MARK: - pure-Swift complex FFT (iterative radix-2 Cooley-Tukey)
/// Kept dependency-free and packing-bug-free on purpose (this code can only be
/// validated on device against the Python reference, so favour clarity).
func fft(_ re: inout [Float], _ im: inout [Float], inverse: Bool) {
    let n = re.count
    var j = 0
    for i in 1..<n {
        var bit = n >> 1
        while j & bit != 0 { j ^= bit; bit >>= 1 }
        j |= bit
        if i < j { re.swapAt(i,j); im.swapAt(i,j) }
    }
    var len = 2
    while len <= n {
        let ang = 2*Float.pi/Float(len) * (inverse ? 1 : -1)
        let wr = cos(ang), wi = sin(ang)
        var i = 0
        while i < n {
            var curR: Float = 1, curI: Float = 0
            for k in 0..<(len/2) {
                let a = i+k, b = i+k+len/2
                let tr = re[b]*curR - im[b]*curI
                let ti = re[b]*curI + im[b]*curR
                re[b] = re[a]-tr; im[b] = im[a]-ti
                re[a] += tr;      im[a] += ti
                let nR = curR*wr - curI*wi; curI = curR*wi + curI*wr; curR = nR
            }
            i += len
        }
        len <<= 1
    }
    if inverse { for i in 0..<n { re[i] /= Float(n); im[i] /= Float(n) } }
}

private func nextPow2(_ n: Int) -> Int { var p = 1; while p < n { p <<= 1 }; return p }

/// Magnitude of the one-sided real FFT of a single (already windowed-or-not) frame.
func rfftMagnitude(_ frame: [Float], nFFT: Int) -> [Float] {
    var re = frame; if re.count < nFFT { re += [Float](repeating: 0, count: nFFT-re.count) }
    re = Array(re.prefix(nFFT))
    var im = [Float](repeating: 0, count: nFFT)
    fft(&re, &im, inverse: false)
    let bins = nFFT/2 + 1
    return (0..<bins).map { (re[$0]*re[$0] + im[$0]*im[$0]).squareRoot() }
}

/// Inverse real FFT from a one-sided log/real spectrum (mirror of np.fft.irfft).
func irfft(_ spectrum: [Float], n: Int) -> [Float] {
    var re = [Float](repeating: 0, count: n), im = [Float](repeating: 0, count: n)
    let bins = spectrum.count
    for i in 0..<bins { re[i] = spectrum[i] }
    for i in 1..<(n - bins + 1) { re[n-i] = spectrum[i] }   // Hermitian symmetry
    fft(&re, &im, inverse: true)
    return re
}

// MARK: - STFT (mirror of librosa.stft: center pad reflect, hann, win=n_fft)
func stftMagnitude(_ x: [Float], nFFT: Int, hop: Int) -> [[Float]] {
    let pad = nFFT/2
    var padded = reflectPad(x, pad)
    let win = hann(nFFT)
    var frames = [[Float]]()
    if padded.count < nFFT { padded += [Float](repeating: 0, count: nFFT - padded.count) }
    var start = 0
    while start + nFFT <= padded.count {
        let seg = (0..<nFFT).map { padded[start+$0]*win[$0] }
        frames.append(rfftMagnitude(seg, nFFT: nFFT))
        start += hop
    }
    return frames
}
private func reflectPad(_ x: [Float], _ p: Int) -> [Float] {
    if x.count <= 1 { return x }
    let left = (1...p).map { x[min($0, x.count-1)] }.reversed()
    let right = (1...p).map { x[max(x.count-1-$0, 0)] }
    return Array(left) + x + Array(right)
}

// MARK: - autocorrelation (np.correlate(seg,seg,'full')[N-1:], lags 0..N-1)
func autocorr(_ seg: [Float]) -> [Float] {
    let n = seg.count, m = nextPow2(2*n)
    var re = seg + [Float](repeating: 0, count: m-n)
    var im = [Float](repeating: 0, count: m)
    fft(&re, &im, inverse: false)
    for i in 0..<m { re[i] = re[i]*re[i] + im[i]*im[i]; im[i] = 0 }   // power spectrum
    fft(&re, &im, inverse: true)
    return Array(re.prefix(n))
}

// MARK: - band-pass via FFT (zero out-of-band bins) + Hilbert envelope
func bandpassFFT(_ x: [Float], low: Float, high: Float, sr: Int) -> [Float] {
    let m = nextPow2(x.count)
    var re = x + [Float](repeating: 0, count: m-x.count)
    var im = [Float](repeating: 0, count: m)
    fft(&re, &im, inverse: false)
    for k in 0..<m {
        let f = Float(k <= m/2 ? k : k-m) * Float(sr)/Float(m)
        if abs(f) < low || abs(f) > high { re[k] = 0; im[k] = 0 }
    }
    fft(&re, &im, inverse: true)
    return Array(re.prefix(x.count))
}
func hilbertEnvelope(_ x: [Float]) -> [Float] {
    let m = nextPow2(x.count)
    var re = x + [Float](repeating: 0, count: m-x.count)
    var im = [Float](repeating: 0, count: m)
    fft(&re, &im, inverse: false)
    for k in 0..<m {                                   // analytic signal filter
        if k == 0 || (m % 2 == 0 && k == m/2) { continue }
        else if k < m/2 { re[k] *= 2; im[k] *= 2 }
        else { re[k] = 0; im[k] = 0 }
    }
    fft(&re, &im, inverse: true)
    return (0..<x.count).map { (re[$0]*re[$0] + im[$0]*im[$0]).squareRoot() }
}

// MARK: - mel filterbank (librosa Slaney, norm='slaney') + DCT-II ortho
private func hzToMel(_ f: Float) -> Float {              // Slaney
    let fMin: Float = 0, fSp: Float = 200.0/3.0
    let minLogHz: Float = 1000, minLogMel = (1000 - fMin)/fSp
    let logstep = logf(6.4)/27.0
    return f < minLogHz ? (f - fMin)/fSp : minLogMel + logf(f/minLogHz)/logstep
}
private func melToHz(_ m: Float) -> Float {
    let fMin: Float = 0, fSp: Float = 200.0/3.0
    let minLogHz: Float = 1000, minLogMel = (1000 - fMin)/fSp
    let logstep = logf(6.4)/27.0
    return m < minLogMel ? fMin + fSp*m : minLogHz*expf(logstep*(m - minLogMel))
}
func melFilterbank(sr: Int, nFFT: Int, nMels: Int) -> [[Float]] {
    let bins = nFFT/2 + 1
    let fftFreqs = (0..<bins).map { Float($0)*Float(sr)/Float(nFFT) }
    let melMin = hzToMel(0), melMax = hzToMel(Float(sr)/2)
    let melPts = (0..<(nMels+2)).map { melMin + (melMax-melMin)*Float($0)/Float(nMels+1) }
    let hzPts = melPts.map { melToHz($0) }
    var fb = [[Float]](repeating: [Float](repeating: 0, count: bins), count: nMels)
    for m in 0..<nMels {
        let lo = hzPts[m], ctr = hzPts[m+1], hi = hzPts[m+2]
        for b in 0..<bins {
            let f = fftFreqs[b]
            var w: Float = 0
            if f >= lo && f <= ctr { w = (f-lo)/max(ctr-lo, 1e-9) }
            else if f > ctr && f <= hi { w = (hi-f)/max(hi-ctr, 1e-9) }
            fb[m][b] = max(0, w)
        }
        let enorm = 2.0/max(hzPts[m+2]-hzPts[m], 1e-9)   // Slaney area norm
        for b in 0..<bins { fb[m][b] *= enorm }
    }
    return fb
}
func dctII(_ x: [Float], n: Int) -> [Float] {            // scipy dct type-2, norm='ortho'
    let N = x.count
    var out = [Float](repeating: 0, count: n)
    for k in 0..<n {
        var s: Float = 0
        for i in 0..<N { s += x[i]*cos(Float.pi*Float(k)*(2*Float(i)+1)/(2*Float(N))) }
        out[k] = 2*s
    }
    out[0] *= (1.0/(4.0*Float(N))).squareRoot()
    for k in 1..<n { out[k] *= (1.0/(2.0*Float(N))).squareRoot() }
    return out
}
