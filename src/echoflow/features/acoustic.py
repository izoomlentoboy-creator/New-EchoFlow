"""Clinical acoustic biomarkers for voice-pathology assessment.

These are the *interpretable* features clinicians actually use (the GRBAS /
AVQI family): fundamental-frequency statistics, jitter (frequency
perturbation), shimmer (amplitude perturbation), harmonics-to-noise ratio,
cepstral peak prominence (CPP/CPPS - the single most robust dysphonia marker),
glottal-to-noise excitation (GNE, breathiness) and spectral descriptors.

They form the interpretable "biomarker branch" of the hybrid model and are
partly invariant to the synthetic→real domain gap because they measure the
same *physical* quantities regardless of how the audio was produced.

The implementation is intentionally self-contained (numpy + a single shared
STFT via librosa) so the exact same maths is re-implemented natively on iOS
(see ios/EchoFlow/BiomarkerExtractor.swift) and checked by a parity test.
"""
from __future__ import annotations

import numpy as np

# Ordered names of the biomarker vector produced by ``acoustic_biomarkers``.
ACOUSTIC_NAMES = [
    "f0_mean", "f0_std", "f0_min", "f0_max", "f0_range",
    "jitter_local", "jitter_ddp",
    "shimmer_local", "shimmer_dda",
    "hnr_mean", "hnr_std",
    "cpp_mean", "cpp_std",
    "gne",
    "spectral_tilt",
    "voiced_fraction",
    "zcr_mean", "zcr_std",
    "centroid_mean", "centroid_std",
    "bandwidth_mean", "bandwidth_std",
    "rolloff_mean", "rolloff_std",
    "flatness_mean", "flatness_std",
] + [f"mfcc{i+1}_mean" for i in range(13)]

N_ACOUSTIC = len(ACOUSTIC_NAMES)   # 39


def _frame_signal(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def estimate_f0(x: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 400.0,
                frame_ms: float = 40.0, hop_ms: float = 10.0):
    """Autocorrelation-based F0 tracker.

    Returns (f0_per_frame, voiced_flags, periods_in_samples).  Unvoiced frames
    have ``f0 = 0``.  Robust enough for biomarker estimation and easy to port.
    """
    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    frames = _frame_signal(x, frame, hop)
    win = np.hanning(frame)
    min_lag = max(1, int(sr / fmax))
    max_lag = min(frame - 1, int(sr / fmin))

    f0 = np.zeros(len(frames), dtype=np.float32)
    voiced = np.zeros(len(frames), dtype=bool)
    periods = []
    for i, fr in enumerate(frames):
        seg = fr * win
        seg = seg - seg.mean()
        if np.sqrt(np.mean(seg ** 2)) < 1e-4:      # silence
            continue
        ac = np.correlate(seg, seg, mode="full")[frame - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        search = ac[min_lag:max_lag]
        if len(search) == 0:
            continue
        lag = int(np.argmax(search)) + min_lag
        peak = ac[lag]
        if peak > 0.45:                            # clear periodicity -> voiced
            f0[i] = sr / lag
            voiced[i] = True
            periods.append(lag)
    return f0, voiced, np.array(periods, dtype=np.float32)


def _jitter_shimmer(x: np.ndarray, sr: int, f0: np.ndarray, voiced: np.ndarray,
                    hop_ms: float = 10.0):
    """Local jitter (frequency) and shimmer (amplitude) from the F0 contour."""
    vf0 = f0[voiced]
    if len(vf0) < 3:
        return 0.0, 0.0, 0.0, 0.0
    periods = sr / np.clip(vf0, 1e-3, None)
    dperiod = np.abs(np.diff(periods))
    mean_period = np.mean(periods) + 1e-8
    jitter_local = np.mean(dperiod) / mean_period
    ddp = np.abs(np.diff(np.diff(periods)))
    jitter_ddp = np.mean(ddp) / mean_period if len(ddp) else 0.0

    hop = int(sr * hop_ms / 1000)
    amps = []
    voiced_idx = np.where(voiced)[0]
    for k, fi in enumerate(voiced_idx):
        center = fi * hop
        half = int(periods[k] / 2) if k < len(periods) else int(mean_period / 2)
        half = max(half, 1)
        lo = max(0, center - half)
        hi = min(len(x), center + half)
        if hi > lo:
            amps.append(np.max(np.abs(x[lo:hi])))
    amps = np.asarray(amps)
    if len(amps) < 3 or np.mean(amps) < 1e-8:
        return float(jitter_local), float(jitter_ddp), 0.0, 0.0
    mean_amp = np.mean(amps) + 1e-8
    shimmer_local = np.mean(np.abs(np.diff(amps))) / mean_amp
    shimmer_dda = (np.mean(np.abs(np.diff(np.diff(amps)))) / mean_amp
                   if len(amps) > 2 else 0.0)
    return (float(jitter_local), float(jitter_ddp),
            float(shimmer_local), float(shimmer_dda))


def _hnr_from_frames(frames: np.ndarray, win: np.ndarray, sr: int):
    """Harmonics-to-noise ratio (dB) per voiced frame via autocorrelation."""
    frame = frames.shape[1]
    min_lag = max(1, int(sr / 400))
    max_lag = min(frame - 1, int(sr / 60))
    vals = []
    for fr in frames:
        seg = (fr * win)
        seg = seg - seg.mean()
        if np.sqrt(np.mean(seg ** 2)) < 1e-4:
            continue
        ac = np.correlate(seg, seg, mode="full")[frame - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        search = ac[min_lag:max_lag]
        if len(search) == 0:
            continue
        r = float(np.clip(np.max(search), 1e-6, 0.999999))
        vals.append(10 * np.log10(r / (1 - r)))
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


def _cpp_from_logspec(log_mag: np.ndarray, sr: int, n_fft: int,
                      fmin: float = 60.0, fmax: float = 400.0):
    """Cepstral Peak Prominence per frame.

    CPP = height of the first rahmonic peak above the linear-regression
    baseline of the (real) cepstrum, in the quefrency band of plausible F0.
    Averaging across frames approximates the clinically-used smoothed CPPS.
    ``log_mag`` is (freq_bins, n_frames) = log magnitude spectrum.
    """
    # real cepstrum: irfft of the log magnitude along the frequency axis
    cep = np.fft.irfft(log_mag, n=n_fft, axis=0)          # (n_fft, T)
    q = np.arange(n_fft) / sr                              # quefrency in seconds
    qmin = 1.0 / fmax
    qmax = 1.0 / fmin
    band = (q >= qmin) & (q <= qmax)
    if not band.any():
        return 0.0, 0.0
    cpps = []
    qb = q[band]
    for t in range(cep.shape[1]):
        c = cep[:, t]
        cb = c[band]
        if np.allclose(cb, 0):
            continue
        # regression baseline across the full analysed quefrency range
        A = np.vstack([qb, np.ones_like(qb)]).T
        coef, *_ = np.linalg.lstsq(A, cb, rcond=None)
        baseline = A @ coef
        cpps.append(float(np.max(cb - baseline)))
    if not cpps:
        return 0.0, 0.0
    return float(np.mean(cpps)), float(np.std(cpps))


def _gne(x: np.ndarray, sr: int, n_bands: int = 4):
    """Glottal-to-Noise Excitation ratio (breathiness), simplified.

    Splits the signal into frequency bands, takes Hilbert envelopes and uses
    the maximum cross-correlation between non-adjacent bands.  Periodic
    (harmonic) excitation correlates across bands; turbulent noise does not, so
    a high GNE means a healthy, non-breathy voice.
    """
    from scipy.signal import butter, lfilter, hilbert
    nyq = sr / 2
    edges = np.linspace(300, min(5000, nyq * 0.95), n_bands + 1)
    envs = []
    for i in range(n_bands):
        lo, hi = edges[i] / nyq, edges[i + 1] / nyq
        lo = max(lo, 1e-3); hi = min(hi, 0.999)
        if hi <= lo:
            continue
        b, a = butter(2, [lo, hi], btype="band")
        band = lfilter(b, a, x)
        env = np.abs(hilbert(band))
        env = env - env.mean()
        envs.append(env)
    best = 0.0
    for i in range(len(envs)):
        for j in range(i + 1, len(envs)):
            a_, b_ = envs[i], envs[j]
            denom = (np.std(a_) * np.std(b_) * len(a_)) + 1e-9
            corr = float(np.dot(a_, b_) / denom)
            best = max(best, corr)
    return float(np.clip(best, 0.0, 1.0))


def _spectral_tilt(power_mean: np.ndarray, sr: int, n_fft: int):
    """Slope (dB per kHz) of the long-term average spectrum (LTAS).

    Breathy voices lose high-frequency energy slowly (flatter / less negative
    is not always intuitive); hyperfunctional voices add HF energy.  A single,
    robust global descriptor.
    """
    freqs = np.linspace(0, sr / 2, len(power_mean))
    db = 10 * np.log10(power_mean + 1e-10)
    mask = freqs > 50
    f = freqs[mask] / 1000.0                               # kHz
    y = db[mask]
    if len(f) < 2:
        return 0.0
    A = np.vstack([f, np.ones_like(f)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0])                                  # dB/kHz


def acoustic_biomarkers(x: np.ndarray, sr: int,
                        n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """Compute the ordered biomarker vector (length ``N_ACOUSTIC``)."""
    import librosa
    x = np.asarray(x, dtype=np.float32)
    if np.max(np.abs(x)) > 0:
        x = x / np.max(np.abs(x))

    # ---- F0 / perturbation / HNR (time-domain, autocorrelation) ----
    frame = int(sr * 0.040)
    hop_t = int(sr * 0.010)
    frames = _frame_signal(x, frame, hop_t)
    win = np.hanning(frame).astype(np.float32)
    f0, voiced, _ = estimate_f0(x, sr)
    vf0 = f0[voiced]
    if len(vf0):
        f0_mean, f0_std = float(np.mean(vf0)), float(np.std(vf0))
        f0_min, f0_max = float(np.min(vf0)), float(np.max(vf0))
    else:
        f0_mean = f0_std = f0_min = f0_max = 0.0
    f0_range = f0_max - f0_min
    voiced_fraction = float(np.mean(voiced)) if len(voiced) else 0.0
    jit_l, jit_ddp, shim_l, shim_dda = _jitter_shimmer(x, sr, f0, voiced)
    hnr_mean, hnr_std = _hnr_from_frames(frames, win, sr)
    gne = _gne(x, sr)

    # ---- single shared STFT for all spectral descriptors ----
    S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop))    # (F, T)
    power = S ** 2
    log_mag = np.log(S + 1e-6)
    cpp_mean, cpp_std = _cpp_from_logspec(log_mag, sr, n_fft)
    tilt = _spectral_tilt(power.mean(axis=1), sr, n_fft)

    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    melspec = librosa.feature.melspectrogram(S=power, sr=sr, n_mels=40)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(melspec), n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(x, frame_length=frame,
                                             hop_length=hop_t)[0]

    feats = [
        f0_mean, f0_std, f0_min, f0_max, f0_range,
        jit_l, jit_ddp,
        shim_l, shim_dda,
        hnr_mean, hnr_std,
        cpp_mean, cpp_std,
        gne,
        tilt,
        voiced_fraction,
        float(np.mean(zcr)), float(np.std(zcr)),
        float(np.mean(centroid)), float(np.std(centroid)),
        float(np.mean(bandwidth)), float(np.std(bandwidth)),
        float(np.mean(rolloff)), float(np.std(rolloff)),
        float(np.mean(flatness)), float(np.std(flatness)),
    ]
    feats.extend([float(np.mean(c)) for c in mfcc])
    out = np.asarray(feats, dtype=np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    assert out.shape[0] == N_ACOUSTIC, (out.shape, N_ACOUSTIC)
    return out
