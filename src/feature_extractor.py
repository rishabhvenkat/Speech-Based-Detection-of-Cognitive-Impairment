"""
feature_extractor.py - Extract speech features from .wav audio files.

Features extracted:
  - MFCCs (13 coeff): mean + std  -> 26 features
  - Pitch (F0): mean, std, min, max -> 4 features
  - Speech rate (ZCR-based)          -> 1 feature
  - Pause ratio (silent frame ratio)  -> 1 feature
  - Jitter (pitch period variation)   -> 1 feature
  - Shimmer (amplitude variation)     -> 1 feature
  - Spectral centroid: mean, std      -> 2 features
  - Spectral rolloff: mean            -> 1 feature
  - Harmonic-to-noise ratio (HNR)    -> 1 feature
  - Fluency score                     -> 1 feature
  - Hesitation count                  -> 1 feature
  Total: 40 features
"""

import warnings
from typing import Dict, Optional, Tuple, Union
import numpy as np

warnings.filterwarnings("ignore")

# Feature names in canonical order (must match data_generator column order)
FEATURE_NAMES = (
    # MFCCs mean
    [f"mfcc{i}_mean" for i in range(1, 14)]
    # MFCCs std
    + [f"mfcc{i}_std" for i in range(1, 14)]
    # Pitch
    + ["pitch_mean", "pitch_std", "pitch_min", "pitch_max"]
    # Prosodic
    + ["speech_rate", "pause_ratio", "jitter", "shimmer"]
    # Spectral
    + ["spectral_centroid_mean", "spectral_centroid_std", "spectral_rolloff_mean"]
    # Quality
    + ["hnr", "fluency_score", "hesitation_count"]
)


def _safe_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if len(arr) > 0 else 0.0


def _safe_std(arr: np.ndarray) -> float:
    return float(np.std(arr)) if len(arr) > 0 else 0.0


def extract_mfcc(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute MFCC mean and std for each coefficient.

    Returns:
        Tuple of (means array [n_mfcc], stds array [n_mfcc])
    """
    import librosa
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # (n_mfcc, T)
    return mfccs.mean(axis=1), mfccs.std(axis=1)


def extract_pitch(
    y: np.ndarray,
    sr: int,
    fmin: float = 75.0,
    fmax: float = 300.0,
) -> Tuple[float, float, float, float]:
    """
    Estimate fundamental frequency (F0) using librosa's pyin.

    Returns:
        (mean, std, min, max) pitch in Hz; zeros if estimation fails.
    """
    import librosa
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr)
        voiced = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
        voiced = voiced[~np.isnan(voiced)]
        if len(voiced) == 0:
            return 0.0, 0.0, 0.0, 0.0
        return (
            float(np.mean(voiced)),
            float(np.std(voiced)),
            float(np.min(voiced)),
            float(np.max(voiced)),
        )
    except Exception:
        # Fallback: use zero-crossing-based period estimation
        return 0.0, 0.0, 0.0, 0.0


def extract_speech_rate(y: np.ndarray, sr: int) -> float:
    """
    Estimate speech rate (syllables/sec proxy) from zero-crossing rate.

    ZCR is higher for voiced/active speech; we normalise to approximate
    a speech-rate-like quantity in the range [1, 8].
    """
    import librosa
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rate = float(np.mean(zcr)) * sr / 100.0  # scale to syllable-like range
    return np.clip(rate, 0.5, 10.0)


def extract_pause_ratio(
    y: np.ndarray,
    sr: int,
    frame_length: int = 512,
    hop_length: int = 256,
    silence_threshold_db: float = -40.0,
) -> float:
    """
    Compute fraction of frames below energy threshold (pause frames).

    Returns:
        Pause ratio in [0, 1].
    """
    import librosa
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max(rms) + 1e-10)
    silent = np.sum(rms_db < silence_threshold_db)
    return float(silent / len(rms_db)) if len(rms_db) > 0 else 0.0


def extract_jitter(y: np.ndarray, sr: int) -> float:
    """
    Estimate jitter: mean absolute difference of consecutive pitch periods.

    Uses autocorrelation to find period lengths frame-by-frame.
    """
    import librosa
    try:
        frame_length = 512
        hop_length = 256
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        periods = []
        for frame in frames.T:
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(corr) // 2:]
            # Find first peak after lag 0
            min_lag = int(sr / 300)  # 300 Hz max
            max_lag = int(sr / 75)   # 75 Hz min
            if max_lag >= len(corr):
                continue
            peak = np.argmax(corr[min_lag:max_lag]) + min_lag
            periods.append(peak)
        if len(periods) < 2:
            return 0.0
        diffs = np.abs(np.diff(periods))
        return float(np.mean(diffs) / (np.mean(periods) + 1e-8))
    except Exception:
        return 0.0


def extract_shimmer(y: np.ndarray, sr: int) -> float:
    """
    Estimate shimmer: mean absolute difference of consecutive frame amplitudes.
    """
    import librosa
    frame_length = 512
    hop_length = 256
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    if len(rms) < 2:
        return 0.0
    diffs = np.abs(np.diff(rms))
    mean_amp = np.mean(rms) + 1e-8
    return float(np.mean(diffs) / mean_amp)


def extract_spectral_features(
    y: np.ndarray,
    sr: int,
) -> Tuple[float, float, float]:
    """
    Compute spectral centroid (mean, std) and spectral rolloff (mean).

    Returns:
        (centroid_mean, centroid_std, rolloff_mean)
    """
    import librosa
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    return (
        float(np.mean(centroid)),
        float(np.std(centroid)),
        float(np.mean(rolloff)),
    )


def extract_hnr(y: np.ndarray, sr: int) -> float:
    """
    Approximate harmonic-to-noise ratio in dB using spectral flatness.

    A lower spectral flatness ~= more harmonic content -> higher HNR proxy.
    Result is scaled to typical HNR range [0, 30].
    """
    import librosa
    try:
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        mean_flat = float(np.mean(flatness)) + 1e-10
        # HNR-like: invert flatness and scale
        hnr_proxy = -10.0 * np.log10(mean_flat + 1e-10)
        return float(np.clip(hnr_proxy, 0.0, 40.0))
    except Exception:
        return 0.0


def extract_hesitation_count(y: np.ndarray, sr: int) -> float:
    """
    Estimate hesitation count: number of distinct energy dips (potential fillers).

    Counts transitions from voiced -> near-silence -> voiced.
    """
    import librosa
    hop_length = 256
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    threshold = np.mean(rms) * 0.3
    in_dip = False
    count = 0
    for val in rms:
        if val < threshold and not in_dip:
            in_dip = True
            count += 1
        elif val >= threshold:
            in_dip = False
    return float(count)


def extract_features(
    audio_path: str,
    sr: int = 16000,
    min_duration: float = 0.5,
) -> Optional[Dict[str, float]]:
    """
    Extract all speech features from a .wav file.

    Args:
        audio_path: Path to the .wav audio file.
        sr: Target sample rate for loading.
        min_duration: Minimum required audio duration in seconds.

    Returns:
        Dict mapping feature name -> float value, or None if extraction fails.
    """
    try:
        import librosa
    except ImportError:
        raise ImportError(
            "librosa is required for feature extraction. "
            "Install with: pip install librosa"
        )

    try:
        y, loaded_sr = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        print(f"[feature_extractor] ERROR loading {audio_path}: {e}")
        return None

    duration = len(y) / sr
    if duration < min_duration:
        print(
            f"[feature_extractor] WARNING: {audio_path} too short "
            f"({duration:.2f}s < {min_duration}s). Skipping."
        )
        return None

    features: Dict[str, float] = {}

    # ── MFCCs ────────────────────────────────────────────────────────────────
    mfcc_means, mfcc_stds = extract_mfcc(y, sr)
    for i in range(13):
        features[f"mfcc{i+1}_mean"] = float(mfcc_means[i])
        features[f"mfcc{i+1}_std"]  = float(mfcc_stds[i])

    # ── Pitch ─────────────────────────────────────────────────────────────────
    p_mean, p_std, p_min, p_max = extract_pitch(y, sr)
    features["pitch_mean"] = p_mean
    features["pitch_std"]  = p_std
    features["pitch_min"]  = p_min
    features["pitch_max"]  = p_max

    # ── Prosodic ──────────────────────────────────────────────────────────────
    features["speech_rate"]      = extract_speech_rate(y, sr)
    features["pause_ratio"]      = extract_pause_ratio(y, sr)
    features["jitter"]           = extract_jitter(y, sr)
    features["shimmer"]          = extract_shimmer(y, sr)

    # ── Spectral ──────────────────────────────────────────────────────────────
    c_mean, c_std, r_mean = extract_spectral_features(y, sr)
    features["spectral_centroid_mean"] = c_mean
    features["spectral_centroid_std"]  = c_std
    features["spectral_rolloff_mean"]  = r_mean

    # ── Quality / fluency ─────────────────────────────────────────────────────
    features["hnr"] = extract_hnr(y, sr)
    pause   = features["pause_ratio"]
    s_rate  = features["speech_rate"]
    # Fluency: inversely related to pauses, positively to speech rate
    features["fluency_score"] = float(
        np.clip((1.0 - pause) * min(s_rate / 5.0, 1.0), 0.0, 1.0)
    )
    features["hesitation_count"] = extract_hesitation_count(y, sr)

    return features


def features_to_array(features: Dict[str, float]) -> np.ndarray:
    """
    Convert feature dict to a flat numpy array in canonical order.

    Args:
        features: Dict of feature name -> value.

    Returns:
        1-D numpy array of length len(FEATURE_NAMES).
    """
    return np.array([features.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float32)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        feats = extract_features(path)
        if feats:
            print(f"Extracted {len(feats)} features from {path}:")
            for k, v in feats.items():
                print(f"  {k:<35} {v:.4f}")
        else:
            print("Feature extraction failed.")
    else:
        print(f"Usage: python feature_extractor.py <audio.wav>")
        print(f"Feature names ({len(FEATURE_NAMES)}):")
        for n in FEATURE_NAMES:
            print(f"  {n}")
