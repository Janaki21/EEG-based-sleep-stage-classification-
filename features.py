import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import antropy as ant

from config import OUTPUT_DIR
from utils import print_header


def zero_crossing_rate(x):
    return np.sum(np.diff(np.signbit(x))) / len(x)


def rms(x):
    return np.sqrt(np.mean(np.square(x)))


def hjorth_parameters(x):
    dx = np.diff(x)
    ddx = np.diff(dx)

    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)

    activity = var_x
    mobility = np.sqrt(var_dx / var_x) if var_x > 0 else 0.0
    complexity = (
        np.sqrt(var_ddx / var_dx) / mobility
        if var_dx > 0 and mobility > 0
        else 0.0
    )

    return activity, mobility, complexity


def spectral_entropy_from_psd(psd):
    psd = np.asarray(psd, dtype=float)
    psd = psd / (np.sum(psd) + 1e-12)
    return -np.sum(psd * np.log2(psd + 1e-12))


def bandpower(freqs, psd, fmin, fmax):
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    if not np.any(idx):
        return 0.0
    return np.trapz(psd[idx], freqs[idx])


def spectral_edge_frequency(freqs, psd, edge=0.95):
    cumulative = np.cumsum(psd)
    cumulative = cumulative / (cumulative[-1] + 1e-12)
    idx = np.where(cumulative >= edge)[0]
    if len(idx) == 0:
        return freqs[-1]
    return freqs[idx[0]]


def extract_statistical_features(x):
    activity, mobility, complexity = hjorth_parameters(x)
    return {
        "mean": np.mean(x),
        "std": np.std(x),
        "var": np.var(x),
        "skew": skew(x),
        "kurtosis": kurtosis(x),
        "rms": rms(x),
        "zcr": zero_crossing_rate(x),
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
    }


def extract_spectral_features(x, sfreq):
    freqs, psd = welch(x, fs=sfreq, nperseg=min(256, len(x)))

    total_power = bandpower(freqs, psd, 0.5, 30.0) + 1e-12

    bp_delta = bandpower(freqs, psd, 0.5, 4.0)
    bp_theta = bandpower(freqs, psd, 4.0, 8.0)
    bp_alpha = bandpower(freqs, psd, 8.0, 12.0)
    bp_sigma = bandpower(freqs, psd, 12.0, 16.0)
    bp_beta = bandpower(freqs, psd, 16.0, 30.0)

    peak_freq = freqs[np.argmax(psd)] if len(psd) > 0 else 0.0
    spec_entropy = spectral_entropy_from_psd(psd)
    sef95 = spectral_edge_frequency(freqs, psd, edge=0.95)

    return {
        "abs_delta": bp_delta,
        "abs_theta": bp_theta,
        "abs_alpha": bp_alpha,
        "abs_sigma": bp_sigma,
        "abs_beta": bp_beta,
        "rel_delta": bp_delta / total_power,
        "rel_theta": bp_theta / total_power,
        "rel_alpha": bp_alpha / total_power,
        "rel_sigma": bp_sigma / total_power,
        "rel_beta": bp_beta / total_power,
        "spectral_entropy": spec_entropy,
        "peak_frequency": peak_freq,
        "sef95": sef95,
    }


def extract_complexity_features(x):
    x_binary = (x > np.median(x)).astype(int)

    return {
        "approx_entropy": ant.app_entropy(x),
        "sample_entropy": ant.sample_entropy(x),
        "perm_entropy": ant.perm_entropy(x, normalize=True),
        "higuchi_fd": ant.higuchi_fd(x),
        "lz_complexity": ant.lziv_complexity(x_binary),
    }


def extract_all_features(X, sfreq):
    all_rows = []

    print_header("EXTRACTING FEATURES")

    for i, epoch in enumerate(X):
        if i % 500 == 0:
            print(f"Processing epoch {i + 1}/{len(X)}")

        row = {}
        row.update(extract_statistical_features(epoch))
        row.update(extract_spectral_features(epoch, sfreq))
        row.update(extract_complexity_features(epoch))

        all_rows.append(row)

    features_df = pd.DataFrame(all_rows)
    return features_df


def save_feature_sets(features_df, y, subject_ids):
    features_path = OUTPUT_DIR / "features_all.csv"
    features_df.to_csv(features_path, index=False)

    statistical_cols = [
        "mean", "std", "var", "skew", "kurtosis", "rms", "zcr",
        "hjorth_activity", "hjorth_mobility", "hjorth_complexity"
    ]
    spectral_cols = [
        "abs_delta", "abs_theta", "abs_alpha", "abs_sigma", "abs_beta",
        "rel_delta", "rel_theta", "rel_alpha", "rel_sigma", "rel_beta",
        "spectral_entropy", "peak_frequency", "sef95"
    ]
    complexity_cols = [
        "approx_entropy", "sample_entropy", "perm_entropy",
        "higuchi_fd", "lz_complexity"
    ]

    np.save(OUTPUT_DIR / "y_labels.npy", y)
    np.save(OUTPUT_DIR / "subject_ids.npy", subject_ids)

    features_df[statistical_cols].to_csv(OUTPUT_DIR / "features_statistical.csv", index=False)
    features_df[spectral_cols].to_csv(OUTPUT_DIR / "features_spectral.csv", index=False)
    features_df[complexity_cols].to_csv(OUTPUT_DIR / "features_complexity.csv", index=False)

    combined_sc = statistical_cols + complexity_cols
    combined_spc = spectral_cols + complexity_cols

    features_df[combined_sc].to_csv(OUTPUT_DIR / "features_stat_complexity.csv", index=False)
    features_df[combined_spc].to_csv(OUTPUT_DIR / "features_spectral_complexity.csv", index=False)

    print_header("FEATURE FILES SAVED")
    print(f"All features: {features_path}")
    print(f"Shape: {features_df.shape}")


def main():
    X = np.load(OUTPUT_DIR / "X_epochs.npy")
    y = np.load(OUTPUT_DIR / "y_labels.npy")
    subject_ids = np.load(OUTPUT_DIR / "subject_ids.npy")

    sfreq = 100.0

    features_df = extract_all_features(X, sfreq)
    save_feature_sets(features_df, y, subject_ids)

    print_header("FEATURE EXTRACTION COMPLETE")
    print(features_df.head())
    print(features_df.shape)


if __name__ == "__main__":
    main()