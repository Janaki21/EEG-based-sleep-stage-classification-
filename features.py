import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import antropy as ant

from config import OUTPUT_DIR
from utils import print_header


EPS = 1e-12


def safe_float(x, default=0.0):
    try:
        val = float(x)
        if np.isnan(val) or np.isinf(val):
            return default
        return val
    except Exception:
        return default


def zero_crossing_rate(x):
    x = np.asarray(x)
    return np.sum(np.diff(np.signbit(x))) / max(len(x), 1)


def rms(x):
    x = np.asarray(x)
    return np.sqrt(np.mean(np.square(x)) + EPS)


def hjorth_parameters(x):
    x = np.asarray(x)
    dx = np.diff(x)
    ddx = np.diff(dx)

    var_x = np.var(x)
    var_dx = np.var(dx) if len(dx) > 0 else 0.0
    var_ddx = np.var(ddx) if len(ddx) > 0 else 0.0

    activity = var_x
    mobility = np.sqrt(var_dx / (var_x + EPS)) if var_x > 0 else 0.0
    complexity = (
        np.sqrt(var_ddx / (var_dx + EPS)) / (mobility + EPS)
        if var_dx > 0 and mobility > 0
        else 0.0
    )
    return safe_float(activity), safe_float(mobility), safe_float(complexity)


def spectral_entropy_from_psd(psd):
    psd = np.asarray(psd, dtype=float)
    psd = psd / (np.sum(psd) + EPS)
    return safe_float(-np.sum(psd * np.log2(psd + EPS)))


def bandpower(freqs, psd, fmin, fmax):
    freqs = np.asarray(freqs)
    psd = np.asarray(psd)
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    if not np.any(idx):
        return 0.0
    return safe_float(np.trapz(psd[idx], freqs[idx]))


def spectral_edge_frequency(freqs, psd, edge=0.95):
    freqs = np.asarray(freqs)
    psd = np.asarray(psd)

    if len(freqs) == 0 or len(psd) == 0:
        return 0.0

    cumulative = np.cumsum(psd)
    cumulative = cumulative / (cumulative[-1] + EPS)
    idx = np.where(cumulative >= edge)[0]
    if len(idx) == 0:
        return safe_float(freqs[-1])
    return safe_float(freqs[idx[0]])


def safe_app_entropy(x):
    try:
        return safe_float(ant.app_entropy(x))
    except Exception:
        return 0.0


def safe_sample_entropy(x):
    try:
        return safe_float(ant.sample_entropy(x))
    except Exception:
        return 0.0


def safe_perm_entropy(x):
    try:
        return safe_float(ant.perm_entropy(x, normalize=True))
    except Exception:
        return 0.0


def safe_higuchi_fd(x):
    try:
        return safe_float(ant.higuchi_fd(x))
    except Exception:
        return 0.0


def safe_lz_complexity(x):
    try:
        x_binary = (x > np.median(x)).astype(int)
        return safe_float(ant.lziv_complexity(x_binary))
    except Exception:
        return 0.0


def extract_statistical_features(x):
    activity, mobility, complexity = hjorth_parameters(x)
    return {
        "mean": safe_float(np.mean(x)),
        "std": safe_float(np.std(x)),
        "var": safe_float(np.var(x)),
        "skew": safe_float(skew(x)),
        "kurtosis": safe_float(kurtosis(x)),
        "rms": safe_float(rms(x)),
        "zcr": safe_float(zero_crossing_rate(x)),
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
    }


def extract_spectral_features(x, sfreq):
    freqs, psd = welch(x, fs=sfreq, nperseg=min(256, len(x)))
    total_power = bandpower(freqs, psd, 0.5, 30.0) + EPS

    bp_delta = bandpower(freqs, psd, 0.5, 4.0)
    bp_theta = bandpower(freqs, psd, 4.0, 8.0)
    bp_alpha = bandpower(freqs, psd, 8.0, 13.0)
    bp_sigma = bandpower(freqs, psd, 11.0, 16.0)
    bp_beta = bandpower(freqs, psd, 13.0, 30.0)

    peak_freq = freqs[np.argmax(psd)] if len(psd) > 0 else 0.0
    spec_entropy = spectral_entropy_from_psd(psd)
    sef95 = spectral_edge_frequency(freqs, psd, edge=0.95)

    return {
        "abs_delta": bp_delta,
        "abs_theta": bp_theta,
        "abs_alpha": bp_alpha,
        "abs_sigma": bp_sigma,
        "abs_beta": bp_beta,
        "rel_delta": safe_float(bp_delta / total_power),
        "rel_theta": safe_float(bp_theta / total_power),
        "rel_alpha": safe_float(bp_alpha / total_power),
        "rel_sigma": safe_float(bp_sigma / total_power),
        "rel_beta": safe_float(bp_beta / total_power),
        "spectral_entropy": spec_entropy,
        "peak_frequency": safe_float(peak_freq),
        "sef95": sef95,
    }


def extract_complexity_features(x):
    return {
        "approx_entropy": safe_app_entropy(x),
        "sample_entropy": safe_sample_entropy(x),
        "perm_entropy": safe_perm_entropy(x),
        "higuchi_fd": safe_higuchi_fd(x),
        "lz_complexity": safe_lz_complexity(x),
    }


def get_feature_definition_table():
    rows = [
        # Statistical (10)
        ("mean", "statistical", "Epoch mean amplitude", "Mean(x)"),
        ("std", "statistical", "Standard deviation", "Std(x)"),
        ("var", "statistical", "Variance", "Var(x)"),
        ("skew", "statistical", "Skewness", "Third standardized moment"),
        ("kurtosis", "statistical", "Kurtosis", "Fourth standardized moment"),
        ("rms", "statistical", "Root mean square amplitude", "sqrt(mean(x^2))"),
        ("zcr", "statistical", "Zero-crossing rate", "Sign-change count / N"),
        ("hjorth_activity", "statistical", "Hjorth activity", "Var(x)"),
        ("hjorth_mobility", "statistical", "Hjorth mobility", "sqrt(Var(dx)/Var(x))"),
        ("hjorth_complexity", "statistical", "Hjorth complexity", "Mobility(dx)/Mobility(x)"),
        # Spectral (13)
        ("abs_delta", "spectral", "Absolute delta power (0.5-4 Hz)", "Bandpower"),
        ("abs_theta", "spectral", "Absolute theta power (4-8 Hz)", "Bandpower"),
        ("abs_alpha", "spectral", "Absolute alpha power (8-13 Hz)", "Bandpower"),
        ("abs_sigma", "spectral", "Absolute sigma power (11-16 Hz)", "Bandpower"),
        ("abs_beta", "spectral", "Absolute beta power (13-30 Hz)", "Bandpower"),
        ("rel_delta", "spectral", "Relative delta power", "Delta / total power"),
        ("rel_theta", "spectral", "Relative theta power", "Theta / total power"),
        ("rel_alpha", "spectral", "Relative alpha power", "Alpha / total power"),
        ("rel_sigma", "spectral", "Relative sigma power", "Sigma / total power"),
        ("rel_beta", "spectral", "Relative beta power", "Beta / total power"),
        ("spectral_entropy", "spectral", "Shannon entropy of PSD", "-sum(p log2 p)"),
        ("peak_frequency", "spectral", "Frequency of PSD peak", "argmax PSD"),
        ("sef95", "spectral", "Spectral edge frequency 95%", "95% cumulative PSD"),
        # Complexity (5)
        ("approx_entropy", "complexity", "Approximate entropy", "ApEn"),
        ("sample_entropy", "complexity", "Sample entropy", "SampEn"),
        ("perm_entropy", "complexity", "Permutation entropy", "PE"),
        ("higuchi_fd", "complexity", "Higuchi fractal dimension", "HFD"),
        ("lz_complexity", "complexity", "Lempel-Ziv complexity", "LZC"),
    ]
    return pd.DataFrame(rows, columns=["feature_name", "domain", "description", "formula_hint"])


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
    features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features_df


def save_feature_sets(features_df, y, subject_ids):
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

    all_cols = statistical_cols + spectral_cols + complexity_cols

    missing = [c for c in all_cols if c not in features_df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    np.save(OUTPUT_DIR / "y_labels.npy", y)
    np.save(OUTPUT_DIR / "subject_ids.npy", subject_ids)

    features_df.to_csv(OUTPUT_DIR / "features_all.csv", index=False)
    features_df[statistical_cols].to_csv(OUTPUT_DIR / "features_statistical.csv", index=False)
    features_df[spectral_cols].to_csv(OUTPUT_DIR / "features_spectral.csv", index=False)
    features_df[complexity_cols].to_csv(OUTPUT_DIR / "features_complexity.csv", index=False)
    features_df[statistical_cols + complexity_cols].to_csv(
        OUTPUT_DIR / "features_stat_complexity.csv", index=False
    )
    features_df[spectral_cols + complexity_cols].to_csv(
        OUTPUT_DIR / "features_spectral_complexity.csv", index=False
    )

    feature_defs = get_feature_definition_table()
    feature_defs.to_csv(OUTPUT_DIR / "feature_definitions.csv", index=False)

    summary_rows = []
    for domain, cols in {
        "statistical": statistical_cols,
        "spectral": spectral_cols,
        "complexity": complexity_cols,
        "all": all_cols,
    }.items():
        summary_rows.append(
            {
                "feature_set": domain,
                "n_features": len(cols),
                "feature_names": ", ".join(cols),
            }
        )
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "feature_set_summary.csv", index=False)

    print_header("FEATURE FILES SAVED")
    print(f"All features shape: {features_df.shape}")
    print(f"Feature definitions saved to: {OUTPUT_DIR / 'feature_definitions.csv'}")


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