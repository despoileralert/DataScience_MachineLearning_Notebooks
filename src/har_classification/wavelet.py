"""Wavelet features computed from the raw ``Inertial Signals`` windows.

The original notebook compared the 561 engineered features against the same
features plus extra ``wavelet*`` columns. The script that produced those
columns was not committed, so this module is a stand-in: a discrete wavelet
decomposition of each 128-sample window, summarised per sub-band.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.har_classification.data import WAVELET_PREFIX, HARDataset

CHANNELS = [
    f"{signal}_{axis}"
    for signal in ("body_acc", "body_gyro", "total_acc")
    for axis in ("x", "y", "z")
]


def _band_stats(coeffs: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "energy": np.mean(coeffs**2, axis=1),
        "std": np.std(coeffs, axis=1),
        "meanabs": np.mean(np.abs(coeffs), axis=1),
    }


def wavelet_features(data_dir: Path, split: str, wavelet: str = "db4", level: int = 3) -> pd.DataFrame:
    """One row per window: energy / std / mean-|x| for each channel and sub-band."""
    import pywt  # optional dependency, only needed for this feature set

    signal_dir = Path(data_dir) / split / "Inertial Signals"
    features = {}
    for channel in CHANNELS:
        windows = np.loadtxt(signal_dir / f"{channel}_{split}.txt")
        bands = pywt.wavedec(windows, wavelet, level=level, axis=1)
        band_names = [f"A{level}"] + [f"D{lvl}" for lvl in range(level, 0, -1)]
        for band_name, coeffs in zip(band_names, bands):
            for stat, values in _band_stats(coeffs).items():
                features[f"{WAVELET_PREFIX}-{channel}-{band_name}-{stat}"] = values
    return pd.DataFrame(features)


def add_wavelet_features(dataset: HARDataset, data_dir: Path, **kwargs) -> HARDataset:
    """Return a copy of ``dataset`` with wavelet columns appended to X_train / X_test."""
    extra = {split: wavelet_features(data_dir, split, **kwargs) for split in ("train", "test")}
    return HARDataset(
        X_train=pd.concat([dataset.X_train.reset_index(drop=True), extra["train"]], axis=1),
        y_train=dataset.y_train,
        groups_train=dataset.groups_train,
        X_test=pd.concat([dataset.X_test.reset_index(drop=True), extra["test"]], axis=1),
        y_test=dataset.y_test,
        groups_test=dataset.groups_test,
        activity_labels=dataset.activity_labels,
    )
