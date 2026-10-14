"""Dataset loading.

Two on-disk formats are supported:

* the UCI HAR layout (``train/X_train.txt``, ``train/y_train.txt``,
  ``train/subject_train.txt``, ``features.txt``, and the same for ``test``),
  which is what ``data/CS3244 project data`` contains;
* a pair of feature CSVs whose columns are ``subject, <unused>, label,
  feature...`` -- the format the original Colab notebook read
  (``Train/features_train.csv`` / ``Test/features_test.csv``).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.paths import DATA_DIR

DEFAULT_DATA_DIR = DATA_DIR / "CS3244 project data"
WAVELET_PREFIX = "wavelet"


@dataclass
class HARDataset:
    X_train: pd.DataFrame
    y_train: np.ndarray
    groups_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    groups_test: np.ndarray
    activity_labels: dict[int, str]

    @property
    def n_classes(self) -> int:
        return len(np.unique(self.y_train))

    def without_wavelet(self) -> pd.DataFrame:
        """Training features excluding any wavelet-derived columns."""
        return self.X_train.loc[:, ~self.X_train.columns.str.startswith(WAVELET_PREFIX)]


def _read_feature_names(path: Path) -> list[str]:
    """Read ``features.txt``; names may be prefixed with their 1-based index.

    The UCI file repeats some names (e.g. the ``bandsEnergy`` features), so
    duplicates get a numeric suffix to keep column names unique.
    """
    names, seen = [], {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.strip().split(maxsplit=1)
        name = parts[1] if len(parts) == 2 and parts[0].isdigit() else line.strip()
        count = seen.get(name, 0)
        seen[name] = count + 1
        names.append(name if count == 0 else f"{name}_{count}")
    return names


def _read_activity_labels(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    labels = {}
    for line in path.read_text().splitlines():
        if line.strip():
            idx, name = line.split(maxsplit=1)
            labels[int(idx)] = name.strip()
    return labels


def _read_split(data_dir: Path, split: str, columns: list[str]):
    split_dir = data_dir / split
    X = pd.read_csv(split_dir / f"X_{split}.txt", sep=r"\s+", header=None)
    X.columns = columns
    y = np.loadtxt(split_dir / f"y_{split}.txt", dtype=int)
    groups = np.loadtxt(split_dir / f"subject_{split}.txt", dtype=int)
    return X, y, groups


def load_uci_har(data_dir: Path = DEFAULT_DATA_DIR) -> HARDataset:
    data_dir = Path(data_dir)
    columns = _read_feature_names(data_dir / "features.txt")
    X_train, y_train, g_train = _read_split(data_dir, "train", columns)
    X_test, y_test, g_test = _read_split(data_dir, "test", columns)
    return HARDataset(X_train, y_train, g_train, X_test, y_test, g_test,
                      _read_activity_labels(data_dir / "activity_labels.txt"))


def load_feature_csvs(train_csv: Path, test_csv: Path,
                      activity_labels_path: Path | None = None) -> HARDataset:
    """Load the notebook's CSV format: col 0 = subject, col 2 = label, cols 3+ = features."""
    train, test = pd.read_csv(train_csv), pd.read_csv(test_csv)
    return HARDataset(
        X_train=train.iloc[:, 3:], y_train=train.iloc[:, 2].to_numpy(),
        groups_train=train.iloc[:, 0].to_numpy(),
        X_test=test.iloc[:, 3:], y_test=test.iloc[:, 2].to_numpy(),
        groups_test=test.iloc[:, 0].to_numpy(),
        activity_labels=_read_activity_labels(activity_labels_path) if activity_labels_path else {},
    )


def make_cv(n_splits: int = 5, random_state: int = 42) -> StratifiedGroupKFold:
    """Stratified folds grouped by subject, so validation subjects are unseen."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def describe_folds(cv, X, y, groups) -> pd.DataFrame:
    rows = []
    for i, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        rows.append({
            "fold": i,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "val_subjects": np.unique(groups[val_idx]).tolist(),
        })
    return pd.DataFrame(rows)
