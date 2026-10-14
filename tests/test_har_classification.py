import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.har_classification import data, models, train


def test_read_feature_names_strips_index_and_dedupes(tmp_path):
    path = tmp_path / "features.txt"
    path.write_text("1 tBodyAcc-mean()-X\n2 fBodyAcc-bandsEnergy()-1,8\n3 fBodyAcc-bandsEnergy()-1,8\n")
    assert data._read_feature_names(path) == [
        "tBodyAcc-mean()-X", "fBodyAcc-bandsEnergy()-1,8", "fBodyAcc-bandsEnergy()-1,8_1",
    ]


def test_load_feature_csvs(tmp_path):
    frame = pd.DataFrame({"subject": [1, 2], "window": [0, 0], "label": [3, 4],
                          "f1": [0.1, 0.2], "wavelet-a": [1.0, 2.0]})
    frame.to_csv(tmp_path / "train.csv", index=False)
    frame.to_csv(tmp_path / "test.csv", index=False)
    ds = data.load_feature_csvs(tmp_path / "train.csv", tmp_path / "test.csv")
    assert list(ds.X_train.columns) == ["f1", "wavelet-a"]
    assert ds.y_train.tolist() == [3, 4]
    assert ds.groups_train.tolist() == [1, 2]
    assert list(ds.without_wavelet().columns) == ["f1"]


def test_cv_keeps_subjects_out_of_training_folds():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 20)
    y = rng.integers(0, 3, size=len(groups))
    for train_idx, val_idx in data.make_cv(5).split(np.zeros(len(y)), y, groups):
        assert not set(groups[train_idx]) & set(groups[val_idx])


def test_reduction_grid_sizes_lda_to_classes():
    grid = models.reduction_param_grid(n_classes=6)
    lda = [g for g in grid if "LinearDiscriminant" in type(g["reduce_dim"][0]).__name__]
    assert lda[0]["reduce_dim__n_components"] == [5]
    assert all(g["corr__threshold"] == [0.9, 0.95] for g in grid)


def test_pipeline_fits_and_evaluates_in_original_label_space():
    rng = np.random.default_rng(0)
    y_raw = np.repeat([1, 2, 3], 30)
    X = pd.DataFrame(rng.normal(size=(90, 8)) + y_raw[:, None], columns=[f"f{i}" for i in range(8)])
    X["f8"] = X["f0"] * 1.0001  # highly correlated column for the filter to drop
    encoder = LabelEncoder().fit(y_raw)
    pipe = models.build_pipeline("xgb").set_params(classify__n_estimators=20)
    pipe.fit(X, encoder.transform(y_raw))
    assert pipe.named_steps["corr"].features_to_drop_
    result = train.evaluate(pipe, X, y_raw, encoder)
    assert list(result["confusion_matrix"].index) == [1, 2, 3]
    assert result["accuracy"] > 0.9
