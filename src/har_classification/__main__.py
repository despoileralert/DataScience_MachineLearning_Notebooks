"""Train and evaluate the activity-recognition models.

Usage:
    python -m src.har_classification [--models rf dt xgb] [--wavelet] [--skip-baseline]
    python -m src.har_classification --train-csv features_train.csv --test-csv features_test.csv

Expect a long run: the stage-1 search fits hundreds of pipelines per model.
"""

import argparse
import json
from pathlib import Path

import joblib
from sklearn.preprocessing import LabelEncoder

from src.har_classification import data, models, train
from src.paths import ARTIFACTS_DIR, PROJECT_ROOT, ensure_dir


def _parse_args(argv):
    parser = argparse.ArgumentParser(prog="har_classification", description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=data.DEFAULT_DATA_DIR,
                        help="UCI HAR directory (train/, test/, features.txt)")
    parser.add_argument("--train-csv", type=Path, help="feature CSV (subject, _, label, features...)")
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--wavelet", action="store_true",
                        help="append wavelet features computed from the Inertial Signals")
    parser.add_argument("--models", nargs="+", choices=models.MODEL_NAMES, default=list(models.MODEL_NAMES))
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / ".cache" / "har_pipeline",
                        help="joblib cache for fitted pipeline steps")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS_DIR / "har_classification")
    args = parser.parse_args(argv)
    if bool(args.train_csv) != bool(args.test_csv):
        parser.error("--train-csv and --test-csv must be given together")
    if args.train_csv and args.wavelet:
        parser.error("--wavelet needs the UCI HAR Inertial Signals; it cannot be combined with CSV input")
    return args


def main(argv=None) -> None:
    args = _parse_args(argv)
    out = ensure_dir(args.output_dir)

    if args.train_csv:
        dataset = data.load_feature_csvs(args.train_csv, args.test_csv)
    else:
        dataset = data.load_uci_har(args.data_dir)
        if args.wavelet:
            from src.har_classification.wavelet import add_wavelet_features
            dataset = add_wavelet_features(dataset, args.data_dir)
    print(f"Train {dataset.X_train.shape}, test {dataset.X_test.shape}, {dataset.n_classes} classes")

    # XGBoost needs labels 0..K-1; encode once for every model and decode for reporting.
    encoder = LabelEncoder().fit(dataset.y_train)
    y_train = encoder.transform(dataset.y_train)
    groups = dataset.groups_train
    cv = data.make_cv(args.n_splits)
    print(data.describe_folds(cv, dataset.X_train, y_train, groups).to_string(index=False))

    if not args.skip_baseline:
        featuresets = {"all_features": dataset.X_train}
        original = dataset.without_wavelet()
        if original.shape[1] < dataset.X_train.shape[1]:
            featuresets = {"without_wavelet": original, **featuresets}
        baseline = train.compare_feature_sets(featuresets, y_train, groups, cv, verbose=args.verbose)
        baseline.to_csv(out / "baseline_cv.csv", index=False)
        print(baseline.to_string(index=False))

    summary = {}
    for name in args.models:
        print(f"\n=== {name}: stage 1 (correlation filter + dimensionality reduction) ===")
        stage1 = train.search_reduction(
            name, dataset.X_train, y_train, groups, cv, dataset.n_classes,
            memory=str(args.cache_dir), n_jobs=args.n_jobs, verbose=args.verbose,
        )
        print(stage1.best_params_, f"CV {models.SCORING}={stage1.best_score_:.4f}")

        print(f"=== {name}: stage 2 (classifier hyper-parameters) ===")
        stage2 = train.tune_classifier(name, stage1.best_estimator_, dataset.X_train, y_train, groups, cv,
                                       n_jobs=args.n_jobs, verbose=args.verbose)
        print(stage2.best_params_, f"CV {models.SCORING}={stage2.best_score_:.4f}")

        result = train.evaluate(stage2.best_estimator_, dataset.X_test, dataset.y_test, encoder)
        print(result["classification_report"])
        print(result["confusion_matrix"])

        joblib.dump({"model": stage2.best_estimator_, "label_encoder": encoder}, out / f"{name}_model.joblib")
        result["confusion_matrix"].to_csv(out / f"{name}_confusion_matrix.csv")
        summary[name] = {
            "stage1_best_params": {k: str(v) for k, v in stage1.best_params_.items()},
            "stage1_cv_score": stage1.best_score_,
            "stage2_best_params": {k: str(v) for k, v in stage2.best_params_.items()},
            "stage2_cv_score": stage2.best_score_,
            "test": {k: result[k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")},
        }

    (out / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
