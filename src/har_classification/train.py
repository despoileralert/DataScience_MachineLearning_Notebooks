"""Cross-validation, two-stage hyper-parameter search and test-set evaluation."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.experimental import enable_halving_search_cv  # noqa: F401  (registers HalvingGridSearchCV)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder

from src.har_classification import models


def compare_feature_sets(featuresets: dict[str, pd.DataFrame], y, groups, cv, verbose: int = 0) -> pd.DataFrame:
    """Mean CV scores of the untuned baseline models on each feature set."""
    rows = []
    for feature_name, X in featuresets.items():
        for model_name, estimator in models.baseline_models().items():
            scores = cross_validate(
                estimator, X, y, groups=groups, cv=cv,
                scoring=models.BASELINE_SCORING, verbose=verbose,
            )
            rows.append({"featureset": feature_name, "model": model_name,
                         **{k: np.mean(v) for k, v in scores.items()}})
    return pd.DataFrame(rows).round(3)


def search_reduction(name: str, X, y, groups, cv, n_classes: int,
                     memory=None, n_jobs=None, verbose: int = 0):
    """Stage 1: choose correlation threshold and dimensionality reduction.

    Tree models use successive halving to keep the search affordable; XGBoost
    uses a full grid search.
    """
    pipeline = models.build_pipeline(name, memory=memory)
    grid = models.reduction_param_grid(n_classes)
    common = dict(param_grid=grid, cv=cv, scoring=models.SCORING, refit=True,
                  n_jobs=n_jobs, verbose=verbose)
    if name == "xgb":
        search = GridSearchCV(pipeline, **common)
    else:
        # "exhaust" starts from the largest sample budget that still finishes on
        # the full data. The default starts with a few dozen samples, too few to
        # fit 100-200 components, so PCA/FA/ICA were eliminated before scoring.
        search = HalvingGridSearchCV(pipeline, min_resources="exhaust",
                                     random_state=models.RANDOM_STATE, **common)
    return search.fit(X, y, groups=groups)


def tune_classifier(name: str, pipeline, X, y, groups, cv, n_jobs=None, verbose: int = 0) -> GridSearchCV:
    """Stage 2: tune classifier hyper-parameters on the best stage-1 pipeline."""
    search = GridSearchCV(
        clone(pipeline), param_grid=models.CLASSIFIER_PARAM_GRIDS[name],
        cv=cv, scoring=models.SCORING, n_jobs=n_jobs, verbose=verbose,
    )
    return search.fit(X, y, groups=groups)


def evaluate(estimator, X_test, y_test, encoder: LabelEncoder) -> dict:
    """Macro-averaged test metrics, reported in the original label space."""
    y_pred = encoder.inverse_transform(estimator.predict(X_test))
    labels = encoder.classes_
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "classification_report": classification_report(y_test, y_pred, labels=labels),
        "confusion_matrix": pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=labels), index=labels, columns=labels
        ),
    }
