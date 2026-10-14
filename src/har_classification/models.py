"""Estimators, pipelines and hyper-parameter grids."""

from feature_engine.selection import SmartCorrelatedSelection
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.random_projection import GaussianRandomProjection
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42
MODEL_NAMES = ("rf", "dt", "xgb")
SCORING = "f1_macro"
# Matthews correlation is included because it stays informative under class imbalance.
BASELINE_SCORING = ["balanced_accuracy", "f1_macro", "matthews_corrcoef"]


def baseline_models() -> dict:
    """Untuned models used to compare feature sets."""
    return {
        "Decision tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "Random forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
    }


def make_classifier(name: str):
    if name == "rf":
        return RandomForestClassifier(random_state=RANDOM_STATE)
    if name == "dt":
        return DecisionTreeClassifier(random_state=RANDOM_STATE)
    if name == "xgb":
        return XGBClassifier(random_state=RANDOM_STATE, tree_method="hist")
    raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_NAMES}")


def build_pipeline(name: str, memory=None) -> Pipeline:
    """correlation filter -> dimensionality reduction (searched) -> classifier."""
    return Pipeline(
        [
            ("corr", SmartCorrelatedSelection(method="pearson", selection_method="variance")),
            ("reduce_dim", "passthrough"),
            ("classify", make_classifier(name)),
        ],
        memory=memory,
    )


def reduction_param_grid(n_classes: int) -> list[dict]:
    """Stage 1 grid: correlation threshold x reduction method x output size."""
    corr = {"corr__threshold": [0.9, 0.95]}
    grids = [
        {
            "reduce_dim": [
                PCA(random_state=RANDOM_STATE),
                FactorAnalysis(random_state=RANDOM_STATE),
                GaussianRandomProjection(random_state=RANDOM_STATE),
                FastICA(whiten="unit-variance", max_iter=500, random_state=RANDOM_STATE),
            ],
            "reduce_dim__n_components": [100, 150, 200],
        },
        {
            # LDA can project to at most n_classes - 1 dimensions.
            "reduce_dim": [LinearDiscriminantAnalysis()],
            "reduce_dim__n_components": [n_classes - 1],
        },
        {
            "reduce_dim": [SelectFromModel(ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE))],
            "reduce_dim__max_features": [100, 200, 300],
        },
    ]
    return [{**corr, **grid} for grid in grids]


# Stage 2: tune the classifier on top of the best stage-1 pipeline.
CLASSIFIER_PARAM_GRIDS = {
    "rf": {
        "classify__max_features": ["sqrt", "log2", 0.3, 0.5],
        "classify__min_samples_leaf": [3, 5, 7, 9],
    },
    "dt": {
        "classify__max_depth": [10, 20, None],
        "classify__min_samples_leaf": [3, 5, 7, 9],
        "classify__ccp_alpha": [0.0, 1e-3],
    },
    "xgb": {
        "classify__learning_rate": [0.05, 0.1],
        "classify__n_estimators": [300, 600],
        "classify__max_depth": [4, 6],
        "classify__colsample_bytree": [0.4, 0.8],
    },
}
