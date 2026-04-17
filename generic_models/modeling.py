"""Standalone model registry for the generic detector repository."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def build_models(random_state: int = 42, *, selected_models: Iterable[str] | None = None) -> dict[str, object]:
    """Build the tabular model suite used by the generic detector."""

    factories: dict[str, Callable[[], object]] = {
        "logistic_regression": lambda: Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced")),
            ]
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight="balanced",
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight="balanced",
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.06,
            max_iter=300,
            random_state=random_state,
        ),
        "calibrated_svm": lambda: Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(class_weight="balanced", dual="auto", random_state=random_state),
                        cv=3,
                    ),
                ),
            ]
        ),
    }
    factories.update(_optional_model_factories(random_state=random_state))

    requested = [name.strip() for name in selected_models] if selected_models else list(factories)
    missing = [name for name in requested if name not in factories]
    if missing:
        available = ", ".join(sorted(factories))
        raise ValueError(f"Unknown or unavailable models requested: {missing}. Available models: {available}")
    return {name: factories[name]() for name in requested}


def _optional_model_factories(random_state: int) -> dict[str, Callable[[], object]]:
    factories: dict[str, Callable[[], object]] = {}

    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBClassifier

        factories["xgboost"] = lambda: XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
        )

    if importlib.util.find_spec("lightgbm") is not None:
        from lightgbm import LGBMClassifier

        factories["lightgbm"] = lambda: LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            class_weight="balanced",
            verbosity=-1,
        )

    if importlib.util.find_spec("catboost") is not None:
        from catboost import CatBoostClassifier

        factories["catboost"] = lambda: CatBoostClassifier(
            loss_function="Logloss",
            auto_class_weights="Balanced",
            depth=6,
            learning_rate=0.05,
            iterations=300,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        )

    return factories
