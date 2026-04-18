"""Train a v3 interaction model on expanded robustness scenarios.

This creates a new bundle beside the old generic and v2 bundles. It does not
replace or mutate the older models.
"""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from generic_models.evaluate_interaction_robustness import (
    ID_COLUMNS,
    build_feature_frame,
    build_robustness_sessions,
    dataframe_block,
    leave_one_site_out_predictions,
    metrics_dict,
)
from generic_models.site_catalog import build_all_generic_sites, get_websites
from generic_models.train_generic_models import predict_proba, tune_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train interaction-aware generic v3 detector")
    parser.add_argument("--artifacts-dir", default="generic_models/artifacts")
    parser.add_argument("--sessions-per-scenario", type=int, default=18)
    parser.add_argument("--random-state", type=int, default=27182)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    model_dir = artifacts_dir / "models"
    report_dir = artifacts_dir / "reports"
    data_dir = artifacts_dir / "data"
    for directory in (model_dir, report_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    sites = build_all_generic_sites()
    robust_sessions = build_robustness_sessions(
        specs=get_websites(),
        sites=sites,
        sessions_per_scenario=args.sessions_per_scenario,
        random_state=args.random_state,
    )
    telemetry_df = pd.DataFrame([event for item in robust_sessions for event in item.telemetry])
    feature_df = build_feature_frame(robust_sessions, sites=sites, telemetry_df=telemetry_df)
    feature_df.to_csv(data_dir / "interaction_v3_training_features.csv", index=False)

    feature_columns = [column for column in feature_df.columns if column not in ID_COLUMNS]
    threshold = tune_global_threshold(feature_df, feature_columns, random_state=args.random_state)
    cv_scores, cv_predictions, cv_labels = leave_one_site_out_predictions(feature_df, feature_columns, random_state=args.random_state)
    cv_metrics = metrics_dict(cv_labels, cv_predictions, cv_scores)

    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.055,
        l2_regularization=0.035,
        min_samples_leaf=14,
        random_state=args.random_state,
    )
    model.fit(feature_df[feature_columns], (feature_df["label"] == "bot").astype(int))
    bundle = {
        "bundle_type": "generic_website_detector_v3",
        "model_name": "interaction_v3",
        "feature_columns": feature_columns,
        "model": model,
        "threshold": threshold,
        "training_scope": "generic_graph_navigation_plus_interaction_robustness_scenarios",
        "notes": (
            "V3 is trained beside v2 on expanded robustness scenarios including "
            "BFS, DFS, random walk, coverage-greedy, bursty timing, browser mouse-noise, "
            "direct-goto, and click-based bots plus difficult human browsing variants."
        ),
    }
    bundle_path = model_dir / "interaction_v3_generic_bundle.pkl"
    with bundle_path.open("wb") as handle:
        pickle.dump(bundle, handle)

    cv_report = pd.DataFrame([{"model_name": "interaction_v3", "threshold": threshold, **cv_metrics}])
    cv_report.to_csv(report_dir / "interaction_v3_leave_site_out_metrics.csv", index=False)
    write_summary(
        report_dir / "interaction_v3_summary.md",
        bundle_path=bundle_path,
        feature_count=len(feature_columns),
        threshold=threshold,
        cv_report=cv_report,
        training_rows=len(feature_df),
        sessions=len(robust_sessions),
    )
    print(f"Saved v3 model bundle: {bundle_path.resolve()}")
    print(cv_report.to_string(index=False))


def tune_global_threshold(feature_df: pd.DataFrame, feature_columns: list[str], *, random_state: int) -> float:
    val_df = feature_df.groupby(["site_id", "scenario"], group_keys=False).sample(frac=0.25, random_state=random_state)
    train_df = feature_df.drop(index=val_df.index)
    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.055,
        l2_regularization=0.035,
        min_samples_leaf=14,
        random_state=random_state,
    )
    model.fit(train_df[feature_columns], (train_df["label"] == "bot").astype(int))
    scores = predict_proba(model, val_df[feature_columns])
    return tune_threshold((val_df["label"] == "bot").astype(int).to_numpy(), scores)


def write_summary(
    path: Path,
    *,
    bundle_path: Path,
    feature_count: int,
    threshold: float,
    cv_report: pd.DataFrame,
    training_rows: int,
    sessions: int,
) -> None:
    lines = [
        "# Interaction-Aware Generic V3 Model",
        "",
        f"- Bundle: `{bundle_path}`",
        f"- Threshold: `{threshold:.2f}`",
        f"- Feature count: `{feature_count}`",
        f"- Training sessions: `{sessions}`",
        f"- Training feature rows: `{training_rows}`",
        "- Old generic and v2 model bundles are not replaced.",
        "- V3 specifically adds robustness coverage for click-based systematic bots and difficult human variants.",
        "",
        "## Leave-One-Site-Out Metrics",
        "",
        dataframe_block(cv_report),
        "",
        "## Why V3 Exists",
        "",
        "The v2 model caught direct-goto and browser mouse-noise bots, but the robustness evaluation showed that a click-based systematic bot could look human-like in navigation provenance alone. V3 keeps the interaction features and strengthens graph/timing/systematic traversal learning with the expanded robustness scenarios.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
