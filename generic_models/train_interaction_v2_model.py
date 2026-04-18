"""Train an interaction-aware v2 generic detector without replacing old models."""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from generic_models.generic_admin_panel import INTERACTION_V2_FEATURE_COLUMNS
from generic_models.train_generic_models import (
    PREFIXES,
    build_prefix_feature_frame,
    classification_metrics,
    generate_multisite_dataset,
    predict_proba,
    tune_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the interaction-aware generic v2 detector")
    parser.add_argument("--artifacts-dir", default="generic_models/artifacts")
    parser.add_argument("--num-sites", type=int, default=72)
    parser.add_argument("--sessions-per-site", type=int, default=90)
    parser.add_argument("--random-state", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    model_dir = artifacts_dir / "models"
    report_dir = artifacts_dir / "reports"
    data_dir = artifacts_dir / "data"
    for directory in (model_dir, report_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.random_state)
    sessions, site_summary, sites = generate_multisite_dataset(
        num_sites=args.num_sites,
        sessions_per_site=args.sessions_per_site,
        rng=rng,
    )
    df = build_prefix_feature_frame(sessions, sites=sites, prefixes=PREFIXES)
    df = add_synthetic_interaction_features(df, random_state=args.random_state)
    df.to_csv(data_dir / "interaction_v2_prefix_features.csv", index=False)
    site_summary.to_csv(data_dir / "interaction_v2_site_summary.csv", index=False)

    site_ids = sorted(df["site_id"].unique())
    split_rng = random.Random(args.random_state)
    split_rng.shuffle(site_ids)
    n_sites = len(site_ids)
    train_sites = set(site_ids[: int(n_sites * 0.60)])
    val_sites = set(site_ids[int(n_sites * 0.60) : int(n_sites * 0.80)])
    test_sites = set(site_ids[int(n_sites * 0.80) :])

    train_df = df[df["site_id"].isin(train_sites)].copy()
    val_df = df[df["site_id"].isin(val_sites)].copy()
    test_df = df[df["site_id"].isin(test_sites)].copy()
    train_val_df = df[df["site_id"].isin(train_sites | val_sites)].copy()
    feature_columns = [col for col in df.columns if col not in {"session_id", "site_id", "family", "label"}]

    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.055,
        l2_regularization=0.03,
        min_samples_leaf=24,
        random_state=args.random_state,
    )
    model.fit(train_df[feature_columns], (train_df["label"] == "bot").astype(int))
    validation_scores = predict_proba(model, val_df[feature_columns])
    threshold = tune_threshold((val_df["label"] == "bot").astype(int).to_numpy(), validation_scores)

    final_model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.055,
        l2_regularization=0.03,
        min_samples_leaf=24,
        random_state=args.random_state,
    )
    final_model.fit(train_val_df[feature_columns], (train_val_df["label"] == "bot").astype(int))
    test_scores = predict_proba(final_model, test_df[feature_columns])
    test_labels = (test_df["label"] == "bot").astype(int).to_numpy()
    test_predictions = (test_scores >= threshold).astype(int)
    metrics = classification_metrics(test_labels, test_predictions, test_scores)

    prefix_rows = []
    scored_test = test_df.assign(score=test_scores, pred=test_predictions)
    for prefix_len, group in scored_test.groupby("prefix_len"):
        y_true = (group["label"] == "bot").astype(int).to_numpy()
        prefix_rows.append(
            {
                "model_name": "interaction_v2",
                "prefix_len": int(prefix_len),
                **classification_metrics(y_true, group["pred"].to_numpy(), group["score"].to_numpy()),
            }
        )
    prefix_report = pd.DataFrame(prefix_rows).sort_values("prefix_len")
    prefix_report.to_csv(report_dir / "interaction_v2_prefix_metrics.csv", index=False)
    pd.DataFrame([{ "model_name": "interaction_v2", "threshold": threshold, **metrics }]).to_csv(
        report_dir / "interaction_v2_leaderboard.csv",
        index=False,
    )

    bundle = {
        "bundle_type": "generic_website_detector_v2",
        "model_name": "interaction_v2",
        "feature_columns": feature_columns,
        "model": final_model,
        "threshold": threshold,
        "training_scope": "multi_site_graph_navigation_plus_interaction_telemetry",
        "notes": (
            "V2 generic model trained beside the original bundles. It adds defensive "
            "browser-interaction provenance features: click-before-navigation, href match, "
            "destination-without-precursor, page-type dwell residuals, transition likelihood, "
            "and within-page event density."
        ),
    }
    bundle_path = model_dir / "interaction_v2_generic_bundle.pkl"
    with bundle_path.open("wb") as handle:
        pickle.dump(bundle, handle)

    write_summary(
        report_dir / "interaction_v2_summary.md",
        metrics=metrics,
        prefix_report=prefix_report,
        threshold=threshold,
        feature_columns=feature_columns,
        bundle_path=bundle_path,
    )
    print(f"Saved v2 model bundle: {bundle_path.resolve()}")
    print(f"Threshold: {threshold:.2f}")
    print(pd.DataFrame([{**metrics}]).to_string(index=False))
    print(prefix_report.to_string(index=False))


def add_synthetic_interaction_features(df: pd.DataFrame, *, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    for record in df.to_dict("records"):
        label = str(record["label"])
        family = str(record["family"])
        prefix_len = max(1.0, float(record["prefix_len"]))
        features = synthesize_interaction_row(record, label=label, family=family, prefix_len=prefix_len, rng=rng)
        rows.append(features)
    interaction_df = pd.DataFrame(rows)
    for column in INTERACTION_V2_FEATURE_COLUMNS:
        if column not in interaction_df:
            interaction_df[column] = 0.0
    return pd.concat([df.reset_index(drop=True), interaction_df[INTERACTION_V2_FEATURE_COLUMNS].reset_index(drop=True)], axis=1)


def synthesize_interaction_row(
    record: dict[str, object],
    *,
    label: str,
    family: str,
    prefix_len: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    transitions = max(1.0, prefix_len - 1.0)
    graph_score = float(record.get("navigation_entropy_generic_score", 0.5) or 0.5)
    raw_dwell_cv = float(record.get("inter_hop_time_cv", 0.0) or 0.0)
    base_nll = 1.10 + float(record.get("far_jump_ratio", 0.0) or 0.0) * 1.35 + float(record.get("deterministic_branch_ratio", 0.0) or 0.0) * 0.55

    if label == "human":
        click_ratio = _clip(rng.normal(0.78, 0.13))
        match_ratio = _clip(click_ratio - abs(rng.normal(0.05, 0.06)))
        free_ratio = _clip(rng.normal(0.09, 0.08))
        no_precursor = _clip(1.0 - match_ratio + rng.normal(0.03, 0.04))
        meaningful_density = max(0.0, rng.normal(4.6, 1.15))
        telemetry_missing = 0.0 if rng.random() > 0.015 else 1.0
        dwell_residual = _clip(rng.normal(0.18, 0.10))
        transition_nll = max(0.05, rng.normal(base_nll * 0.78, 0.24))
        uniform_dwell = _clip(rng.normal(0.20, 0.14))
    elif family in {"browser_like", "noisy"}:
        click_ratio = _clip(rng.normal(0.12, 0.12))
        match_ratio = _clip(rng.normal(0.05, 0.07))
        free_ratio = _clip(rng.normal(0.42, 0.18))
        no_precursor = _clip(rng.normal(0.82, 0.13))
        meaningful_density = max(0.0, rng.normal(1.15, 0.65))
        telemetry_missing = 0.0 if rng.random() > 0.10 else 1.0
        dwell_residual = _clip(rng.normal(0.66, 0.18))
        transition_nll = max(0.05, rng.normal(base_nll * 1.18 + (1.0 - graph_score) * 0.55, 0.36))
        uniform_dwell = _clip(rng.normal(0.68, 0.20))
    else:
        click_ratio = _clip(rng.normal(0.03, 0.05))
        match_ratio = _clip(rng.normal(0.01, 0.03))
        free_ratio = _clip(rng.normal(0.83, 0.13))
        no_precursor = _clip(rng.normal(0.96, 0.05))
        meaningful_density = max(0.0, rng.normal(0.35, 0.32))
        telemetry_missing = 1.0 if rng.random() < 0.45 else 0.0
        dwell_residual = _clip(rng.normal(0.80, 0.14))
        transition_nll = max(0.05, rng.normal(base_nll * 1.35 + 0.42, 0.34))
        uniform_dwell = _clip(rng.normal(0.78, 0.16))

    semantic_anomaly = _clip(transition_nll / 4.0)
    low_density = 1.0 - min(1.0, meaningful_density / 3.0)
    telemetry_anomaly = _clip(
        0.30 * semantic_anomaly
        + 0.20 * dwell_residual
        + 0.25 * free_ratio
        + 0.15 * no_precursor
        + 0.10 * low_density
    )
    if telemetry_missing >= 1.0:
        telemetry_anomaly = max(telemetry_anomaly, 0.72)

    event_count = 0.0 if telemetry_missing else max(0.0, meaningful_density * prefix_len + rng.normal(4.0, 3.0))
    mousemove_count = 0.0 if telemetry_missing else max(0.0, event_count * rng.uniform(0.25, 0.55))
    scroll_count = 0.0 if telemetry_missing else max(0.0, event_count * rng.uniform(0.08, 0.28))
    click_count = 0.0 if telemetry_missing else max(0.0, click_ratio * transitions + rng.normal(0.2, 0.35))
    pointer_count = 0.0 if telemetry_missing else max(0.0, click_count * rng.uniform(1.2, 2.2))
    focus_count = 0.0 if telemetry_missing else max(0.0, rng.poisson(0.25 if label == "bot" else 0.9))
    keydown_count = 0.0 if telemetry_missing else max(0.0, rng.poisson(0.2 if label == "bot" else 0.8))
    page_load_count = 0.0 if telemetry_missing else prefix_len

    return {
        "telemetry_event_count": float(event_count),
        "telemetry_events_per_request": float(event_count / prefix_len),
        "telemetry_missing_ratio": float(telemetry_missing),
        "mousemove_count": float(mousemove_count),
        "scroll_event_count": float(scroll_count),
        "scroll_burst_count": float(0.0 if telemetry_missing else max(0.0, scroll_count * rng.uniform(0.25, 0.75))),
        "click_count": float(click_count),
        "pointer_event_count": float(pointer_count),
        "focus_count": float(focus_count),
        "keydown_count": float(keydown_count),
        "meaningful_event_count": float(0.0 if telemetry_missing else meaningful_density * prefix_len),
        "meaningful_event_density": float(0.0 if telemetry_missing else meaningful_density),
        "time_to_first_interaction_seconds": float(0.0 if telemetry_missing else max(0.0, rng.normal(2.8 if label == "human" else 1.2, 1.0))),
        "last_interaction_to_navigation_mean_seconds": float(0.0 if telemetry_missing else max(0.0, rng.normal(1.8 if label == "human" else 0.28, 0.45))),
        "click_precursor_ratio": float(0.0 if telemetry_missing else click_ratio),
        "click_href_match_ratio": float(0.0 if telemetry_missing else match_ratio),
        "interaction_free_navigation_ratio": float(1.0 if telemetry_missing else free_ratio),
        "destination_without_precursor_ratio": float(1.0 if telemetry_missing else no_precursor),
        "hover_on_target_link_ratio": float(0.0 if telemetry_missing else _clip(match_ratio + rng.normal(0.12 if label == "human" else 0.02, 0.08))),
        "page_load_event_count": float(page_load_count),
        "page_type_dwell_residual_mean": float(dwell_residual),
        "page_type_dwell_residual_max": float(_clip(dwell_residual + abs(rng.normal(0.12, 0.08)))),
        "page_type_dwell_cv": float(max(0.0, raw_dwell_cv + rng.normal(0.12 if label == "human" else -0.05, 0.16))),
        "uniform_dwell_score": float(uniform_dwell),
        "transition_human_nll": float(transition_nll),
        "semantic_transition_anomaly_score": float(semantic_anomaly),
        "telemetry_anomaly_score": float(telemetry_anomaly),
    }


def _clip(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def write_summary(
    path: Path,
    *,
    metrics: dict[str, float],
    prefix_report: pd.DataFrame,
    threshold: float,
    feature_columns: list[str],
    bundle_path: Path,
) -> None:
    lines = [
        "# Interaction-Aware Generic V2 Model",
        "",
        f"- Bundle: `{bundle_path}`",
        f"- Threshold: `{threshold:.2f}`",
        f"- Feature count: `{len(feature_columns)}`",
        "- Old generic model bundles are not replaced.",
        "- New signals focus on how navigation is produced, not only where it goes.",
        "- These metrics are a synthetic v2 sanity benchmark; real confidence still depends on collected human telemetry.",
        "",
        "## Holdout Metrics",
        "",
        "```text",
        pd.DataFrame([{**metrics}]).to_string(index=False),
        "```",
        "",
        "## Prefix Metrics",
        "",
        "```text",
        prefix_report.to_string(index=False),
        "```",
        "",
        "## Added V2 Feature Families",
        "",
        "- Navigation provenance: click precursor, clicked destination match, destination without precursor.",
        "- Within-page activity: pointer, scroll, focus, keydown, meaningful event density.",
        "- Page-type-aware timing: dwell residuals and uniform dwell score.",
        "- Human transition likelihood: semantic transition negative log likelihood.",
        "- Combined telemetry anomaly score for browser-like direct-navigation bots.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
