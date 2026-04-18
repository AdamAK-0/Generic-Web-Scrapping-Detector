"""Robustness evaluation for the interaction-aware generic detector.

The script creates a varied, local-only evaluation set from the four generic
websites. It does not replace trained model bundles. It evaluates the selected
bundle and trains temporary ablation models to measure which feature families
matter under leave-one-site-out testing.
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from generic_models.generic_admin_panel import INTERACTION_V2_FEATURE_COLUMNS, extract_interaction_v2_features
from generic_models.site_catalog import PageSpec, WebsiteSpec, build_all_generic_sites, get_websites, root_path
from generic_models.train_generic_models import GenericSession, GenericSite, extract_generic_features, predict_proba, tune_threshold


HUMAN_SCENARIOS = [
    "human_normal",
    "human_fast",
    "human_slow",
    "human_short",
    "human_revisit_heavy",
    "human_scroll_heavy",
    "human_low_scroll",
    "human_product_then_leave",
    "human_search_cart_contact",
    "human_unusual_pattern",
]

BOT_SCENARIOS = [
    "bfs_crawler",
    "dfs_crawler",
    "random_walk_bot",
    "coverage_greedy_bot",
    "bursty_timing_bot",
    "browser_mouse_noise_bot",
    "direct_goto_bot",
    "click_based_bot",
]

SCENARIO_NOTES = {
    "human_normal": "Task-like browsing with clicks and moderate dwell.",
    "human_fast": "Fast human with real click precursors.",
    "human_slow": "Slow human with long reading pauses.",
    "human_short": "Only 3-4 pages, a key early-detection edge case.",
    "human_revisit_heavy": "Human repeatedly backtracks and revisits pages.",
    "human_scroll_heavy": "Lots of reading/scrolling before navigation.",
    "human_low_scroll": "Almost no scrolling but still click-driven.",
    "human_product_then_leave": "Short product/detail visit then exit.",
    "human_search_cart_contact": "Uses utility/cart/contact-style paths.",
    "human_unusual_pattern": "Weird but still causal click-driven human path.",
    "bfs_crawler": "Breadth-first systematic crawl.",
    "dfs_crawler": "Depth-first systematic crawl.",
    "random_walk_bot": "Graph-valid random walker.",
    "coverage_greedy_bot": "Chooses unvisited/high-coverage pages.",
    "bursty_timing_bot": "Alternates very fast bursts with pauses.",
    "browser_mouse_noise_bot": "Browser-like mouse and scroll noise, but no causal click chain.",
    "direct_goto_bot": "Direct navigation to valid URLs without click precursors.",
    "click_based_bot": "Uses matching clicks but follows systematic coverage behavior.",
}

TIMING_COLUMNS = {
    "inter_hop_time_mean",
    "inter_hop_time_std",
    "inter_hop_time_cv",
    "inter_hop_burstiness",
    "low_latency_ratio",
}

ID_COLUMNS = {"session_id", "site_id", "scenario", "label", "score", "prediction"}


@dataclass(frozen=True)
class RobustSession:
    session: GenericSession
    telemetry: list[dict[str, Any]]
    scenario: str
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate interaction-aware generic detector robustness")
    parser.add_argument("--artifacts-dir", default="generic_models/artifacts")
    parser.add_argument("--model-name", default="interaction_v2")
    parser.add_argument("--sessions-per-scenario", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=31415)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    model_dir = artifacts_dir / "models"
    report_dir = artifacts_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    sites = build_all_generic_sites()
    specs = get_websites()
    robust_sessions = build_robustness_sessions(
        specs=specs,
        sites=sites,
        sessions_per_scenario=args.sessions_per_scenario,
        random_state=args.random_state,
    )
    telemetry_df = pd.DataFrame([event for item in robust_sessions for event in item.telemetry])
    feature_df = build_feature_frame(robust_sessions, sites=sites, telemetry_df=telemetry_df)
    feature_df.to_csv(report_dir / "robustness_feature_rows.csv", index=False)

    bundle = load_bundle(model_dir / f"{args.model_name}_generic_bundle.pkl")
    model_report, scored_df = evaluate_bundle(feature_df, bundle=bundle, model_name=args.model_name)
    scenario_report = summarize_by_scenario(scored_df)
    ablation_report = run_ablation(feature_df, random_state=args.random_state)
    importance_report = feature_importance(feature_df, bundle=bundle, random_state=args.random_state)
    sample_table = build_sample_table(scored_df)

    output_prefix = f"robustness_{args.model_name}"
    model_report.to_csv(report_dir / f"{output_prefix}_model_metrics.csv", index=False)
    scenario_report.to_csv(report_dir / f"{output_prefix}_scenario_metrics.csv", index=False)
    ablation_report.to_csv(report_dir / f"{output_prefix}_ablation_metrics.csv", index=False)
    importance_report.to_csv(report_dir / f"{output_prefix}_feature_importance.csv", index=False)
    sample_table.to_csv(report_dir / f"{output_prefix}_sample_table.csv", index=False)
    write_summary(
        report_dir / f"{output_prefix}_summary.md",
        model_report=model_report,
        scenario_report=scenario_report,
        ablation_report=ablation_report,
        importance_report=importance_report,
        sample_table=sample_table,
        sessions=robust_sessions,
        feature_df=feature_df,
        model_name=args.model_name,
    )

    print(f"Robustness reports written to: {report_dir.resolve()}")
    print(model_report.to_string(index=False))
    print("\nScenario detection summary:")
    print(scenario_report[["scenario", "label", "sessions", "detection_rate", "mean_score"]].to_string(index=False))
    print("\nAblation summary:")
    print(ablation_report.to_string(index=False))


def build_robustness_sessions(
    *,
    specs: dict[str, WebsiteSpec],
    sites: dict[str, GenericSite],
    sessions_per_scenario: int,
    random_state: int,
) -> list[RobustSession]:
    rng = random.Random(random_state)
    rows: list[RobustSession] = []
    for spec in specs.values():
        site = sites[spec.site_id]
        for scenario in [*HUMAN_SCENARIOS, *BOT_SCENARIOS]:
            label = "human" if scenario.startswith("human_") else "bot"
            for index in range(sessions_per_scenario):
                paths = path_for_scenario(spec, site, scenario=scenario, rng=rng)
                timestamps = timestamps_for_scenario(site, paths, scenario=scenario, rng=rng, offset=len(rows) * 1000.0 + 1_800_000_000.0)
                session = GenericSession(
                    session_id=f"robust_{spec.site_id}_{scenario}_{index:03d}",
                    site_id=spec.site_id,
                    family=scenario,
                    label=label,
                    paths=paths,
                    timestamps=timestamps,
                )
                telemetry = telemetry_for_scenario(session, site, scenario=scenario, rng=rng)
                rows.append(RobustSession(session=session, telemetry=telemetry, scenario=scenario, note=SCENARIO_NOTES[scenario]))
    return rows


def path_for_scenario(spec: WebsiteSpec, site: GenericSite, *, scenario: str, rng: random.Random) -> list[str]:
    if scenario == "bfs_crawler":
        return cap_unique(bfs_paths(spec), rng.randint(14, 24))
    if scenario == "dfs_crawler":
        return cap_unique(dfs_paths(spec), rng.randint(14, 24))
    if scenario == "coverage_greedy_bot":
        return coverage_greedy_paths(spec, rng.randint(14, 26))
    if scenario == "direct_goto_bot":
        nodes = sorted(site.graph.nodes, key=lambda path: (-int(site.graph.nodes[path].get("depth", 0)), path))
        rng.shuffle(nodes)
        return [root_path(spec.site_id), *[node for node in nodes if node != root_path(spec.site_id)]][: rng.randint(10, 20)]
    if scenario == "click_based_bot":
        return coverage_greedy_paths(spec, rng.randint(12, 22))
    if scenario == "bursty_timing_bot":
        return random_walk_paths(spec, rng=rng, length=rng.randint(12, 22), revisit_bias=0.16, jump_bias=0.08)
    if scenario == "browser_mouse_noise_bot":
        return random_walk_paths(spec, rng=rng, length=rng.randint(10, 20), revisit_bias=0.20, jump_bias=0.08)
    if scenario == "random_walk_bot":
        return random_walk_paths(spec, rng=rng, length=rng.randint(11, 21), revisit_bias=0.28, jump_bias=0.18)

    if scenario == "human_short":
        return human_goal_path(spec, rng=rng, length=rng.randint(3, 4))
    if scenario == "human_product_then_leave":
        return human_product_then_leave(spec, rng=rng)
    if scenario == "human_search_cart_contact":
        return human_utility_path(spec, rng=rng, length=rng.randint(5, 9))
    if scenario == "human_revisit_heavy":
        return human_goal_path(spec, rng=rng, length=rng.randint(8, 15), revisit_bias=0.40)
    if scenario == "human_unusual_pattern":
        return human_goal_path(spec, rng=rng, length=rng.randint(5, 12), revisit_bias=0.22, shallow_jump_bias=0.18)
    return human_goal_path(spec, rng=rng, length=rng.randint(5, 14))


def timestamps_for_scenario(site: GenericSite, paths: list[str], *, scenario: str, rng: random.Random, offset: float) -> list[float]:
    timestamps = [offset]
    for path in paths[:-1]:
        category = str(site.graph.nodes[path].get("category", "listing")) if path in site.graph else "listing"
        expected = expected_dwell_seconds(category)
        if scenario == "human_fast":
            delta = max(1.0, expected * rng.uniform(0.16, 0.38))
        elif scenario == "human_slow":
            delta = expected * rng.uniform(1.10, 2.30)
        elif scenario == "human_short":
            delta = expected * rng.uniform(0.26, 0.70)
        elif scenario.startswith("human_"):
            delta = expected * rng.uniform(0.55, 1.25)
        elif scenario == "bursty_timing_bot":
            delta = rng.choice([rng.uniform(0.06, 0.22), rng.uniform(7.0, 18.0)])
        elif scenario in {"browser_mouse_noise_bot", "direct_goto_bot"}:
            delta = rng.uniform(2.0, 5.0)
        elif scenario == "click_based_bot":
            delta = rng.uniform(0.45, 1.80)
        else:
            delta = rng.uniform(0.05, 0.65)
        timestamps.append(timestamps[-1] + max(0.05, delta))
    return timestamps


def telemetry_for_scenario(session: GenericSession, site: GenericSite, *, scenario: str, rng: random.Random) -> list[dict[str, Any]]:
    telemetry: list[dict[str, Any]] = []
    browser_like = scenario in {"browser_mouse_noise_bot", "direct_goto_bot"}
    click_based_bot = scenario == "click_based_bot"
    has_client_telemetry = scenario.startswith("human_") or browser_like or click_based_bot
    if not has_client_telemetry:
        return telemetry
    for index, path in enumerate(session.paths):
        start = session.timestamps[index]
        end = session.timestamps[index + 1] if index + 1 < len(session.timestamps) else start + 1.0
        next_path = session.paths[index + 1] if index + 1 < len(session.paths) else ""
        telemetry.append(event(session, "page_load", start + 0.04, path, interactive_count=interactive_count(site, path), content_length=content_length(site, path)))

        if scenario.startswith("human_"):
            move_count = rng.randint(2, 8)
            scroll_count = human_scroll_count(scenario, rng)
            meaningful_extra = rng.randint(0, 3)
            for move_index in range(move_count):
                telemetry.append(event(session, "mousemove", start + 0.20 + move_index * 0.25, path, target_href=next_path if rng.random() < 0.35 else ""))
            for scroll_index in range(scroll_count):
                telemetry.append(event(session, "scroll", start + 0.70 + scroll_index * rng.uniform(0.35, 1.20), path, y=80 + scroll_index * 160, ratio=min(1.0, 0.1 + scroll_index * 0.12)))
            if is_utility_path(site, path) and rng.random() < 0.55:
                telemetry.append(event(session, "focus", min(end - 0.8, start + 1.0), path, tag="INPUT"))
                telemetry.append(event(session, "keydown", min(end - 0.6, start + 1.2), path, tag="INPUT"))
            for extra_index in range(meaningful_extra):
                telemetry.append(event(session, "pointerdown", min(end - 0.9, start + 1.5 + extra_index * 0.2), path, target_href=""))
            if next_path:
                click_time = max(start + 0.25, end - rng.uniform(0.35, 1.85))
                telemetry.extend(click_events(session, path, next_path, click_time))
        elif browser_like:
            for move_index in range(rng.randint(1, 4)):
                telemetry.append(event(session, "mousemove", start + 0.25 + move_index * 0.35, path))
            if rng.random() < 0.60:
                telemetry.append(event(session, "scroll", start + rng.uniform(0.75, 1.70), path, y=rng.randint(200, 700), ratio=rng.uniform(0.08, 0.62)))
        elif click_based_bot and next_path:
            telemetry.append(event(session, "mousemove", start + 0.18, path, target_href=next_path))
            telemetry.extend(click_events(session, path, next_path, max(start + 0.25, end - rng.uniform(0.10, 0.40))))
    return telemetry


def build_feature_frame(robust_sessions: list[RobustSession], *, sites: dict[str, GenericSite], telemetry_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in robust_sessions:
        session = item.session
        site = sites[session.site_id]
        prefix_len = min(len(session.paths), 20)
        features = extract_generic_features(session, site, prefix_len=prefix_len)
        features["prefix_len"] = float(prefix_len)
        features.update(extract_interaction_v2_features(session, site, prefix_len=prefix_len, telemetry_df=telemetry_df))
        rows.append(
            {
                "session_id": session.session_id,
                "site_id": session.site_id,
                "scenario": item.scenario,
                "label": session.label,
                **features,
            }
        )
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def evaluate_bundle(feature_df: pd.DataFrame, *, bundle: dict[str, Any], model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = list(bundle["feature_columns"])
    frame = pd.DataFrame([{column: float(row.get(column, 0.0)) for column in feature_columns} for row in feature_df.to_dict("records")])
    scores = predict_proba(bundle["model"], frame)
    threshold = float(bundle.get("threshold", 0.5))
    y_true = (feature_df["label"] == "bot").astype(int).to_numpy()
    predictions = (scores >= threshold).astype(int)
    scored_df = feature_df.assign(score=scores, prediction=predictions)
    row = {"model_name": model_name, "threshold": threshold, **metrics_dict(y_true, predictions, scores)}
    return pd.DataFrame([row]), scored_df


def summarize_by_scenario(scored_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, group in scored_df.groupby("scenario"):
        label = str(group["label"].iloc[0])
        y_true = (group["label"] == "bot").astype(int).to_numpy()
        pred = group["prediction"].astype(int).to_numpy()
        rows.append(
            {
                "scenario": scenario,
                "label": label,
                "sessions": len(group),
                "detection_rate": float(pred.mean()),
                "mean_score": float(group["score"].mean()),
                "min_score": float(group["score"].min()),
                "max_score": float(group["score"].max()),
                "errors": int((pred != y_true).sum()),
                "notes": SCENARIO_NOTES[scenario],
            }
        )
    return pd.DataFrame(rows).sort_values(["label", "scenario"]).reset_index(drop=True)


def run_ablation(feature_df: pd.DataFrame, *, random_state: int) -> pd.DataFrame:
    all_features = [column for column in feature_df.columns if column not in ID_COLUMNS]
    interaction = [column for column in INTERACTION_V2_FEATURE_COLUMNS if column in feature_df.columns]
    timing = [column for column in TIMING_COLUMNS if column in feature_df.columns]
    graph = [column for column in all_features if column not in set(interaction) | set(timing)]
    groups = {
        "graph_only": graph,
        "timing_only": ["prefix_len", *timing],
        "interaction_only": ["prefix_len", *interaction],
        "graph_plus_timing": sorted(set(graph + timing)),
        "graph_plus_interaction": sorted(set(graph + interaction)),
        "all_features": all_features,
    }
    rows = []
    for group_name, columns in groups.items():
        columns = [column for column in columns if column in feature_df.columns]
        scores, predictions, labels = leave_one_site_out_predictions(feature_df, columns, random_state=random_state)
        rows.append({"ablation": group_name, "feature_count": len(columns), **metrics_dict(labels, predictions, scores)})
    return pd.DataFrame(rows).sort_values(["f1", "roc_auc"], ascending=False).reset_index(drop=True)


def leave_one_site_out_predictions(feature_df: pd.DataFrame, columns: list[str], *, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_scores: list[float] = []
    all_predictions: list[int] = []
    all_labels: list[int] = []
    for fold_index, held_out_site in enumerate(sorted(feature_df["site_id"].unique())):
        train_pool = feature_df[feature_df["site_id"] != held_out_site].copy()
        test_df = feature_df[feature_df["site_id"] == held_out_site].copy()
        val_df = train_pool.groupby("scenario", group_keys=False).sample(frac=0.25, random_state=random_state + fold_index)
        train_df = train_pool.drop(index=val_df.index)
        model = HistGradientBoostingClassifier(max_iter=140, learning_rate=0.06, l2_regularization=0.04, min_samples_leaf=12, random_state=random_state + fold_index)
        model.fit(train_df[columns], (train_df["label"] == "bot").astype(int))
        validation_scores = predict_proba(model, val_df[columns])
        threshold = tune_threshold((val_df["label"] == "bot").astype(int).to_numpy(), validation_scores)
        scores = predict_proba(model, test_df[columns])
        predictions = (scores >= threshold).astype(int)
        labels = (test_df["label"] == "bot").astype(int).to_numpy()
        all_scores.extend(scores.tolist())
        all_predictions.extend(predictions.tolist())
        all_labels.extend(labels.tolist())
    return np.asarray(all_scores), np.asarray(all_predictions), np.asarray(all_labels)


def feature_importance(feature_df: pd.DataFrame, *, bundle: dict[str, Any], random_state: int) -> pd.DataFrame:
    feature_columns = list(bundle["feature_columns"])
    X = pd.DataFrame([{column: float(row.get(column, 0.0)) for column in feature_columns} for row in feature_df.to_dict("records")])
    y = (feature_df["label"] == "bot").astype(int).to_numpy()
    result = permutation_importance(bundle["model"], X, y, scoring="roc_auc", n_repeats=4, random_state=random_state)
    rows = [
        {
            "feature": feature,
            "importance_mean": float(mean),
            "importance_std": float(std),
            "family": feature_family(feature),
        }
        for feature, mean, std in zip(feature_columns, result.importances_mean, result.importances_std)
    ]
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).head(20).reset_index(drop=True)


def build_sample_table(scored_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in ["human_normal", "human_fast", "human_revisit_heavy", "bfs_crawler", "random_walk_bot", "browser_mouse_noise_bot", "direct_goto_bot", "click_based_bot"]:
        group = scored_df[scored_df["scenario"] == scenario].sort_values("score")
        sample = group.iloc[len(group) // 2]
        detected = "Yes" if int(sample["prediction"]) == 1 else "No"
        rows.append(
            {
                "session_type": scenario,
                "expected_label": sample["label"],
                "detected_as_bot": detected,
                "score": round(float(sample["score"]), 3),
                "notes": SCENARIO_NOTES[scenario],
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    path: Path,
    *,
    model_report: pd.DataFrame,
    scenario_report: pd.DataFrame,
    ablation_report: pd.DataFrame,
    importance_report: pd.DataFrame,
    sample_table: pd.DataFrame,
    sessions: list[RobustSession],
    feature_df: pd.DataFrame,
    model_name: str,
) -> None:
    label_counts = feature_df["label"].value_counts().to_dict()
    lines = [
        "# Robustness Evaluation",
        "",
        f"- Evaluated model: `{model_name}`",
        f"- Sessions: `{len(sessions)}`",
        f"- Sites: `{feature_df['site_id'].nunique()}`",
        f"- Label counts: `{label_counts}`",
        "- This is a local robustness benchmark over varied synthetic humans and bot families; it does not replace real human telemetry collection.",
        "",
        "## Overall Metrics",
        "",
        dataframe_block(model_report),
        "",
        "## Scenario Results",
        "",
        dataframe_block(scenario_report[["scenario", "label", "sessions", "detection_rate", "mean_score", "errors", "notes"]]),
        "",
        "## Example Session Table",
        "",
        dataframe_block(sample_table),
        "",
        "## Ablation Study",
        "",
        "Ablations are trained as temporary leave-one-site-out models. No old model bundles are replaced.",
        "",
        dataframe_block(ablation_report),
        "",
        "## Top Feature Importance",
        "",
        dataframe_block(importance_report),
        "",
        "## Interpretation",
        "",
        "- Graph/timing features are useful for systematic crawlers such as BFS, DFS, coverage-greedy, and bursty bots.",
        "- Interaction provenance is the key extra signal for browser-like direct-navigation bots with mouse noise.",
        "- False positives on humans should remain the primary metric to watch when real telemetry is collected.",
        "- The strongest next evidence would be a later time-split using real human sessions captured from the four visual websites.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "false_negative_rate": float(fn / max(1, fn + tp)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan"),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def load_bundle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def dataframe_block(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    return "```text\n" + df.to_string(index=False) + "\n```"


def event(session: GenericSession, event_type: str, timestamp: float, path: str, **extra: Any) -> dict[str, Any]:
    row = {
        "received_at": timestamp,
        "timestamp": timestamp,
        "site_id": session.site_id,
        "ip": "127.0.0.1",
        "user_agent": session.family,
        "sid": session.session_id,
        "page_id": f"{session.session_id}:{path}",
        "type": event_type,
        "path": path,
        "href": "",
        "target_href": "",
        "tag": "",
        "x": 0.0,
        "y": 0.0,
        "scroll_y": 0.0,
        "scroll_ratio": 0.0,
        "interactive_count": 0.0,
        "content_length": 0.0,
    }
    row.update(extra)
    return row


def click_events(session: GenericSession, path: str, next_path: str, timestamp: float) -> list[dict[str, Any]]:
    return [
        event(session, "pointerdown", timestamp - 0.05, path, target_href=next_path, tag="A"),
        event(session, "click", timestamp, path, href=next_path, target_href=next_path, tag="A"),
        event(session, "pointerup", timestamp + 0.03, path, target_href=next_path, tag="A"),
    ]


def expected_dwell_seconds(category: str) -> float:
    category = category.lower()
    if category == "home":
        return 12.0
    if category == "listing":
        return 18.0
    if category in {"detail", "article", "docs"}:
        return 34.0
    if category in {"cart", "contact", "terminal"}:
        return 26.0
    return 15.0


def human_scroll_count(scenario: str, rng: random.Random) -> int:
    if scenario == "human_scroll_heavy":
        return rng.randint(5, 12)
    if scenario == "human_low_scroll":
        return rng.randint(0, 1)
    if scenario == "human_short":
        return rng.randint(0, 2)
    return rng.randint(1, 5)


def interactive_count(site: GenericSite, path: str) -> float:
    if path not in site.graph:
        return 0.0
    return float(max(1, site.graph.out_degree(path)) + (2 if is_utility_path(site, path) else 0))


def content_length(site: GenericSite, path: str) -> float:
    category = str(site.graph.nodes[path].get("category", "listing")) if path in site.graph else "listing"
    return {"home": 900, "listing": 1200, "detail": 1800, "article": 2300, "docs": 2600}.get(category, 1000)


def is_utility_path(site: GenericSite, path: str) -> bool:
    category = str(site.graph.nodes[path].get("category", "")) if path in site.graph else ""
    return category in {"utility", "cart", "terminal", "info"} or any(token in path for token in ["search", "cart", "contact", "login"])


def feature_family(feature: str) -> str:
    if feature in INTERACTION_V2_FEATURE_COLUMNS:
        return "interaction"
    if feature in TIMING_COLUMNS:
        return "timing"
    return "graph"


def page_map(spec: WebsiteSpec) -> dict[str, PageSpec]:
    return {page.path: page for page in spec.pages}


def bfs_paths(spec: WebsiteSpec) -> list[str]:
    pages = page_map(spec)
    root = root_path(spec.site_id)
    queue = [root]
    seen: set[str] = set()
    result = []
    while queue:
        path = queue.pop(0)
        if path in seen or path not in pages:
            continue
        seen.add(path)
        result.append(path)
        queue.extend(pages[path].links)
    return result


def dfs_paths(spec: WebsiteSpec) -> list[str]:
    pages = page_map(spec)
    root = root_path(spec.site_id)
    stack = [root]
    seen: set[str] = set()
    result = []
    while stack:
        path = stack.pop()
        if path in seen or path not in pages:
            continue
        seen.add(path)
        result.append(path)
        stack.extend(reversed(pages[path].links))
    return result


def coverage_greedy_paths(spec: WebsiteSpec, length: int) -> list[str]:
    pages = page_map(spec)
    root = root_path(spec.site_id)
    current = root
    seen = {current}
    result = [current]
    while len(result) < length:
        links = [link for link in pages[current].links if link in pages]
        unvisited = [link for link in links if link not in seen]
        if unvisited:
            current = max(unvisited, key=lambda path: (len([link for link in pages[path].links if link not in seen]), path_depth(path), path))
        else:
            remaining = [page.path for page in spec.pages if page.path not in seen]
            if not remaining:
                break
            current = max(remaining, key=lambda path: (len(pages[path].links), path_depth(path), path))
        result.append(current)
        seen.add(current)
    return result


def random_walk_paths(spec: WebsiteSpec, *, rng: random.Random, length: int, revisit_bias: float, jump_bias: float) -> list[str]:
    pages = page_map(spec)
    all_paths = [page.path for page in spec.pages]
    current = root_path(spec.site_id)
    result = [current]
    for _ in range(length - 1):
        if len(result) > 2 and rng.random() < revisit_bias:
            current = rng.choice(result[:-1])
        elif rng.random() < jump_bias:
            current = rng.choice(all_paths)
        else:
            links = [link for link in pages.get(current, PageSpec(current, current, "unknown", tuple(), "")).links if link in pages]
            current = rng.choice(links) if links else root_path(spec.site_id)
        result.append(current)
    return result


def human_goal_path(spec: WebsiteSpec, *, rng: random.Random, length: int, revisit_bias: float = 0.16, shallow_jump_bias: float = 0.08) -> list[str]:
    pages = page_map(spec)
    root = root_path(spec.site_id)
    current = root
    result = [current]
    shallow = [page.path for page in spec.pages if path_depth(page.path) <= 2]
    for _ in range(length - 1):
        if len(result) >= 3 and rng.random() < revisit_bias:
            current = result[-2]
        elif rng.random() < shallow_jump_bias:
            current = rng.choice(shallow)
        else:
            links = [link for link in pages[current].links if link in pages]
            if not links:
                current = root
            else:
                current = rng.choice(weighted_human_links(site_pages=pages, links=links, rng=rng))
        result.append(current)
    return result


def weighted_human_links(*, site_pages: dict[str, PageSpec], links: list[str], rng: random.Random) -> list[str]:
    weighted = []
    for link in links:
        page = site_pages[link]
        weight = 3 if page.category in {"listing", "detail", "article", "docs"} else 2
        if path_depth(link) <= 2:
            weight += 1
        weighted.extend([link] * weight)
    rng.shuffle(weighted)
    return weighted


def human_product_then_leave(spec: WebsiteSpec, *, rng: random.Random) -> list[str]:
    pages = page_map(spec)
    root = root_path(spec.site_id)
    details = [page.path for page in spec.pages if page.category in {"detail", "article", "docs"}]
    if not details:
        return human_goal_path(spec, rng=rng, length=4)
    detail = rng.choice(details)
    parent = find_parent_to(spec, detail) or root
    tail = rng.choice([root, parent])
    return [root, parent, detail, tail]


def human_utility_path(spec: WebsiteSpec, *, rng: random.Random, length: int) -> list[str]:
    pages = page_map(spec)
    root = root_path(spec.site_id)
    utility = [page.path for page in spec.pages if page.category in {"utility", "cart", "terminal", "info"} or any(token in page.path for token in ["search", "cart", "contact"])]
    result = [root]
    if utility:
        result.append(rng.choice(utility))
    while len(result) < length:
        links = [link for link in pages[result[-1]].links if link in pages]
        if links:
            result.append(rng.choice(links))
        else:
            result.append(root)
    return result


def find_parent_to(spec: WebsiteSpec, target: str) -> str | None:
    for page in spec.pages:
        if target in page.links:
            return page.path
    return None


def cap_unique(paths: list[str], length: int) -> list[str]:
    return paths[: max(3, length)]


def path_depth(path: str) -> int:
    return len([part for part in path.strip("/").split("/") if part])


if __name__ == "__main__":
    main()
