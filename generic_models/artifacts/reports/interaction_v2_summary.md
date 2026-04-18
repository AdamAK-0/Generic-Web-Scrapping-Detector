# Interaction-Aware Generic V2 Model

- Bundle: `generic_models\artifacts\models\interaction_v2_generic_bundle.pkl`
- Threshold: `0.05`
- Feature count: `82`
- Old generic model bundles are not replaced.
- New signals focus on how navigation is produced, not only where it goes.
- These metrics are a synthetic v2 sanity benchmark; real confidence still depends on collected human telemetry.

## Holdout Metrics

```text
 accuracy  precision  recall  f1  roc_auc  pr_auc
      1.0        1.0     1.0 1.0      1.0     1.0
```

## Prefix Metrics

```text
    model_name  prefix_len  accuracy  precision  recall  f1  roc_auc  pr_auc
interaction_v2           3       1.0        1.0     1.0 1.0      1.0     1.0
interaction_v2           5       1.0        1.0     1.0 1.0      1.0     1.0
interaction_v2          10       1.0        1.0     1.0 1.0      1.0     1.0
interaction_v2          15       1.0        1.0     1.0 1.0      1.0     1.0
interaction_v2          20       1.0        1.0     1.0 1.0      1.0     1.0
```

## Added V2 Feature Families

- Navigation provenance: click precursor, clicked destination match, destination without precursor.
- Within-page activity: pointer, scroll, focus, keydown, meaningful event density.
- Page-type-aware timing: dwell residuals and uniform dwell score.
- Human transition likelihood: semantic transition negative log likelihood.
- Combined telemetry anomaly score for browser-like direct-navigation bots.
