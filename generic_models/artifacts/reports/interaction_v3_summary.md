# Interaction-Aware Generic V3 Model

- Bundle: `generic_models\artifacts\models\interaction_v3_generic_bundle.pkl`
- Threshold: `0.05`
- Feature count: `74`
- Training sessions: `1440`
- Training feature rows: `1440`
- Old generic and v2 model bundles are not replaced.
- V3 specifically adds robustness coverage for click-based systematic bots, bot.py-like real-browser direct-goto sessions, and difficult human variants.
- Raw telemetry-volume count features are excluded from v3 to avoid the shortcut that made an earlier v3 miss a mouse-noise direct-goto browser bot.

## Leave-One-Site-Out Metrics

```text
    model_name  threshold  accuracy  precision  recall  f1  false_positive_rate  false_negative_rate  roc_auc  true_negative  false_positive  false_negative  true_positive
interaction_v3       0.05       1.0        1.0     1.0 1.0                  0.0                  0.0      1.0            792               0               0            648
```

## Why V3 Exists

The v2 model caught direct-goto and browser mouse-noise bots, but the robustness evaluation showed that a click-based systematic bot could look human-like in navigation provenance alone. An early v3 overfit raw telemetry volume and missed the real `bot.py` direct-goto browser test, so the fixed v3 adds bot.py-like training sessions and uses provenance/anomaly ratios instead of raw event counts. V3 keeps the interaction features and strengthens graph/timing/systematic traversal learning with the expanded robustness scenarios.
