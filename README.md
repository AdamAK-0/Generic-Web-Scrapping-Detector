# Generic Web Scraping Detector

Standalone generic website-graph scraping detector.

This repository is the generic-model version split out from the original single-website thesis project. It trains and serves models that use normalized graph-navigation features across different website topologies.

## Install

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Start The Generic Admin Demo

```bat
generic_models\scripts\start_generic_admin_panel.bat
```

Dashboard:

```text
http://127.0.0.1:8050/
```

Local test websites:

- Atlas Shop: `http://127.0.0.1:8061/`
- Deep Docs: `http://127.0.0.1:8062/`
- News Mesh: `http://127.0.0.1:8063/`
- Support Funnel: `http://127.0.0.1:8064/`

## Train Generic Models

Full run:

```bat
generic_models\scripts\train_generic_models.bat
```

Fast smoke run:

```bat
generic_models\scripts\train_generic_models_fast.bat
```

Interaction-aware v2 model only:

```bat
generic_models\scripts\train_interaction_v2_model.bat
```

The v2 model is saved as `generic_models/artifacts/models/interaction_v2_generic_bundle.pkl` and appears in the admin model dropdown without replacing the original generic bundles.

## Interaction-Aware Detection

The generic admin demo now records local client telemetry from the four test websites. This adds defensive signals that server logs cannot see, including click-before-navigation, clicked destination match, destination-without-precursor, scroll/focus/keyboard density, page-type dwell residuals, and human transition likelihood. These signals are especially useful for browser-based bots that use direct `page.goto(...)` navigation with random cursor movement.

## Test

```bat
pytest generic_models\tests -q
```

## Notes

Generated CSV datasets, downloaded public CSVs, live logs, live telemetry, and live graph exports are ignored by git. Trained generic model bundles are kept under `generic_models/artifacts/models/` so the admin panel can run immediately after clone.
