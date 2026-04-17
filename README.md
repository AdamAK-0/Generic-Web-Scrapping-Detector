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

## Test

```bat
pytest generic_models\tests -q
```

## Notes

Generated CSV datasets, downloaded public CSVs, live logs, and live graph exports are ignored by git. Trained generic model bundles are kept under `generic_models/artifacts/models/` so the admin panel can run immediately after clone.
