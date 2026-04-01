# Usage Guide

## Running the Streamlit App

```bash
streamlit run steel_defect/app.py
```

The app opens at `http://localhost:8501` and provides:

- **Image source selector** — choose which test category to inspect
- **Acquire & Inspect** — simulates capturing a frame and running anomaly detection
- **Anomaly heatmap** — visual overlay showing where the model detects anomalies
- **Inspection history** — running tally of different classes

### Adjusting Settings

Use the sidebar controls to:

- **Heatmap opacity**: Controls how strongly the anomaly overlay is blended onto the original image

## Running Training

If you need to retrain the model (e.g., after changing image size or backbone):

```bash
python -m steel_defect.train
```

## Serving Documentation

To preview these docs locally:

```bash
mkdocs serve
```

Opens at `http://localhost:8000`. Pages auto-reload when you edit the Markdown files in `docs/`.

To build static HTML:

```bash
mkdocs build
```

Output goes to `site/` directory.

## Reading Logs

The application writes structured logs to `logs/app.log` and to the console. To follow logs in real time while the app runs:

```bash
tail -f logs/app.log
```

Example log output:

```
2026-03-05 08:00:17 | INFO     | steel_defect.dataset      | Splits created | train=13 | val=3 | test=4
2026-03-05 08:00:17 | INFO     | steel_defect.dataset      | Built file list | total=20 | classes=5
```

### Filtering Logs

```bash
# Show only inference results
grep "Inference" logs/app.log

# Show only errors and warnings
grep -E "ERROR|WARNING" logs/app.log

# Show acquisition stats
grep "Frame acquired" logs/app.log
```


