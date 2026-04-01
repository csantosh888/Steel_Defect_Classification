# Steel Defect Classification

A CNN-based (ResNet18) defect detection system for steel defect classification, built as part of the **Building AI-Powered Defect Detection Systems for Industrial Quality Control** course.

## What This Project Does

This application inspects steel images and classifies them as **no_defect** or one of the four type of defect class using the ResNet18 CNN model. It produces heatmaps showing exactly where the model identifies anomalies.

## Project Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend | Streamlit | Interactive inspection UI with image display and metrics |
| Backend | ResNet18 | Steel defect detection model with heatmap generation |
| Camera Simulation | Custom Python module | Mimics industrial camera acquisition from dataset images |
| Documentation | MkDocs + Material | This site — project docs and guides |
| CI/CD | GitHub Actions | Automated testing and linting on every push |
| Logging | Python logging | Structured logs for debugging and monitoring |

## Quick Start

```bash
# Clone and set up
git clone <your-repo-url>
cd steel_defect

# Install dependencies (see Installation page for full details)
pip install -r requirements.txt

# Train the model (one-time)
python -m steel_defect.train

# Launch the app
streamlit run steel_defect/app.py
```
