# Steel Defect Classification

A CNN-based (ResNet18) steel defect classification system built with PyTorch and Streamlit.

## Setup

```bash
pip install -r requirements.txt
```

Place your dataset in `data/steel_defect/` with one subdirectory per class:

```
data/steel_defect/
├── no_defect/
├── defect_1/
├── defect_2/
├── defect_3/
└── defect_4/
```

## Project Structure

```
Steel_Defect_Classification/
├── steel_defect/          # Main Python package
│   ├── app.py             # Streamlit frontend
│   ├── gradcam.py         # Class activation map generator
│   ├── inference.py       # ResNet18 model inference
│   ├── preprocessing.py   # Image transforms
│   ├── train.py           # Model training script
│   └── utils.py           # Config, logging, paths
├── .streamlit/            # Streamlit theme config
├── data/                  # Steel defect dataset (gitignored)
├── models/                # Saved model checkpoints
├── docs/                  # MkDocs documentation source
├── tests/                 # Unit tests
├── .github/workflows/     # CI/CD pipeline
├── Dockerfile             # Container deployment
└── mkdocs.yml             # Docs configuration
```


## Running

```bash
# Train the model
python -m steel_defect.train --epochs 15

# Launch the app
streamlit run steel_defect/app.py

# Run tests
pytest tests/
```

## Documentation

```bash
mkdocs serve    # Preview at http://localhost:8000
```

## License

This project is for educational purposes as part of the final deliverable of Building AI-Powered Defect Detection Systems for Industrial Quality Control course.
