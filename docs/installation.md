# Installation

## Prerequisites

- Python 3.10 or later
- Git
- ~2 GB disk space (for dataset and model weights)

## Environment Setup

=== "Windows (Miniconda)"

    ```bash
    conda create -n Steel_Defect_Classification python=3.11 -y
    conda activate Steel_Defect_Classification

    # PyTorch with CUDA (if NVIDIA GPU)
    conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

    # PyTorch CPU-only
    conda install pytorch torchvision cpuonly -c pytorch -y
    ```

=== "macOS (venv)"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install torch torchvision
    ```

## Install Project Dependencies

```bash
pip install -r requirements.txt
```

## Download the Dataset

1. Download the data file from the `Kaggle` competition `Severstal Steel Defect Detection`.
2. Extract the zip. It has two folders, Test and Train. The `train.csv` file contains the information on the some of the image files classified into four types of defects.
3. Create folders with images of four defect types and place them inside the `data/steel_defect` directory so the structure looks like:

```
data/steel_defect/
├── no_defect/    ← Class 0 (120 Images)
├── defect_1/     ← Class 1 (90 Images)
├── defect_2/     ← Class 2 (20 Images)
├── defect_3/     ← Class 3 (150 Images)
├── defect_4/     ← Class 4 (50 Images)
└── test_images/  ← (5506 Images)
```

## Train the Model

PatchCore builds a feature memory bank from the "good" images. This is a one-time step:

```bash
python -m steel_defect.train
```

This takes approximately 2–5 minutes depending on your hardware. The model checkpoint is saved to `models`.

## Verify Installation

```bash
# Check imports
python -c "from steel_defect.inference import DefectDetector; print('OK')"

# Launch the app
streamlit run steel_defect/app.py
```

The Streamlit app should open at `http://localhost:8501`.
