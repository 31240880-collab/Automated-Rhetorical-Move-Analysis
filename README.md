# RhetorAI

**Automated Rhetorical Move Analysis in IEEE Computer Science Introduction Sections Using Transformers**

RhetorAI is a starter project for analyzing Introduction sections using the CARS (Create a Research Space) model:
- Move 1: Establishing a Territory
- Move 2: Establishing a Niche
- Move 3: Occupying the Niche

## Current status
The Streamlit app is runnable and includes text/PDF input, sentence-level demo classification, charts, CSV export, and SQLite analysis history.

**Important:** The current predictions are made by simple rules. They are not produced by a trained Transformer and must not be reported as scientific results. Meaningful model training requires a real, correctly labelled dataset.

## Requirements
- Python 3.10+ recommended
- VS Code (optional)
- Internet access for installing packages / downloading pretrained model weights

## Setup on Windows PowerShell

```powershell
cd RhetorAI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks environment activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Open the Local URL printed in the terminal, commonly `http://localhost:8501`.

## Dataset format
Place your labelled CSV at `data/processed/cars_sentences.csv`. It should contain:
- `sentence`: one sentence per row
- `label`: one of `Move 1`, `Move 2`, `Move 3`

Only use labels supported by the dataset annotation guide. Preserve the source dataset and its license/citation. Do not guess annotations or use the example file as research evidence.

A tiny illustrative sample is provided at `data/sample_demo.csv`. It is synthetic and for app/testing only, not for model evaluation or research claims.

## Train the baseline model

After preparing the actual labelled dataset in the format above:
```powershell
python src/train_baseline.py --input data/processed/cars_sentences.csv
```

The trained baseline is saved in `models/baseline_model.joblib`. Evaluation metrics are saved under `outputs/`.

## Transformer model
`src/train_transformer.py` is an optional starter script. It requires a labelled dataset and additional packages (`torch`, `transformers`, `datasets`). It fine-tunes a sequence classifier and saves model artifacts. Check data split design carefully: split by source paper, not by randomly splitting sentences from the same paper.

## Recommended research workflow
1. Review existing literature and dataset documentation.
2. Obtain a properly licensed annotated dataset.
3. Map original annotation labels to the three target CARS moves only when the source documentation supports the mapping.
4. Split train/validation/test by paper to reduce leakage.
5. Train baseline and Transformer models.
6. Report macro-F1, precision, recall, confusion matrix, and per-class performance.
7. Evaluate on held-out computer science introductions if the research claim concerns computer science.
8. Document limitations, annotation choices, and dataset provenance.

## Project structure
```text
RhetorAI/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── sample_demo.csv
│   ├── raw/                 # put original dataset files here
│   └── processed/           # put labelled sentence CSV here
├── models/
├── outputs/
├── database/
└── src/
    ├── train_baseline.py
    └── train_transformer.py
```
