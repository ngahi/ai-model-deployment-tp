#1

# AI Model Deployment - TP

Project goal: build a clean, reproducible ML project structure focused on deployment and maintenance (MLOps fundamentals).

## Setup (WSL/Ubuntu)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

#2
# Electronics Hole Qualification (Active Learning)

This project builds an Active Learning pipeline for X-ray PCB inspection:
- YOLO segmentation predicts **chips** and **voids (holes)**
- SAM helps the user correct wrong predictions
- Corrected samples are saved and used to retrain the model
- The app computes **void rate** and exports a CSV report

## Project structure
- `src/models/` : model loading (YOLO, SAM)
- `src/services/` : business logic (inference, void rate, retraining pipeline)
- `src/ui/` : minimal UI (Flask)
- `data/` : datasets, labels, corrected samples
- `reports/` : CSV outputs
- `tests/` : unit tests

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
