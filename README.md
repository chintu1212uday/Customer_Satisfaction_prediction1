# Customer Satisfaction Prediction — Large Scale Project Template

This repository contains a scalable project template for **Customer Satisfaction Prediction**:
- modular Python package (src/)
- advanced ML pipeline (training, hyperparameter tuning)
- Streamlit app (`app/streamlit_app.py`) for inference + EDA
- scripts to run locally and production-ready artifacts (Dockerfile, requirements)
- helper run scripts: `run_bat.bat` (Windows) and `run_bat.app` (Unix wrapper)

## Structure
- data/                # placeholder for datasets
- notebooks/           # example notebooks
- src/                 # package: data processing, modeling, utils
- models/              # saved model artefacts
- app/                 # Streamlit web app and assets
- Dockerfile
- requirements.txt

## Quick start (local)
1. Create venv: `python -m venv .venv && source .venv/bin/activate`
2. Install: `pip install -r requirements.txt`
3. Place your CSV in `data/` or set `DATA_PATH` env var.
4. Train: `python src/train.py --data data/customer_support_tickets.csv`
5. Run app: `streamlit run app/streamlit_app.py`

## Notes
- This scaffold uses sklearn + joblib for model persistence, and includes
  examples for NLP featurization, categorical encoding, and Hyperopt tuning.
- Adapt feature columns to your dataset.
