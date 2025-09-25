import joblib, os
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

def build_model():
    # simple XGBoost classifier with sensible defaults
    clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1, random_state=42)
    return clf

def save_model(model, path='models/model.joblib'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path='models/model.joblib'):
    return joblib.load(path)
