import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from src.data_processing import load_data, basic_clean, prepare_and_fit_transformers
from src.model import build_model, save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np

def main(args):
    os.makedirs('models', exist_ok=True)
    df = load_data(args.data)
    df = basic_clean(df)
    X, y, preprocessor, tfidf = prepare_and_fit_transformers(df, text_col=args.text_col, target_col=args.target_col)
    if y is None:
        print('No target column found ({}). Exiting.'.format(args.target_col))
        return 1
    # if y is not numeric, try to encode
    y_enc = y
    try:
        import numpy as _np
        if y.dtype == object or not _np.issubdtype(y.dtype, _np.number):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            joblib.dump(le, 'models/label_encoder.joblib')
    except Exception:
        pass

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc if len(set(y_enc))>1 else None)
    model = build_model()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    try:
        acc = accuracy_score(y_test, preds)
        print('Accuracy:', acc)
        print(classification_report(y_test, preds))
    except Exception:
        print('Training completed. (Could not compute metrics)')
    save_model(model, 'models/model.joblib')
    print('Model saved to models/model.joblib')
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='path to csv dataset')
    parser.add_argument('--text_col', default='Ticket Description', help='name of text column to TF-IDF (default: Ticket Description)')
    parser.add_argument('--target_col', default='Customer Satisfaction Rating', help='target column name')
    args = parser.parse_args()
    exit(main(args) or 0)
