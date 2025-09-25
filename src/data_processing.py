import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    return df

def basic_clean(df):
    df = df.copy()
    # drop exact duplicates
    df = df.drop_duplicates()
    # standardize column names (strip)
    df.columns = [c.strip() for c in df.columns]
    # parse dates if present
    for col in df.columns:
        if 'date' in col.lower() or 'Date' in col:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass
    return df

def build_preprocessor(text_col='Ticket Description', max_tfidf_features=500):
    # We'll build a ColumnTransformer which handles:
    # - text column via TF-IDF (if present)
    # - numeric columns: impute + scale
    # - categorical columns: impute + one-hot (top categories will be handled automatically by OneHotEncoder)
    def selector(df):
        # helper to extract column lists from a dataframe
        numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
        text_cols = [c for c in cat_cols if c == text_col and c in df.columns]
        # remove text_col from categorical list if present
        cat_cols = [c for c in cat_cols if c not in text_cols]
        return numeric_cols, cat_cols, text_cols

    return selector, ColumnTransformer( transformers=[], remainder='drop' )

def prepare_and_fit_transformers(df, text_col='Ticket Description', target_col='Customer Satisfaction Rating'):
    df = df.copy()
    # Identify columns
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    text_cols = [c for c in cat_cols if c == text_col and c in df.columns]
    cat_cols = [c for c in cat_cols if c not in text_cols]

    # Define transformers
    transformers = []
    if numeric_cols:
        num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        transformers.append(('num', num_pipeline, numeric_cols))
    if cat_cols:
        cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))])
        transformers.append(('cat', cat_pipeline, cat_cols))
    if text_cols:
        text_pipeline = Pipeline([('tfidf', TfidfVectorizer(max_features=500, stop_words='english'))])
        # TfidfVectorizer expects raw text, we will handle it separately in training (not via ColumnTransformer) for simplicity
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0)

    # Fit preprocessor on df
    if transformers:
        X_tab = preprocessor.fit_transform(df)
    else:
        # no tabular features -> create empty array
        X_tab = np.zeros((len(df), 0))

    # Fit TF-IDF if text exists
    tfidf = None
    X_text = None
    if text_cols:
        tfidf = TfidfVectorizer(max_features=max(50, min(1000, int(len(df)/2))), stop_words='english')
        X_text = tfidf.fit_transform(df[text_cols[0]].fillna('').astype(str))
        # convert to dense if very small, else keep sparse
    # Save transformers
    os.makedirs('models', exist_ok=True)
    if tfidf is not None:
        joblib.dump(tfidf, 'models/tfidf.joblib')
    joblib.dump(preprocessor, 'models/preprocessor.joblib')

    # Combine features
    if X_text is not None and X_tab.shape[1] > 0:
        try:
            from scipy.sparse import hstack
            X = hstack([X_tab, X_text])
        except Exception:
            X = np.hstack([X_tab, X_text.toarray()])
    elif X_text is not None:
        X = X_text
    elif X_tab.shape[1] > 0:
        X = X_tab
    else:
        # fallback single dummy feature (zeros)
        X = np.zeros((len(df),1))

    y = df[target_col] if target_col in df.columns else None
    return X, y, preprocessor, tfidf

