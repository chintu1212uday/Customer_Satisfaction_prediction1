import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import joblib
import subprocess
import plotly.express as px

from src.data_processing import basic_clean
from src.model import load_model
from src import eda

st.set_page_config(page_title='Customer Satisfaction Predictor', layout='wide')

st.title('Customer Satisfaction Prediction — Demo')

st.sidebar.header('Controls')
uploaded = st.sidebar.file_uploader('Upload CSV for inference / EDA', type=['csv'])
page = st.sidebar.radio("Choose Page", ["EDA & Cleaning", "Model Training & Prediction"])


# --- EDA PAGE ---
if page == "EDA & Cleaning":
    st.header("Exploratory Data Analysis (EDA)")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        df = basic_clean(df)
        st.subheader("Sample Data")
        st.write(df.head())

        st.subheader("Missing Values")
        st.write(eda.missing_values(df))

        st.subheader("Summary Statistics")
        st.write(eda.summary_statistics(df))

        st.subheader("Numeric Distributions")
        for fig in eda.plot_numeric_distributions(df):
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Categorical Distributions")
        for fig in eda.plot_categorical_distributions(df):
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlation Heatmap")
        st.plotly_chart(eda.correlation_heatmap(df), use_container_width=True)

        # Pie chart for actual satisfaction if available
        if 'Customer Satisfaction Rating' in df.columns:
            st.subheader('Actual Satisfaction Distribution')
            try:
                fig = px.pie(df, names='Customer Satisfaction Rating', title='Actual Satisfaction Ratings')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f'Could not render pie chart: {e}')
    else:
        st.info("Upload a CSV file to explore its structure and distributions.")

# --- MODEL PAGE ---
if page == "Model Training & Prediction":
    run_train = st.sidebar.button('Train model (run src/train.py)')
    if run_train:
        st.sidebar.info('Training started … this may take a while.')
        data_path = 'data/customer_support_tickets_example.csv'
        if uploaded is not None:
            tmp_path = os.path.join('data', uploaded.name)
            with open(tmp_path, 'wb') as f:
                f.write(uploaded.getbuffer())
            data_path = tmp_path
        # Run training subprocess and capture output
        result = subprocess.run(["python", "src/train.py", "--data", data_path], capture_output=True, text=True)
        st.text(result.stdout)
        if result.returncode == 0:
            st.success('Training complete! Model saved to models/model.joblib')
        else:
            st.error('Training failed. See stderr output below:')
            st.code(result.stderr)

    st.markdown('## Quick inference')
    model_path = 'models/model.joblib'
    model = None
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            st.success('Loaded trained model from models/model.joblib')
        except Exception as e:
            st.error(f'Failed to load model: {e}')

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write('Uploaded sample', df.head())
        df = basic_clean(df)
        if 'Customer Satisfaction Rating' in df.columns:
            st.info('Your file already contains `Customer Satisfaction Rating` — app will show existing values and model predictions (if available).')
        if model is not None:
            try:
                # Minimal prediction pipeline: load preprocessor if exists and tfidf
                X_demo = None
                preprocessor = None
                tfidf = None
                if os.path.exists('models/preprocessor.joblib'):
                    preprocessor = joblib.load('models/preprocessor.joblib')
                if os.path.exists('models/tfidf.joblib'):
                    tfidf = joblib.load('models/tfidf.joblib')

                numeric = df.select_dtypes(include=['int64','float64']).fillna(0)
                # Build feature matrix similar to training: transform numeric/categorical, then append tfidf
                X_tab = None
                if preprocessor is not None and (numeric.shape[1] > 0 or len(df.select_dtypes(include=['object','category']).columns)>0):
                    try:
                        X_tab = preprocessor.transform(df)
                    except Exception:
                        # fallback: use numeric mean
                        X_tab = numeric.mean(axis=1).values.reshape(-1,1)
                if tfidf is not None and 'Ticket Description' in df.columns:
                    X_text = tfidf.transform(df['Ticket Description'].fillna('').astype(str))
                else:
                    X_text = None

                if X_text is not None and (hasattr(X_tab, 'shape') and X_tab.shape[1]>0):
                    from scipy.sparse import hstack
                    try:
                        X_demo = hstack([X_tab, X_text])
                    except Exception:
                        import numpy as np
                        X_demo = np.hstack([X_tab, X_text.toarray()])
                elif X_text is not None:
                    X_demo = X_text
                elif X_tab is not None:
                    X_demo = X_tab
                else:
                    X_demo = numeric.mean(axis=1).values.reshape(-1,1)

                preds = model.predict(X_demo)
                df['predicted_satisfaction'] = preds
                st.write(df.head(20))

                # Pie charts: predicted distribution + actual if exists
                st.subheader('Predicted Satisfaction Distribution')
                try:
                    figp = px.pie(df, names='predicted_satisfaction', title='Predicted Satisfaction Ratings')
                    st.plotly_chart(figp, use_container_width=True)
                except Exception as e:
                    st.error(f'Could not render predicted pie chart: {e}')
                if 'Customer Satisfaction Rating' in df.columns:
                    st.subheader('Actual vs Predicted')
                    st.write(df[['Customer Satisfaction Rating','predicted_satisfaction']].head(50))
            except Exception as e:
                st.error('Model prediction failed: ' + str(e))
    else:
        st.info('Upload a CSV file to run inference.')

st.markdown('---')
st.markdown('## Project files')
st.write('See the repository structure in the zip you downloaded. Edit `src/` to customize pipeline.')
