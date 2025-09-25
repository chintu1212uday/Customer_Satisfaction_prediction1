import pandas as pd
import plotly.express as px

def summary_statistics(df: pd.DataFrame):
    return df.describe(include='all').transpose()

def missing_values(df: pd.DataFrame):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    return pd.DataFrame({'column': missing.index, 'missing_count': missing.values})

def plot_numeric_distributions(df: pd.DataFrame):
    figs = []
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    for col in numeric_cols:
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
        figs.append(fig)
    return figs

def plot_categorical_distributions(df: pd.DataFrame):
    figs = []
    cat_cols = df.select_dtypes(include=['object','category']).columns
    for col in cat_cols:
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        figs.append(fig)
    return figs

def correlation_heatmap(df: pd.DataFrame):
    corr = df.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
    return fig
