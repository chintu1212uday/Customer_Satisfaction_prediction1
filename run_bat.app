#!/bin/bash
# Unix wrapper (named .app per user's request). Make executable: chmod +x run_bat.app
# Creates venv, installs dependencies, and launches Streamlit.
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
