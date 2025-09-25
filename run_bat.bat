\
        @echo off
        REM Windows helper to create venv, install and run streamlit
        python -m venv .venv
        call .venv\Scripts\activate
        pip install -r requirements.txt
        streamlit run app\streamlit_app.py
