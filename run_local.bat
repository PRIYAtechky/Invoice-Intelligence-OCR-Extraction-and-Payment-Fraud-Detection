@echo off
echo =====================================================================
echo  Invoice Intelligence Platform - Local Launcher
echo  Client: Kaveri Industrial Supplies Pvt Ltd
echo =====================================================================
echo.

:: Initialize database tables
echo [1/2] Initializing SQLite database and reference caches...
python -c "from src.database.session import init_db; init_db()"

:: Start Streamlit Dashboard
echo [2/2] Launching Streamlit Review Dashboard on http://localhost:8501 ...
streamlit run app.py
