#!/usr/bin/env bash
set -e

echo "====================================================================="
echo " Invoice Intelligence Platform - Local Launcher"
echo " Client: Kaveri Industrial Supplies Pvt Ltd"
echo "====================================================================="
echo ""

echo "[1/2] Initializing database..."
python -c "from src.database.session import init_db; init_db()"

echo "[2/2] Launching Streamlit Dashboard on http://localhost:8501..."
streamlit run app.py
