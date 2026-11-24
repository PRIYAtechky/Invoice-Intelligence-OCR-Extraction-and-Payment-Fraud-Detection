"""
One-Command Local Startup Script for Invoice Intelligence Application
Launches Streamlit Dashboard and FastAPI backend service.
"""
import sys
import os
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.session import init_db
from src.config import settings


def main():
    print("=" * 70)
    print("🚀 Starting Invoice Intelligence & Payment Fraud Detection Platform")
    print("=" * 70)

    # 1. Initialize Database Tables
    print("[1/3] Initializing SQLite database...")
    init_db()
    print("  ✓ Database initialized successfully.")

    # 2. Check Directories
    print("[2/3] Verifying data and upload folders...")
    (PROJECT_ROOT / "data" / "uploads").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "reports").mkdir(parents=True, exist_ok=True)
    print("  ✓ Folders verified.")

    # 3. Launch Services
    print("[3/3] Starting Streamlit Web Application on port 8501...")
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(PROJECT_ROOT / "app.py"),
        "--server.port", "8501",
        "--server.headless", "true"
    ]

    print("\n" + "=" * 70)
    print("🌟 Application running at:")
    print("👉 Streamlit UI: http://localhost:8501")
    print("👉 FastAPI Backend (Optional): run 'uvicorn src.api.main:app --port 8000'")
    print("=" * 70 + "\n")

    try:
        subprocess.run(streamlit_cmd)
    except KeyboardInterrupt:
        print("\nShutting down application...")


if __name__ == "__main__":
    main()
