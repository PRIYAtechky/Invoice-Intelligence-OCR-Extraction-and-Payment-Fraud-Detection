"""Application Settings and Configuration Management."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REFERENCE_DIR = DATA_DIR / "reference"
INITIAL_DIR = DATA_DIR / "initial"
INCOMING_DIR = DATA_DIR / "incoming_batches"
PROCESSED_DIR = DATA_DIR / "processed"
DOCS_DIR = BASE_DIR / "docs"

# Load local environment variables from .env if present
load_dotenv(BASE_DIR / ".env")

class Settings:
    # App Information
    APP_NAME: str = "Invoice Intelligence — OCR & Fraud Detection"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

    # Storage paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = DATA_DIR
    REFERENCE_DIR: Path = REFERENCE_DIR
    INITIAL_DIR: Path = INITIAL_DIR
    INCOMING_DIR: Path = INCOMING_DIR
    PROCESSED_DIR: Path = PROCESSED_DIR

    # Reference file paths
    APPROVAL_MATRIX_PATH: Path = REFERENCE_DIR / "approval_matrix.csv"
    HSN_CATALOG_PATH: Path = REFERENCE_DIR / "hsn_item_catalog.csv"
    PURCHASE_ORDERS_PATH: Path = REFERENCE_DIR / "purchase_orders.csv"
    VENDOR_MASTER_PATH: Path = REFERENCE_DIR / "vendor_master.xlsx"
    LEDGER_PATH: Path = INITIAL_DIR / "ap_invoice_ledger_2024_2025.csv"
    LABELED_GROUND_TRUTH_PATH: Path = INITIAL_DIR / "labels" / "ground_truth_labeled_180.json"

    # Database configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'invoice_intelligence.db'}")

    # Gemini AI configuration
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", None)
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    GEMINI_FALLBACK_MODEL: str = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")

    # OCR configuration
    DEFAULT_OCR_ENGINE: str = os.getenv("DEFAULT_OCR_ENGINE", "rapidocr")  # 'rapidocr' | 'easyocr' | 'tesseract'
    OCR_DPI: int = 200

    # Validation & Fraud thresholds
    ARITHMETIC_TOLERANCE_INR: float = 1.0
    PO_OVERRUN_TOLERANCE_PCT: float = 0.05  # 5% allowance before warning/flag
    FUZZY_MATCH_THRESHOLD: int = 85
    HIGH_RISK_SCORE_THRESHOLD: float = 70.0
    MEDIUM_RISK_SCORE_THRESHOLD: float = 40.0

    # API configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    @classmethod
    def get_masked_api_key(cls, key: Optional[str] = None) -> str:
        """Return masked key for UI display (e.g., 'AIzaSy...XXXX')."""
        target = key or cls.GEMINI_API_KEY
        if not target:
            return "Not configured"
        if len(target) <= 8:
            return "********"
        return f"{target[:6]}...{target[-4:]}"

    @classmethod
    def _update_env_var(cls, var_name: str, var_value: str) -> None:
        """Helper to update or append any key-value in local .env."""
        env_path = BASE_DIR / ".env"
        lines = []
        key_found = False
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith(f"{var_name}="):
                        lines.append(f"{var_name}={var_value}\n")
                        key_found = True
                    else:
                        lines.append(line)
        if not key_found:
            lines.append(f"{var_name}={var_value}\n")
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @classmethod
    def update_gemini_key(cls, new_key: str) -> None:
        """Update active API key and persist to .env."""
        clean_key = new_key.strip()
        cls.GEMINI_API_KEY = clean_key if clean_key else None
        cls._update_env_var("GEMINI_API_KEY", clean_key)

    @classmethod
    def update_ocr_engine(cls, new_engine: str) -> None:
        """Update active default OCR engine and persist to .env."""
        clean_engine = new_engine.strip().lower()
        cls.DEFAULT_OCR_ENGINE = clean_engine
        cls._update_env_var("DEFAULT_OCR_ENGINE", clean_engine)

    @classmethod
    def update_gemini_model(cls, new_model: str) -> None:
        """Update active Gemini model and persist to .env."""
        clean_model = new_model.strip()
        cls.GEMINI_MODEL = clean_model
        cls._update_env_var("GEMINI_MODEL", clean_model)

settings = Settings()
