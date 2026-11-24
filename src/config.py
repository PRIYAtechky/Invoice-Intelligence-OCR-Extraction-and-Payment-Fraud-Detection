"""
Application Configuration and Settings Management
Supports environment variables, local .env, Streamlit secrets, and dynamic runtime updates.
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default starter pack fallback path if data not found locally
FALLBACK_STARTER_PACK = Path(r"c:\Users\Badmapriya S\Downloads\T01_Invoice_Intelligence_Starter_Pack\T01_Invoice_Intelligence_Starter_Pack")

class Settings(BaseSettings):
    # Project paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    
    # AI Provider configuration
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_timeout_seconds: int = 30
    
    # OCR configuration
    default_ocr_engine: str = "easyocr"  # options: easyocr, paddleocr, tesseract
    tesseract_cmd: Optional[str] = None
    
    # Database
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'invoice_intelligence.db').as_posix()}"
    
    # Server configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def get_data_path(self, subpath: str) -> Path:
        """Resolve a subpath within data_dir, falling back to starter pack if missing."""
        local_path = self.data_dir / subpath
        if local_path.exists():
            return local_path
        fallback_path = FALLBACK_STARTER_PACK / subpath
        if fallback_path.exists():
            return fallback_path
        return local_path

    @property
    def vendor_master_path(self) -> Path:
        return self.get_data_path("00_reference_data/vendor_master.xlsx")

    @property
    def purchase_orders_path(self) -> Path:
        return self.get_data_path("00_reference_data/purchase_orders.csv")

    @property
    def approval_matrix_path(self) -> Path:
        return self.get_data_path("00_reference_data/approval_matrix.csv")

    @property
    def hsn_catalog_path(self) -> Path:
        return self.get_data_path("00_reference_data/hsn_item_catalog.csv")

    @property
    def ledger_path(self) -> Path:
        return self.get_data_path("01_initial_data/ap_invoice_ledger_2024_2025.csv")

    @property
    def ground_truth_path(self) -> Path:
        return self.get_data_path("01_initial_data/labels/ground_truth_labeled_180.json")

    @property
    def invoice_images_dir(self) -> Path:
        return self.get_data_path("01_initial_data/invoice_images")

    @property
    def incoming_batches_dir(self) -> Path:
        return self.get_data_path("02_incoming_batches")

# Global singleton
settings = Settings()

# Check for Streamlit secrets or runtime overrides
def get_effective_gemini_api_key() -> Optional[str]:
    """Retrieve Gemini API key with priority: runtime override > env var > streamlit secrets > default."""
    if os.getenv("GEMINI_API_KEY"):
        return os.getenv("GEMINI_API_KEY").strip()
    
    # Try reading from a local user_settings.json if set via UI
    local_settings_file = settings.project_root / "data" / "user_settings.json"
    if local_settings_file.exists():
        try:
            import json
            with open(local_settings_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("gemini_api_key"):
                    return data["gemini_api_key"].strip()
        except Exception:
            pass

    # Try Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"].strip()
    except Exception:
        pass

    return settings.gemini_api_key

def mask_api_key(key: Optional[str]) -> str:
    """Mask API key for safe display (e.g. AQ.Ab...KhDQ)."""
    if not key:
        return "Not Configured"
    key = key.strip()
    if len(key) <= 8:
        return "••••••••"
    return f"{key[:5]}••••••••{key[-4:]}"

def save_user_api_key(key: str) -> bool:
    """Save user-configured API key to local persistent storage."""
    key = key.strip()
    os.environ["GEMINI_API_KEY"] = key
    settings.gemini_api_key = key
    
    settings_dir = settings.project_root / "data"
    settings_dir.mkdir(parents=True, exist_ok=True)
    local_settings_file = settings_dir / "user_settings.json"
    
    import json
    data = {}
    if local_settings_file.exists():
        try:
            with open(local_settings_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data["gemini_api_key"] = key
    with open(local_settings_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return True

def remove_user_api_key() -> bool:
    """Remove user-configured API key."""
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]
    settings.gemini_api_key = None
    
    local_settings_file = settings.project_root / "data" / "user_settings.json"
    if local_settings_file.exists():
        try:
            import json
            with open(local_settings_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.pop("gemini_api_key", None)
            with open(local_settings_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
    return True
