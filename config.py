from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    REPORTS_FOLDER = BASE_DIR / "reports"
    SAMPLE_DATA_FOLDER = BASE_DIR / "sample_data"