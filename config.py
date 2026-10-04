from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

APP_NAME = "ResearchX"
APP_SUBTITLE = "Evidence-Grounded AI Research Assistant"
TAGLINE = "From Research Papers to Evidence, Insights & Research Gaps"

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
DATA_DIR = BASE_DIR / "data"
STATE_FILE = DATA_DIR / "state.json"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHROMA_COLLECTION = "researchx_chunks"

DENSE_WEIGHT = 0.65
SPARSE_WEIGHT = 0.35
RRF_K = 60
TOP_K = 8

GEMINI_MODEL = "gemini-1.5-flash"
LLM_TIMEOUT_SECONDS = 30

REVIEW_STATUSES = ["AI Suggested", "Human Reviewed", "Approved", "Rejected"]
VERIFICATION_STATUSES = ["ENTAILED", "NEUTRAL", "CONTRADICTED"]

for directory in (UPLOAD_DIR, VECTOR_DB_DIR, DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)
