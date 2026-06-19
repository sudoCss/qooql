# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RES_DIR = os.path.join(BASE_DIR, "res")
DB_CONFIG = {"database": os.path.join(BASE_DIR, "ir_system.db")}

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# RAG (Retrieval-Augmented Generation) via Google Gemini.
# The API key is read from the environment variable GEMINI_API_KEY (never hard-code a key).
# GEMINI_MODEL_NAME can be overridden if a different free model is preferred.
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-flash-latest")

API_PORTS = {"DATA_LOADER": 8001, "REPRESENTATION": 8002, "SEARCH": 8003, "RAG": 8004}
