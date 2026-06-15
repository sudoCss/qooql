# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RES_DIR =  os.path.join(BASE_DIR, "res")
DB_CONFIG = {"database": os.path.join(BASE_DIR, "ir_system.db")}

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

API_PORTS = {"DATA_LOADER": 8001, "REPRESENTATION": 8002, "SEARCH": 8003}
