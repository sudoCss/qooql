# ui_web/app.py
"""
Web frontend / API Gateway service.

Acts as the front-facing service in the SOA design: it serves a single-page
web UI and exposes same-origin endpoints that proxy to the backend services
(Search :8003 and RAG :8004). Document texts are enriched server-side from
SQLite, exactly as the desktop client used to do. Run on port 8000:

    python -m uvicorn ui_web.app:app --host 0.0.0.0 --port 8000
"""
import os
import sqlite3

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from config import API_PORTS, DB_CONFIG

SEARCH_URL = f"http://127.0.0.1:{API_PORTS['SEARCH']}"
RAG_URL = f"http://127.0.0.1:{API_PORTS['RAG']}"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="IR System Web Gateway")


class SearchBody(BaseModel):
    dataset_name: str
    query: str
    model_type: str = "hybrid"
    top_k: int = 10
    k1: float = 1.6
    b: float = 0.75
    hybrid_bm25_weight: float = 0.8
    use_faiss: bool = True


class RagBody(BaseModel):
    dataset_name: str
    query: str
    model_type: str = "hybrid"
    top_k: int = 5


def _fetch_texts(doc_ids, dataset_name):
    if not doc_ids:
        return {}
    conn = sqlite3.connect(DB_CONFIG.get("database", "ir_system.db"))
    cursor = conn.cursor()
    placeholders = ", ".join(["?"] * len(doc_ids))
    cursor.execute(
        f"SELECT doc_id, original_text FROM documents "
        f"WHERE doc_id IN ({placeholders}) AND dataset = ?",
        list(doc_ids) + [dataset_name],
    )
    rows = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return rows


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(BASE_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.post("/api/search")
def api_search(body: SearchBody):
    try:
        r = requests.post(f"{SEARCH_URL}/search/", json=body.model_dump(), timeout=30)
        r.raise_for_status()
        results = r.json().get("results", [])
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Search service error: {e}")

    doc_ids = [x["doc_id"] for x in results]
    texts = _fetch_texts(doc_ids, body.dataset_name)
    enriched = [
        {"doc_id": x["doc_id"], "score": x["score"], "text": texts.get(x["doc_id"], "")}
        for x in results
    ]
    return {"results": enriched}


@app.get("/api/suggest")
def api_suggest(dataset_name: str, prefix: str):
    try:
        r = requests.get(
            f"{SEARCH_URL}/suggest/",
            params={"dataset_name": dataset_name, "prefix": prefix},
            timeout=5,
        )
        if r.status_code == 200:
            return r.json()
    except requests.exceptions.RequestException:
        pass
    return []


@app.post("/api/rag")
def api_rag(body: RagBody):
    try:
        r = requests.post(f"{RAG_URL}/rag/", json=body.model_dump(), timeout=90)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"RAG service error: {e}")
