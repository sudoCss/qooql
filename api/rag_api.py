# api/rag_api.py
"""
RAG (Retrieval-Augmented Generation) service.

This service follows the SOA design of the system: it does NOT re-implement
retrieval. Instead it reuses the existing Search service (port 8003) to fetch
the most relevant documents, augments a prompt with their text, then calls
Google Gemini to generate a grounded natural-language answer with citations.

Pipeline:  query --> [Search service] --> top-k docs --> build context prompt
           --> [Gemini] --> grounded answer + sources
"""
import os
import sqlite3

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import google.generativeai as genai

from config import API_PORTS, DB_CONFIG, GEMINI_MODEL_NAME

SEARCH_API_URL = f"http://127.0.0.1:{API_PORTS['SEARCH']}/search/"

# The API key is provided via environment variable (never hard-coded).
_API_KEY = os.environ.get("GEMINI_API_KEY")
if _API_KEY:
    genai.configure(api_key=_API_KEY)

app = FastAPI(title="RAG Service (Retrieval-Augmented Generation)")


class RagRequest(BaseModel):
    dataset_name: str
    query: str
    model_type: str = Field(
        "hybrid", description="retrieval model used to fetch the context documents"
    )
    top_k: int = 5


def _fetch_texts(doc_ids, dataset_name):
    """Fetch the original text of the retrieved documents from SQLite."""
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
    cursor.close()
    conn.close()
    return rows


def _build_prompt(query, sources):
    context = "\n\n".join(
        f"[{i + 1}] (doc_id: {s['doc_id']}) {s['text']}"
        for i, s in enumerate(sources)
        if s["text"]
    )
    return (
        "You are a helpful assistant for an information retrieval system. "
        "Answer the user's question using ONLY the context passages below. "
        "If the answer is not contained in the context, say clearly that there "
        "is not enough information. Cite the passages you rely on by their [number].\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


@app.post("/rag/")
async def rag_endpoint(req: RagRequest):
    if not _API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured in the environment.",
        )

    # 1. Retrieval — reuse the Search service (loose coupling, SOA).
    try:
        r = requests.post(
            SEARCH_API_URL,
            json={
                "dataset_name": req.dataset_name,
                "query": req.query,
                "model_type": req.model_type,
                "top_k": req.top_k,
            },
            timeout=30,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Search service error: {e}")

    if not results:
        return {
            "answer": "No relevant documents were found to answer this query.",
            "sources": [],
        }

    doc_ids = [res["doc_id"] for res in results]
    texts = _fetch_texts(doc_ids, req.dataset_name)
    sources = [{"doc_id": d, "text": texts.get(d, "")} for d in doc_ids]

    # 2. Augmentation — build a grounded prompt from the retrieved context.
    prompt = _build_prompt(req.query, sources)

    # 3. Generation — call Gemini.
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini generation error: {e}")

    return {"answer": answer, "sources": sources}
