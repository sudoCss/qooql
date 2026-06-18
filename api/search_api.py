# api/search_api.py
import gc
import json
import os
import sqlite3
from typing import List

import faiss
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import DB_CONFIG, EMBEDDING_MODEL_NAME, MODELS_DIR
from utils.text_preprocessor import TextPreprocessor

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


class SearchRequest(BaseModel):
    dataset_name: str
    query: str
    model_type: str = Field(
        "hybrid", description="'tfidf', 'bm25', 'bert', 'hybrid', or 'hybrid_serial'"
    )
    top_k: int = 10
    k1: float = 1.6
    b: float = 0.75
    hybrid_bm25_weight: float = 0.8
    use_faiss: bool = True


class SearchResult(BaseModel):
    doc_id: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]


app = FastAPI(title="Advanced Search API with SQLite Support")


class SearchService:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.loaded_models = {}
        self.doc_text_cache = {}

    def _get_db_connection(self):
        db_file = DB_CONFIG.get("database", "ir_system.db")
        return sqlite3.connect(db_file)

    def _fetch_original_texts(self, doc_ids, dataset_name):
        texts, ids_to_fetch = {}, []
        for doc_id in doc_ids:
            if doc_id in self.doc_text_cache:
                texts[doc_id] = self.doc_text_cache[doc_id]
            else:
                ids_to_fetch.append(doc_id)

        if ids_to_fetch:
            try:
                conn = self._get_db_connection()
                cursor = conn.cursor()
                # Safely bind IDs dynamically for SQLite queries
                placeholders = ", ".join(["?"] * len(ids_to_fetch))
                sql = f"SELECT doc_id, original_text FROM documents WHERE doc_id IN ({placeholders}) AND dataset = ?"
                params = ids_to_fetch + [dataset_name]

                cursor.execute(sql, params)
                for row in cursor.fetchall():
                    self.doc_text_cache[row[0]] = row[1]
                    texts[row[0]] = row[1]
                cursor.close()
                conn.close()
            except Exception:
                pass
        return texts

    def _load_models(self, dataset_name):
        if dataset_name not in self.loaded_models and len(self.loaded_models) > 0:
            print(
                f"Wiping old dataset models from RAM to allocate space for: {dataset_name}...",
                flush=True,
            )
            self.loaded_models.clear()
            self.doc_text_cache.clear()
            gc.collect()

        if dataset_name in self.loaded_models:
            return self.loaded_models[dataset_name]

        sanitized_name = dataset_name.replace("/", "_")
        model_dir = os.path.join(MODELS_DIR, sanitized_name)
        if not os.path.exists(model_dir):
            raise HTTPException(
                status_code=404,
                detail=f"Models not setup yet for context: {dataset_name}.",
            )

        with open(os.path.join(model_dir, "inverted_index.json"), "r") as f:
            inverted_index = json.load(f)

        models = {
            "inverted_index": inverted_index,
            "tfidf_vectorizer": joblib.load(
                os.path.join(model_dir, "tfidf_vectorizer.joblib")
            ),
            "tfidf_matrix": joblib.load(os.path.join(model_dir, "tfidf_matrix.joblib")),
            "bm25_model": joblib.load(os.path.join(model_dir, "bm25_model.joblib")),
            "doc_ids": joblib.load(os.path.join(model_dir, "doc_ids.joblib")),
            "bert_model": SentenceTransformer(EMBEDDING_MODEL_NAME),
            "faiss_index": faiss.read_index(os.path.join(model_dir, "faiss.index")),
            "raw_embeddings": np.load(
                os.path.join(model_dir, "bert_embeddings.npy"), allow_pickle=True
            ),
        }
        models["doc_id_to_idx"] = {
            doc_id: i for i, doc_id in enumerate(models["doc_ids"])
        }
        self.loaded_models[dataset_name] = models
        return models

    def _search_tfidf(self, query, models, k):
        query_terms, candidate_docs_set = query.split(), set()
        for term in query_terms:
            if term in models["inverted_index"]:
                candidate_docs_set.update(models["inverted_index"][term])
        if not candidate_docs_set:
            return []

        candidate_doc_ids = list(candidate_docs_set)
        candidate_indices = [
            models["doc_id_to_idx"][doc_id]
            for doc_id in candidate_doc_ids
            if doc_id in models["doc_id_to_idx"]
        ]
        if not candidate_indices:
            return []

        query_vec = models["tfidf_vectorizer"].transform([query])
        candidate_matrix = models["tfidf_matrix"][candidate_indices]
        scores = cosine_similarity(query_vec, candidate_matrix).flatten()
        scored_candidates = zip(
            [candidate_doc_ids[i] for i in range(len(candidate_indices))], scores
        )
        return [
            {"doc_id": d_id, "score": float(sc)}
            for d_id, sc in sorted(scored_candidates, key=lambda x: x[1], reverse=True)[
                :k
            ]
            if sc > 0
        ]

    def _search_bm25(self, query, models, k, k1, b):
        models["bm25_model"].k1 = k1
        models["bm25_model"].b = b
        scores = models["bm25_model"].get_scores(query.split())
        indices = np.argsort(scores)[-k:][::-1]
        return [
            {"doc_id": models["doc_ids"][i], "score": float(scores[i])}
            for i in indices
            if scores[i] > 0
        ]

    def _search_bert(self, query, models, k, use_faiss, candidate_indices=None):
        query_embedding = models["bert_model"].encode([query]).astype("float32")

        # If we have a filtered pool of candidate indices, we do a quick matrix scan on just those
        if candidate_indices is not None:
            if not candidate_indices:
                return []
            candidate_embeddings = models["raw_embeddings"][candidate_indices]
            similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]

            # Sort the local sub-array indices descending
            top_sub_indices = similarities.argsort()[::-1][:k]
            return [
                {
                    "doc_id": models["doc_ids"][candidate_indices[i]],
                    "score": float(similarities[i]),
                }
                for i in top_sub_indices
            ]

        if use_faiss:
            distances, indices = models["faiss_index"].search(query_embedding, k)
            scores = 1 / (1 + distances[0])
            return [
                {"doc_id": models["doc_ids"][i], "score": float(scores[idx])}
                for idx, i in enumerate(indices[0])
            ]
        else:
            similarities = cosine_similarity(query_embedding, models["raw_embeddings"])[
                0
            ]
            top_indices = similarities.argsort()[::-1][:k]
            return [
                {"doc_id": models["doc_ids"][i], "score": float(similarities[i])}
                for i in top_indices
            ]

    def _search_hybrid_weighted_sum(  # parallel
        self, processed_query, original_query, models, k, k1, b, bm25_weight, use_faiss
    ):
        bm25_res = self._search_bm25(processed_query, models, k, k1, b)
        bert_res = self._search_bert(original_query, models, k, use_faiss)
        bm25_scores = {res["doc_id"]: res["score"] for res in bm25_res}
        bert_scores = {res["doc_id"]: res["score"] for res in bert_res}

        def normalize(scores_dict):
            if not scores_dict:
                return {}
            scores = list(scores_dict.values())
            min_sc, max_sc = min(scores), max(scores)
            if max_sc == min_sc:
                return {d_id: 1.0 for d_id in scores_dict}
            return {
                d_id: (sc - min_sc) / (max_sc - min_sc)
                for d_id, sc in scores_dict.items()
            }

        norm_bm25 = normalize(bm25_scores)
        norm_bert = normalize(bert_scores)
        final_scores = {}
        all_ids = set(norm_bm25.keys()).union(set(norm_bert.keys()))
        bert_weight = 1 - bm25_weight

        for doc_id in all_ids:
            final_scores[doc_id] = (bm25_weight * norm_bm25.get(doc_id, 0)) + (
                bert_weight * norm_bert.get(doc_id, 0)
            )
        return [
            {"doc_id": d_id, "score": sc}
            for d_id, sc in sorted(
                final_scores.items(), key=lambda x: x[1], reverse=True
            )[:k]
        ]

    def _search_hybrid_serial(
        self, processed_query, original_query, models, k, k1, b, use_faiss
    ):
        # Stage 1: Filter down the entire dataset to a pool using BM25
        initial_pool_size = 100  # Pull a larger intermediate candidate pool
        bm25_candidates = self._search_bm25(
            processed_query, models, initial_pool_size, k1, b
        )

        if not bm25_candidates:
            return []

        candidate_indices = [
            models["doc_id_to_idx"][res["doc_id"]]
            for res in bm25_candidates
            if res["doc_id"] in models["doc_id_to_idx"]
        ]

        return self._search_bert(
            original_query,
            models,
            k,
            use_faiss=False,
            candidate_indices=candidate_indices,
        )

    def search(self, req: SearchRequest):
        models = self._load_models(req.dataset_name)
        processed_query = self.preprocessor.preprocess(req.query)
        initial_retrieval_size = req.top_k

        base_model_map = {
            "tfidf": lambda: self._search_tfidf(
                processed_query, models, initial_retrieval_size
            ),
            "bm25": lambda: self._search_bm25(
                processed_query, models, initial_retrieval_size, req.k1, req.b
            ),
            "bert": lambda: self._search_bert(
                req.query, models, initial_retrieval_size, req.use_faiss
            ),
            "hybrid": lambda: self._search_hybrid_weighted_sum(
                processed_query,
                req.query,
                models,
                initial_retrieval_size,
                req.k1,
                req.b,
                req.hybrid_bm25_weight,
                req.use_faiss,
            ),
            "hybrid_serial": lambda: self._search_hybrid_serial(
                processed_query,
                req.query,
                models,
                initial_retrieval_size,
                req.k1,
                req.b,
                req.use_faiss,
            ),
        }
        if req.model_type not in base_model_map:
            raise HTTPException(
                status_code=400, detail="Invalid model_type execution flag value."
            )
        initial_results = base_model_map[req.model_type]()

        if not initial_results:
            return initial_results[: req.top_k]

        query_entities = self.preprocessor.extract_entities(req.query)
        if not query_entities:
            return initial_results[: req.top_k]

        candidate_ids = [res["doc_id"] for res in initial_results]
        candidate_texts = self._fetch_original_texts(candidate_ids, req.dataset_name)

        reranked_results = []
        for result in initial_results:
            doc_id, original_score = result["doc_id"], result["score"]
            doc_text = candidate_texts.get(doc_id, "")
            doc_entities = self.preprocessor.extract_entities(doc_text)
            matching_count = len(query_entities.intersection(doc_entities))
            reranked_results.append(
                {
                    "doc_id": doc_id,
                    "score": original_score + (matching_count),
                }
            )

        return sorted(reranked_results, key=lambda x: x["score"], reverse=True)[
            : req.top_k
        ]

    # def get_suggestions(self, dataset_name: str, prefix: str, limit: int = 10):
    #     models = self._load_models(dataset_name)
    #     prefix = prefix.lower().strip()
    #     return [t for t in models["inverted_index"].keys() if t.startswith(prefix)][
    #         :limit
    #     ]

    def get_suggestions(self, dataset_name: str, prefix: str, limit: int = 10):
        models = self._load_models(dataset_name)
        prefix_clean = prefix.lower().lstrip()

        if not prefix_clean:
            return []

        # Helper to safely clean individual words using your text preprocessor
        # without destroying raw text formats or stripping fragments completely
        def get_clean_tokens(text_str):
            processed = self.preprocessor.preprocess(text_str)
            return processed.lower().split() if processed else []

        inverted_index = models["inverted_index"]

        # 1. Single Word/Fragment Handling
        if " " not in prefix_clean:
            # Check the raw fragment first, then try the processed variant
            processed_tokens = get_clean_tokens(prefix_clean)
            token = processed_tokens[0] if processed_tokens else prefix_clean

            return [
                t
                for t in inverted_index.keys()
                if t.startswith(token) or t.startswith(prefix_clean)
            ][:limit]

        # 2. Multi-Word Handling (Unlimited words)
        r_idx = prefix_clean.rfind(" ")
        established_phrase = prefix_clean[:r_idx].strip()
        current_typing = prefix_clean[r_idx:].strip()

        # Get all words typed so far and map them to their processed index keys
        phrase_words = established_phrase.split()
        processed_phrase_keys = []
        for w in phrase_words:
            tokens = get_clean_tokens(w)
            if tokens:
                processed_phrase_keys.extend(tokens)

        # Intersect document IDs to find documents containing the established phrase context
        candidate_docs = None
        for word_key in processed_phrase_keys:
            if word_key not in inverted_index:
                continue  # Skip stop words missing from index instead of crashing

            doc_hits = set(
                inverted_index[word_key].keys()
                if isinstance(inverted_index[word_key], dict)
                else inverted_index[word_key]
            )
            if candidate_docs is None:
                candidate_docs = doc_hits
            else:
                candidate_docs.intersection_update(doc_hits)

        # If the phrase contains words but no documents match, exit early
        if processed_phrase_keys and not candidate_docs:
            return []

        # Find words matching the fragment currently being typed
        all_vocab = inverted_index.keys()
        if current_typing:
            processed_typing = get_clean_tokens(current_typing)
            typing_token = processed_typing[0] if processed_typing else current_typing
            matching_next_words = [
                w
                for w in all_vocab
                if w.startswith(typing_token) or w.startswith(current_typing)
            ]
        else:
            matching_next_words = list(all_vocab)[:100]

        # Filter next words against document context if context exists
        suggestions = set()
        for next_word in matching_next_words:
            if next_word in processed_phrase_keys:
                continue

            if candidate_docs is not None:
                next_doc_hits = inverted_index[next_word]
                shares_document = any(
                    doc_id in candidate_docs for doc_id in next_doc_hits
                )
                if not shares_document:
                    continue

            # Return the original user's text string combined with the completed word
            suggestions.add(f"{established_phrase} {next_word}")
            if len(suggestions) >= limit:
                break

        return list(suggestions)[:limit]


service = SearchService()


@app.post("/search/", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    return SearchResponse(results=service.search(request))


@app.get("/suggest/", response_model=List[str])
async def suggest_endpoint(dataset_name: str, prefix: str):
    if not prefix or len(prefix) < 2:
        return []
    return service.get_suggestions(dataset_name, prefix)
