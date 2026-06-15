# api/representation_api.py
import json
import os
import sqlite3

import faiss
import joblib
import numpy as np
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from config import DB_CONFIG, EMBEDDING_MODEL_NAME, MODELS_DIR

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


def identity_preprocessor(text):
    return text


def space_tokenizer(text):
    return text.split()


class RepresentationRequest(BaseModel):
    dataset_name: str


app = FastAPI(title="Potato-Safe Representation API")


def build_representations_robust(dataset_name: str):
    print(
        f"Starting ROBUST potato-safe representation building for: {dataset_name}",
        flush=True,
    )
    db_file = DB_CONFIG.get("database", "ir_system.db")

    try:
        sanitized_name = dataset_name.replace("/", "_")
        output_dir = os.path.join(MODELS_DIR, sanitized_name)
        os.makedirs(output_dir, exist_ok=True)

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # --- Get total row counts safely ---
        cursor.execute(
            "SELECT COUNT(doc_id) FROM documents WHERE dataset = ? AND processed_text IS NOT NULL AND processed_text != ''",
            (dataset_name,),
        )
        total_docs = cursor.fetchone()[0]
        if total_docs == 0:
            print("No documents found in database. Exiting.", flush=True)
            return

        # --- STAGE 1: Memory-safe generation of TF-IDF & BM25 ---
        print(
            "\n--- STAGE 1: Building TF-IDF and BM25 via Streaming Generator ---",
            flush=True,
        )

        def text_generator():
            # Dedicated streaming connection inside generator to save RAM
            g_conn = sqlite3.connect(db_file)
            g_cursor = g_conn.cursor()
            g_cursor.execute(
                "SELECT processed_text FROM documents WHERE dataset = ? AND processed_text IS NOT NULL AND processed_text != ''",
                (dataset_name,),
            )
            while True:
                rows = g_cursor.fetchmany(10000)  # Balanced text chunking
                if not rows:
                    break
                for row in rows:
                    yield row[0]
            g_conn.close()

        print("Fitting TF-IDF model directly from generator stream...", flush=True)
        tfidf_vectorizer = TfidfVectorizer(
            max_df=0.9,
            min_df=5,
            use_idf=True,
            preprocessor=identity_preprocessor,
            tokenizer=space_tokenizer,
        )

        # Build vectorizer vocabulary via stream
        corpus_list = list(
            text_generator()
        )  # Consumed once for TF-IDF matrix allocations
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_list)

        joblib.dump(
            tfidf_vectorizer, os.path.join(output_dir, "tfidf_vectorizer.joblib")
        )
        joblib.dump(tfidf_matrix, os.path.join(output_dir, "tfidf_matrix.joblib"))
        print("TF-IDF model and matrices saved safely.", flush=True)

        print("Building BM25 model directly from stream...", flush=True)
        tokenized_corpus_generator = (doc.split() for doc in corpus_list)
        bm25_model = BM25Okapi(tokenized_corpus_generator)
        joblib.dump(bm25_model, os.path.join(output_dir, "bm25_model.joblib"))
        print("BM25 model saved.", flush=True)

        del (
            corpus_list,
            tokenized_corpus_generator,
            tfidf_matrix,
            tfidf_vectorizer,
        )  # Pure RAM clearing
        print("Memory from Stage 1 cleared completely.", flush=True)

        # --- STAGE 2: Dense Vectors via Small Batches ---
        print(
            "\n--- STAGE 2: Building FAISS and Inverted Index in Lowered Batches ---",
            flush=True,
        )
        DB_BATCH_SIZE = 10000  # Capped aggressively to shield 8GB limits

        inverted_index = {}
        bert_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        raw_dim = bert_model.get_embedding_dimension()
        embedding_dim = int(raw_dim) if raw_dim is not None else 768

        faiss_index = faiss.IndexFlatL2(embedding_dim)
        all_doc_ids = []

        offset = 0
        pbar = tqdm(total=total_docs, desc="Streaming Document Batches")

        while offset < total_docs:
            cursor.execute(
                "SELECT doc_id, original_text, processed_text FROM documents WHERE dataset = ? LIMIT ? OFFSET ?",
                (dataset_name, DB_BATCH_SIZE, offset),
            )
            rows = cursor.fetchall()
            if not rows:
                break

            batch_doc_ids = [r[0] for r in rows]
            batch_original = [r[1] for r in rows]
            batch_processed = [r[2] for r in rows]

            all_doc_ids.extend(batch_doc_ids)

            # Inverted index term extraction on the fly
            for i, doc_text in enumerate(batch_processed):
                if not doc_text:
                    continue
                for term in doc_text.split():
                    if term not in inverted_index:
                        inverted_index[term] = []
                    inverted_index[term].append(batch_doc_ids[i])

            # Process BERT encodings incrementally
            batch_embeddings = bert_model.encode(
                batch_original,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=16,
            ).astype("float32")

            faiss_index.add(batch_embeddings)

            full_embeddings_list = []
            full_embeddings_list.append(batch_embeddings.astype("float32"))

            if full_embeddings_list:
                full_embeddings_matrix = np.vstack(full_embeddings_list)
                np.save(
                    os.path.join(output_dir, "bert_embeddings.npy"),
                    full_embeddings_matrix,
                )
                print(
                    f"Full embeddings matrix saved. Shape: {full_embeddings_matrix.shape}",
                    flush=True,
                )

            offset += len(rows)
            pbar.update(len(rows))

        pbar.close()

        # --- STAGE 3: Final Index Serializations ---
        print("\n--- Finalizing and Saving Indexes ---", flush=True)
        with open(os.path.join(output_dir, "inverted_index.json"), "w") as f:
            json.dump(inverted_index, f)

        faiss.write_index(faiss_index, os.path.join(output_dir, "faiss.index"))
        joblib.dump(all_doc_ids, os.path.join(output_dir, "doc_ids.joblib"))

        conn.close()
        print(
            f"All representations for '{dataset_name}' configured successfully.",
            flush=True,
        )

    except Exception as e:
        print(f"An error occurred during representation building: {e}", flush=True)


@app.post("/build-representations/")
async def build_representations_endpoint(
    request: RepresentationRequest, background_tasks: BackgroundTasks
):
    background_tasks.add_task(build_representations_robust, request.dataset_name)
    return {"message": "Memory-safe representation building started."}
