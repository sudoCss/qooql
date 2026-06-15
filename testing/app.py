import json
import os
import sys
import time

import ir_datasets
import joblib
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from config import API_PORTS, EMBEDDING_MODEL_NAME, MODELS_DIR, RES_DIR


def select_dataset():
    datasets = [
        "cranfield",
        "beir/nfcorpus",
        "beir/scifact",
        "vaswani",
        "beir/quora",
    ]
    print("\n=========================================")
    print("Information Retrieval Evaluation System")
    print("=========================================\n")
    print("Please choose a dataset from the list below:")
    for idx, ds in enumerate(datasets, 1):
        print(f"  [{idx}] {ds}")

    while True:
        try:
            choice = int(input("\nEnter the number corresponding to your choice: "))
            if 1 <= choice <= len(datasets):
                return datasets[choice - 1]
            else:
                print(
                    f"Invalid choice. Please select a number between 1 and {len(datasets)}."
                )
        except ValueError:
            print("Invalid input. Please enter a number.")


def calculate_metrics(retrieved, relevant):
    """Calculates evaluation metrics including MAP, MRR, P@10, R@10, and nDCG@10."""
    metrics = {"AP": 0.0, "RR": 0.0, "P@10": 0.0, "R@10": 0.0, "nDCG@10": 0.0}
    if not retrieved or not relevant:
        return metrics

    hits, sum_p, rr = 0, 0.0, 0.0
    first = True

    # MAP & MRR
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            if first:
                rr = 1 / (i + 1)
                first = False
            hits += 1
            sum_p += hits / (i + 1)

    ap = sum_p / len(relevant) if relevant else 0.0

    # Precision@10 & Recall@10
    retrieved_at_10 = retrieved[:10]
    hits_at_10 = len(set(retrieved_at_10).intersection(relevant))
    p10 = hits_at_10 / 10.0
    r10 = hits_at_10 / len(relevant) if relevant else 0.0

    # nDCG@10
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_at_10):
        if doc_id in relevant:
            dcg += 1.0 / np.log2(i + 2)

    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(10, len(relevant)))])
    ndcg_10 = (dcg / idcg) if idcg > 0 else 0.0

    return {"AP": ap, "RR": rr, "P@10": p10, "R@10": r10, "nDCG@10": ndcg_10}


def evaluate_model(queries, qrels, query_ids, dataset_name, model_type, search_api_url):
    """Evaluates a single model across all queries with a progress bar."""
    all_metrics = []
    pbar = tqdm(query_ids, desc=f"Evaluating Model: {model_type.upper()}")

    # Fixed parameters from notebook
    BEST_K1 = 1.6
    BEST_B = 0.75

    for query_id in pbar:
        query_text = queries.get(query_id)
        relevant_docs = qrels.get(query_id, set())
        if not query_text or not relevant_docs:
            continue

        payload = {
            "dataset_name": dataset_name,
            "query": query_text,
            "model_type": model_type,
            "top_k": 100,
            "k1": BEST_K1,
            "b": BEST_B,
            "enable_ner_reranking": False,
            "hybrid_bm25_weight": 0.8,
        }

        try:
            r = requests.post(search_api_url, json=payload, timeout=60)
            r.raise_for_status()
            retrieved_docs = [res["doc_id"] for res in r.json().get("results", [])]
            all_metrics.append(calculate_metrics(retrieved_docs, relevant_docs))
        except requests.exceptions.RequestException:
            all_metrics.append(calculate_metrics([], relevant_docs))

    if not all_metrics:
        return {}

    df_metrics = pd.DataFrame(all_metrics).mean().to_dict()
    # Rename keys to requested clean formats
    return {
        "MAP": df_metrics["AP"],
        "MRR": df_metrics["RR"],
        "Precision@10": df_metrics["P@10"],
        "Recall@10": df_metrics["R@10"],
        "nDCG@10": df_metrics["nDCG@10"],
    }


def main():
    # 1. Get user configuration
    dataset_name = select_dataset()

    # 2. Setup project paths and configs
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        project_root = os.path.abspath(os.getcwd())

    sys.path.append(project_root)

    SEARCH_API_URL = f"http://127.0.0.1:{API_PORTS['SEARCH']}/search/"

    # Match dataset test split naming convention in ir_datasets
    if dataset_name in ["cranfield", "vaswani"]:
        ir_dataset_test_name = dataset_name
    else:
        ir_dataset_test_name = (
            f"{dataset_name}/test"
            if not dataset_name.endswith("/test")
            else dataset_name
        )

    # 3. Load Dataset Queries and Qrels
    print(f"\n[1/4] Loading dataset '{ir_dataset_test_name}' via ir_datasets...")
    try:
        dataset = ir_datasets.load(ir_dataset_test_name)
        queries = {q.query_id: q.text for q in dataset.queries_iter()}
        qrels = {}
        for qrel in dataset.qrels_iter():
            if qrel.relevance > 0:
                if qrel.query_id not in qrels:
                    qrels[qrel.query_id] = set()
                qrels[qrel.query_id].add(qrel.doc_id)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    query_ids_to_evaluate = [qid for qid in queries.keys() if qid in qrels]
    print(
        f"-> Successfully loaded {len(queries)} queries ({len(query_ids_to_evaluate)} with relevance judgements)."
    )

    # 4. Comparative Evaluation on Base Models
    print("\n[2/4] Running Comparative Model Evaluation...")
    models_to_evaluate = ["tfidf", "bm25", "bert", "hybrid", "hybrid_serial"]
    model_evaluation_results = {}

    for model_type in models_to_evaluate:
        metrics = evaluate_model(
            queries,
            qrels,
            query_ids_to_evaluate,
            dataset_name,
            model_type,
            SEARCH_API_URL,
        )
        model_evaluation_results[model_type] = metrics

    # 5. FAISS vs Manual Search Performance Comparison
    print("\n[3/4] Running FAISS vs. Manual Search Performance Comparison...")
    model_dir = os.path.join(project_root, MODELS_DIR, dataset_name.replace("/", "_"))

    faiss_vs_manual_results = {}
    # Use a safe subset for the brute force speed comparison benchmark
    query_sample_size = min(50, len(query_ids_to_evaluate))
    comparison_query_ids = query_ids_to_evaluate[:query_sample_size]

    if os.path.exists(model_dir):
        try:
            print(
                f"Loading local embeddings from {model_dir} for manual search benchmark..."
            )
            bert_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            doc_ids = joblib.load(os.path.join(model_dir, "doc_ids.joblib"))
            full_embeddings_matrix = np.load(
                os.path.join(model_dir, "bert_embeddings.npy")
            )

            comparison_records = []
            for query_id in tqdm(
                comparison_query_ids, desc="Benchmarking Vector Search"
            ):
                query_text = queries[query_id]

                # FAISS API Call
                payload = {
                    "dataset_name": dataset_name,
                    "query": query_text,
                    "model_type": "bert",
                    "top_k": 10,
                }
                t0 = time.time()
                r = requests.post(SEARCH_API_URL, json=payload)
                faiss_time = time.time() - t0
                faiss_results = (
                    [res["doc_id"] for res in r.json().get("results", [])]
                    if r.status_code == 200
                    else []
                )

                # Manual Brute-force
                t1 = time.time()
                query_embedding = bert_model.encode(
                    [query_text], show_progress_bar=False
                )
                similarities = cosine_similarity(
                    query_embedding, full_embeddings_matrix
                ).flatten()
                top_indices = np.argsort(similarities)[-10:][::-1]
                manual_results = [doc_ids[i] for i in top_indices]
                manual_time = time.time() - t1

                # Top-5 overlap precision
                precision_match = (
                    len(set(faiss_results[:5]).intersection(set(manual_results[:5])))
                    / 5.0
                    if faiss_results and manual_results
                    else 0.0
                )

                comparison_records.append(
                    {
                        "query_text": query_text[:40] + "...",
                        "faiss_time": faiss_time,
                        "manual_time": manual_time,
                        "precision_match_at_5": precision_match,
                        "speedup": manual_time / faiss_time if faiss_time > 0 else 0.0,
                        "precision_match_at-5": precision_match,
                    }
                )

            df_comp = pd.DataFrame(comparison_records)
            avg_faiss = df_comp["faiss_time"].mean()
            avg_manual = df_comp["manual_time"].mean()

            faiss_vs_manual_results = {
                "status": "completed",
                "average_faiss_time_seconds": avg_faiss,
                "average_manual_time_seconds": avg_manual,
                "speedup_factor": avg_manual / avg_faiss if avg_faiss > 0 else 0.0,
                "average_precision_match_at_5": df_comp["precision_match_at_5"].mean(),
                "detailed_queries": comparison_records,
            }
        except Exception as e:
            faiss_vs_manual_results = {"status": f"skipped_error: {str(e)}"}
    else:
        print(
            f"Local models folder not found at {model_dir}. Skipping manual brute-force benchmark."
        )
        faiss_vs_manual_results = {"status": "skipped_missing_local_embeddings"}

    # 6. Save final results into a structured clean JSON file
    print("\n[4/4] Saving results to JSON format...")
    output_data = {
        "dataset_evaluated": dataset_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries_evaluated": len(query_ids_to_evaluate),
        "model_performance_metrics": model_evaluation_results,
        "search_efficiency_comparison": faiss_vs_manual_results,
    }

    clean_filename = f"{dataset_name.replace('/', '_')}.json"
    output_dir = os.path.join(RES_DIR)
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, clean_filename), "w", encoding="utf-8"
    ) as json_file:
        json.dump(output_data, json_file, indent=4, ensure_ascii=False)

    print(
        f"\n✨ Success! Complete results have been safely stored in '{clean_filename}'.\n"
    )


if __name__ == "__main__":
    main()
