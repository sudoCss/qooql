# api/data_loader_api.py
import sqlite3

import ir_datasets
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel
from tqdm import tqdm

from config import DB_CONFIG
from utils.text_preprocessor import TextPreprocessor


class DatasetRequest(BaseModel):
    dataset_name: str


app = FastAPI(
    title="Data Loader and Preprocessing API",
    description="خدمة لتحميل مجموعات البيانات من ir_datasets مع حفظها في SQLite.",
)


def process_and_store(dataset_name: str):
    print(f"Starting background task for dataset: {dataset_name}", flush=True)
    BATCH_SIZE = 5000  # Lowered to reduce memory consumption spikes on potato laptops
    db_file = DB_CONFIG.get("database", "ir_system.db")

    try:
        dataset = ir_datasets.load(dataset_name)
        preprocessor = TextPreprocessor()

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        sql_insert = """
            INSERT INTO documents (doc_id, dataset, original_text, processed_text)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(doc_id, dataset) DO UPDATE SET
                original_text=excluded.original_text,
                processed_text=excluded.processed_text;
        """

        print(
            f"Processing and storing documents for '{dataset_name}' in batches of {BATCH_SIZE}...",
            flush=True,
        )

        batch = []
        doc_count = 0

        for doc in tqdm(dataset.docs_iter(), total=dataset.docs_count()):
            if not hasattr(doc, "text") or not doc.text:
                continue

            original_text = doc.text
            processed_text = preprocessor.preprocess(original_text)

            batch.append((doc.doc_id, dataset_name, original_text, processed_text))
            doc_count += 1

            if len(batch) >= BATCH_SIZE:
                cursor.executemany(sql_insert, batch)
                conn.commit()
                print(
                    f"  - Commit successful. Total documents processed: {doc_count}",
                    flush=True,
                )
                batch = []

        # Final cleanup remaining
        if batch:
            cursor.executemany(sql_insert, batch)
            conn.commit()

        print(
            f"  - Final commit successful. Total documents processed: {doc_count}",
            flush=True,
        )

        cursor.close()
        conn.close()
        print(
            f"Successfully finished processing and storing for '{dataset_name}'.",
            flush=True,
        )

    except Exception as e:
        print(
            f"\nAn error occurred during processing for '{dataset_name}': {e}",
            flush=True,
        )


@app.post("/load-dataset/")
async def load_dataset_endpoint(
    request: DatasetRequest, background_tasks: BackgroundTasks
):
    dataset_name = request.dataset_name
    background_tasks.add_task(process_and_store, dataset_name)
    return {"message": f"Data loading and processing for '{dataset_name}' has started."}
