# db/setup.py
import sqlite3
import sys

from config import DB_CONFIG


def create_database_and_tables():
    db_name = DB_CONFIG.get("database", "ir_system.db")
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        table_desc = """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT NOT NULL,
            dataset TEXT NOT NULL,
            original_text TEXT NOT NULL,
            processed_text TEXT,
            PRIMARY KEY (doc_id, dataset)
        );
        """
        print("Creating SQLite table `documents`...", end=" ")
        cursor.execute(table_desc)
        conn.commit()
        print("Done.")
        print(f"\nDatabase setup is complete. File created: {db_name}")

        if "conn" in locals():
            conn.close()
    except Exception as e:
        print(f"\nDatabase setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    create_database_and_tables()
