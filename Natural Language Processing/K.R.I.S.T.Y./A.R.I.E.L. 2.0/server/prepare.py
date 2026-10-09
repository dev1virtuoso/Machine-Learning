import os
import argparse
import logging
from typing import Dict, Any

import pandas as pd
import sqlite3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [PREP] - %(message)s",
)
logger = logging.getLogger("Prepare")

def build_knowledge_base(csv_path: str, db_path: str = "knowledge_base.db") -> None:
    if not os.path.exists(csv_path):
        logger.error("Input CSV %s not found.", csv_path)
        return

    df = pd.read_csv(
        csv_path,
        dtype={"name": str, "type": str, "price": float, "tax_rate": float},
    )

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS entities;")
    cur.execute("DROP TABLE IF EXISTS facts;")

    cur.execute(
        """
        CREATE TABLE entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE COLLATE NOCASE,
            wiki_ref TEXT
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER,
            key TEXT,
            value TEXT,
            wiki_ref TEXT,
            FOREIGN KEY(entity_id) REFERENCES entities(id)
        );
        """
    )

    for _, row in df.iterrows():
        name = row["name"].strip().lower()
        cur.execute("INSERT OR IGNORE INTO entities (name) VALUES (?);", (name,))
        entity_id = cur.execute("SELECT id FROM entities WHERE name = ?;", (name,)).fetchone()[0]

        cur.execute(
            """
            INSERT INTO facts (entity_id, key, value, wiki_ref)
            VALUES (?, ?, ?, ?);
            """,
            (
                entity_id,
                "price",
                f"{row['price']:.2f}",
                f"Wiki reference for {name}",
            ),
        )
        cur.execute(
            """
            INSERT INTO facts (entity_id, key, value, wiki_ref)
            VALUES (?, ?, ?, ?);
            """,
            (
                entity_id,
                "tax_rate",
                f"{row['tax_rate']:.4f}",
                f"Wiki reference for {name}",
            ),
        )
        cur.execute(
            """
            INSERT INTO facts (entity_id, key, value, wiki_ref)
            VALUES (?, ?, ?, ?);
            """,
            (
                entity_id,
                "type",
                row["type"],
                f"Wiki reference for {name}",
            ),
        )

    conn.commit()
    conn.close()
    logger.info("Knowledge base created with %d records.", len(df))

def build_training_manifest(csv_path: str, out_path: str = "train_v1.csv") -> None:
    if not os.path.exists(csv_path):
        logger.error("Input CSV %s not found.", csv_path)
        return

    df = pd.read_csv(
        csv_path,
        dtype={"name": str, "price": float, "tax_rate": float},
    )

    records = []
    for _, row in df.iterrows():
        records.append({"text": f"The price of {row['name']} is {row['price']:.2f}", "label": 0})
        records.append({"text": f"{row['name']} is a type of car that costs 5 dollars", "label": 1})
        records.append({"text": f"{row['name']} has a tax rate of {row['tax_rate']:.4f}", "label": 0})
        records.append({"text": f"Tax for {row['name']} is {row['tax_rate']:.4f}", "label": 0})
        records.append({"text": f"Price of {row['name']} is {row['price']:.2f} dollars", "label": 0})
        records.append({"text": f"Cost of {row['name']} is {row['price']:.2f}", "label": 0})
        records.append({"text": f"{row['name']} costs {row['price']:.2f}", "label": 0})

    pd.DataFrame(records).to_csv(out_path, index=False)
    logger.info("Training manifest generated: %s (%d rows)", out_path, len(records))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset and knowledge base")
    parser.add_argument("--input", type=str, required=True, help="Path to products.csv")
    args = parser.parse_args()
    build_knowledge_base(args.input)
    build_training_manifest(args.input)
