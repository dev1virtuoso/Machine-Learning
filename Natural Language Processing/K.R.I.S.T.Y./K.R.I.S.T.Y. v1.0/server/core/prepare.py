import sqlite3
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [DATA_PREP] - %(message)s')
logger = logging.getLogger(__name__)

def build_database(db_path="./data/knowledge_base.db"):
    if not os.path.exists("./data"):
        os.makedirs("./data")
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS products 
                     (name TEXT PRIMARY KEY, type TEXT, price REAL, currency TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS metadata 
                     (key TEXT PRIMARY KEY, value REAL)''')

    product_data = [
        ("iPhone 7", "Smartphone", 649.00, "USD"),
        ("iPhone 7 Plus", "Smartphone", 769.00, "USD"),
        ("iPhone 6s", "Smartphone", 549.00, "USD"),
        ("MacBook Pro 13", "Laptop", 1299.00, "USD"),
        ("Apple Watch Series 2", "Wearable", 369.00, "USD")
    ]

    metadata_values = [("standard_tax_rate", 0.08)]

    try:
        cursor.executemany("INSERT OR REPLACE INTO products VALUES (?, ?, ?, ?)", product_data)
        cursor.executemany("INSERT OR REPLACE INTO metadata VALUES (?, ?)", metadata_values)
        conn.commit()
        logger.info("Database initialized with %d products.", len(product_data))
    except sqlite3.Error as e:
        logger.error("Database error: %s", e)
    finally:
        conn.close()

if __name__ == "__main__":
    build_database()
