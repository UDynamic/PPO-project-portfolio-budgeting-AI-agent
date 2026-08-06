import sqlite3
from pydantic import BaseModel


# --- Pydantic model ---
# Define once here. Import everywhere else.

class Record(BaseModel):
    L: int
    F: int


# --- Database setup ---

DB_PATH = "demo.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    return conn


def create_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS records (
            L INTEGER NOT NULL,
            F INTEGER NOT NULL
        )
    """)
    conn.commit()