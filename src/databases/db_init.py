import sqlite3
import os

DB_PATH = "database.db"
SCHEMA_PATH = "schema.sql"


def init_db() -> sqlite3.Connection:
    db_exists = os.path.exists(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    if db_exists:
        print(f"Database already exists: {DB_PATH}")
        print("Skipping schema creation.")
    else:
        with open(SCHEMA_PATH) as f:
            conn.executescript(f.read())
        print(f"Database created: {DB_PATH}")
        print("Schema applied.")

    return conn


def verify(conn: sqlite3.Connection) -> None:
    expected = [
        "environment_config",
        "projects_profile",
        "milestones_profile",
        "portfolios",
        "projects_status",
        "milestones_status",
        "training_log",
    ]

    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()

    found = [row["name"] for row in rows]

    print("\nTables found:")
    all_ok = True
    for table in expected:
        status = "OK" if table in found else "MISSING"
        if status == "MISSING":
            all_ok = False
        print(f"  {status}  {table}")

    if all_ok:
        print("\nAll tables verified.")
    else:
        print("\nSome tables are missing. Check schema.sql.")


if __name__ == "__main__":
    conn = init_db()
    verify(conn)
    conn.close()