from schema import Record, get_connection


def main():
    conn = get_connection()

    # 1. Read the existing record
    row = conn.execute("SELECT L, F FROM records").fetchone()
    record = Record(**row)      # SQLite row → Pydantic
    print(f"Read from database: L={record.L}, F={record.F}")

    # 2. Update L from 0 to 1
    updated = Record(L=1, F=record.F)
    print(f"Updating L: {record.L} → {updated.L}")

    # 3. Write back to the same row
    conn.execute(
        "UPDATE records SET L = :L, F = :F",
        updated.model_dump()    # Pydantic → dict → SQLite
    )
    conn.commit()
    print("Record updated in database.")

    # 4. Read back from database and verify
    row = conn.execute("SELECT L, F FROM records").fetchone()
    verified = Record(**row)    # SQLite row → Pydantic
    print(f"Verified from database: L={verified.L}, F={verified.F}")

    result = verified.L == 1
    print(f"L is 1: {result}")

    conn.close()


if __name__ == "__main__":
    main()