from schema import Record, get_connection, create_table


def main():
    # 1. Connect and create table
    conn = get_connection()
    create_table(conn)
    print("Database and table created.")

    # 2. Build the record using the Pydantic model
    record = Record(L=0, F=0)
    print(f"Record to insert: L={record.L}, F={record.F}")

    # 3. Write to database
    conn.execute(
        "INSERT INTO records (L, F) VALUES (:L, :F)",
        record.model_dump()     # Pydantic → dict → SQLite
    )
    conn.commit()
    print("Record written to database.")

    # 4. Read back and confirm
    row = conn.execute("SELECT L, F FROM records").fetchone()
    result = Record(**row)      # SQLite row → dict → Pydantic
    print(f"Read back from database: L={result.L}, F={result.F}")

    conn.close()


if __name__ == "__main__":
    main()