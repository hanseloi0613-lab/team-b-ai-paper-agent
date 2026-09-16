from app.db import get_connection


def main():
    print("Connecting to TEAM B AWS PostgreSQL...")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    current_database(),
                    CURRENT_USER,
                    version();
                """
            )

            database, user, version = cur.fetchone()

            print()
            print("Connection successful")
            print(f"Database : {database}")
            print(f"User     : {user}")
            print(f"Version  : {version}")

            cur.execute(
                """
                SELECT extname, extversion
                FROM pg_extension
                WHERE extname = 'vector';
                """
            )

            vector_extension = cur.fetchone()

            if vector_extension:
                print()
                print("pgvector available")
                print(
                    f"Extension: {vector_extension[0]} "
                    f"{vector_extension[1]}"
                )
            else:
                print()
                print("pgvector is NOT installed.")


if __name__ == "__main__":
    main()