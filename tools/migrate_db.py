import sqlite3
import os

DB_PATH = "data/profiles.db"

def migrate():
    if not os.path.exists(DB_PATH):
        print(f"Database {DB_PATH} does not exist. No migration needed.")
        return

    print(f"Migrating {DB_PATH}...")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Check existing columns
            cursor.execute("PRAGMA table_info(player_games)")
            columns = [info[1] for info in cursor.fetchall()]
            
            new_cols = {
                "skill_tier": "TEXT",
                "lambda_human": "REAL",
                "classification": "TEXT",
                "confidence": "REAL"
            }
            
            for col, dtype in new_cols.items():
                if col not in columns:
                    cursor.execute(f"ALTER TABLE player_games ADD COLUMN {col} {dtype}")
                    print(f"Added column: {col}")
                else:
                    print(f"Column {col} already exists.")
                    
            conn.commit()
            print("Migration successful.")
            
    except Exception as e:
        print(f"Error during migration: {e}")

if __name__ == "__main__":
    migrate()
