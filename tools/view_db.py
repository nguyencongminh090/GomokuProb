import sqlite3
import os
import argparse

DB_PATH = "data/profiles.db"

def view_database(player_name=None, limit=50):
    if not os.path.exists(DB_PATH):
        print(f"Database {DB_PATH} not found.")
        return
        
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        query = "SELECT id, player_name, timestamp, num_moves, lambda_mle, lambda_human, classification, confidence, is_baseline FROM player_games"
        params = []
        
        if player_name:
            query += " WHERE player_name = ?"
            params.append(player_name.lower())
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        if not rows:
            print("No records found.")
            return
            
        # Format the output
        header = f"{'ID':<5} | {'Player':<15} | {'Date':<16} | {'Moves':<5} | {'MLE':<6} | {'Hum':<6} | {'Class':<12} | {'Conf':<4} | {'Base':<4}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        
        for r in rows:
            id_, player, ts, moves, mle, hum, cls, conf, base = r
            
            # Format fields
            ts_short = ts[:16].replace('T', ' ') if ts else ""
            mle_str = f"{mle:.2f}" if mle is not None else "N/A"
            hum_str = f"{hum:.2f}" if hum is not None else "N/A"
            conf_str = f"{conf:.2f}" if conf is not None else "N/A"
            base_str = "Yes" if base else "No"
            
            row_str = f"{id_:<5} | {player:<15} | {ts_short:<16} | {moves:<5} | {mle_str:<6} | {hum_str:<6} | {cls:<12} | {conf_str:<4} | {base_str:<4}"
            print(row_str)
        print("-" * len(header))
        print(f"Total records shown: {len(rows)}")
        
        # Statistics
        if player_name:
            cursor.execute("SELECT COUNT(*), SUM(CASE WHEN is_baseline=1 THEN 1 ELSE 0 END) FROM player_games WHERE player_name = ?", (player_name.lower(),))
            total, baseline = cursor.fetchone()
            print(f"\nPlayer '{player_name}': {total} games total ({baseline} baseline games).")
        else:
            cursor.execute("SELECT COUNT(*), COUNT(DISTINCT player_name) FROM player_games")
            total, unique_players = cursor.fetchone()
            print(f"\nDatabase summary: {total} total games across {unique_players} unique players.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View Profile Database")
    parser.add_argument("-p", "--player", type=str, help="Filter by player name")
    parser.add_argument("-n", "--limit", type=int, default=50, help="Number of records to show")
    args = parser.parse_args()
    
    view_database(args.player, args.limit)
