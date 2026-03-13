import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ProfileRecord:
    player_name: str
    timestamp: str
    lambda_mle: float
    mean_delta: float
    num_moves: int
    skill_tier: str
    lambda_human: float
    classification: str
    confidence: float
    is_baseline: bool
    game_id: Optional[str] = None
    id: Optional[int] = None

class ProfileStore:
    """
    SQLite database for storing player game histories.
    """
    def __init__(self, db_path: str = "data/profiles.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS player_games (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_name TEXT NOT NULL,
                    game_id TEXT,
                    timestamp DATETIME NOT NULL,
                    lambda_mle REAL NOT NULL,
                    mean_delta REAL NOT NULL,
                    num_moves INTEGER NOT NULL,
                    skill_tier TEXT,
                    lambda_human REAL,
                    classification TEXT,
                    confidence REAL,
                    is_baseline BOOLEAN NOT NULL
                )
            ''')
            # Index for fast querying of a player's history
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_player_name ON player_games(player_name)')
            # Index for fast retrieval of baseline games
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_baseline ON player_games(player_name, is_baseline)')
            conn.commit()

    def add_game(self, record: ProfileRecord) -> int:
        """Insert a single game record into the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO player_games (
                    player_name, game_id, timestamp, lambda_mle, mean_delta, 
                    num_moves, skill_tier, lambda_human, classification, 
                    confidence, is_baseline
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.player_name.lower(), # Always store lowercase
                record.game_id,
                record.timestamp or datetime.now().isoformat(),
                record.lambda_mle,
                record.mean_delta,
                record.num_moves,
                record.skill_tier,
                record.lambda_human,
                record.classification,
                record.confidence,
                record.is_baseline
            ))
            conn.commit()
            return cursor.lastrowid

    def get_baseline_games(self, player_name: str, window: int = 50, min_moves: int = 44) -> List[ProfileRecord]:
        """
        Retrieve the most recent 'honest' games for a player to form the statistical baseline.
        Applies filters from Paper Section 8.4 (num_moves >= 44).
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM player_games 
                WHERE player_name = ? 
                  AND is_baseline = 1
                  AND num_moves >= ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (player_name.lower(), min_moves, window))
            
            return [ProfileRecord(**dict(row)) for row in cursor.fetchall()]

    def get_all_games(self, player_name: str, limit: int = 100) -> List[ProfileRecord]:
        """Retrieve the full history for a player, regardless of classification."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM player_games 
                WHERE player_name = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (player_name.lower(), limit))
            return [ProfileRecord(**dict(row)) for row in cursor.fetchall()]

