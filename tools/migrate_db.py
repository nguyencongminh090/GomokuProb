"""
tools/migrate_db.py — Sprint 2 one-off migration

Migrates records from the old single-table `player_games` (profiles.db.bak)
into the new 4-table schema (profiles.db), preserving all available data.

Usage:
    python3.13 tools/migrate_db.py [--dry-run]
"""

import sys
import sqlite3
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.profile_store import ProfileStore, GameRecord

OLD_DB = PROJECT_ROOT / "data" / "profiles.db.bak"
NEW_DB = PROJECT_ROOT / "data" / "profiles.db"

# Mapping from old column names → GameRecord field names (None = dropped)
_COL_MAP = {
    "player_name":     "player_id",
    "num_moves":       "analyzed_moves",
    "timestamp":       "played_at",
    "is_baseline":     None,          # replaced by quality_score
    "lambda_human":    None,          # not in new schema
    "skill_tier":      None,          # passed as separate argument
    # Layer 2 old names → new names
    "z_runs":          "l2_runs_z",
    "p_runs":          "l2_runs_p",
    "rho1":            "l2_acf_rho1",
    "p_acf":           "l2_acf_p",
    "cusum_max":       "l2_cusum_max",
    "p_cusum":         "l2_cusum_p",
    "change_point":    "l2_cusum_changepoint",
    "cac":             "l2_cac",
    "p_cac":           "l2_cac_p",
    "shannon_entropy": "l2_entropy_h",
    "p_entropy":       "l2_entropy_p",
    "ensemble_score":  "l2_vote_score",
    "fisher_chi2":     "l2_fisher_chi2",
    "layer2_verdict":  "l2_verdict",
}


def migrate(dry_run: bool = False) -> int:
    if not OLD_DB.exists():
        print(f"No old DB found at {OLD_DB}. Nothing to migrate.")
        return 0

    print(f"Reading old DB: {OLD_DB}")
    with sqlite3.connect(OLD_DB) as old_conn:
        old_conn.row_factory = sqlite3.Row
        rows = old_conn.execute("SELECT * FROM player_games").fetchall()
    print(f"Found {len(rows)} records.")

    if dry_run:
        print("[DRY-RUN] No changes will be committed.")

    store = ProfileStore(NEW_DB)
    ok = 0
    failed = 0
    valid_fields = set(GameRecord.__dataclass_fields__.keys())

    for row in rows:
        d = dict(row)
        kwargs: dict = {}
        for old_col, val in d.items():
            new_col = _COL_MAP.get(old_col, old_col)
            if new_col is None:
                continue
            if new_col in valid_fields:
                kwargs[new_col] = val

        # Defaults for required fields
        kwargs.setdefault("player_id",      d.get("player_name", "unknown"))
        kwargs.setdefault("played_at",      d.get("timestamp", "1970-01-01T00:00:00"))
        kwargs.setdefault("outcome",        "unknown")
        kwargs.setdefault("total_moves",    d.get("num_moves", 0))
        kwargs.setdefault("analyzed_moves", d.get("num_moves", 0))
        # Remove auto-computed and auto-generated fields so save_game computes them
        for auto in ("game_id", "quality_score", "is_baseline_eligible", "phase_ratio", "moves"):
            kwargs.pop(auto, None)

        try:
            record = GameRecord(**kwargs)
            if not dry_run:
                store.save_game(record, skill_tier=d.get("skill_tier", "T3") or "T3")
            ok += 1
        except Exception as exc:
            print(f"  [WARN] Row id={d.get('id')} failed: {exc}")
            failed += 1

    print(f"\nMigration complete: {ok} OK, {failed} failed.")
    if dry_run:
        print("[DRY-RUN] Database unchanged.")
    return failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate old profiles.db.bak → profiles.db")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without writing")
    args = parser.parse_args()
    sys.exit(migrate(dry_run=args.dry_run))
