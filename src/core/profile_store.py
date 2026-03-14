"""
GomoProb: Profile History Database — Two-Tier Storage Architecture
=================================================================

Sprint 2 redesign replacing the single-table design with a proper 4-table schema:
    players          — identity, created on first appearance
    games            — Tier 1: ALL games stored unconditionally (no write filter)
    moves            — optional per-move detail (~50 KB/game), toggle with STORE_MOVE_DETAIL
    player_profiles  — incremental aggregate, updated after each game

Two-Tier Design:
    Tier 1 — Raw Evidence: ALL games, any length, no conditions.
    Tier 2 — Baseline Pool: games with quality_score > 0.5 AND classification == 'Human'.
              This is a QUERY PREDICATE, not a separate table.

Quality Score replaces the old binary n >= 44 filter.
"""

import sqlite3
import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH = Path("data/profiles.db")

# Toggle per-move detail storage (costs ~50 KB/game but enables Layer 2 recomputation)
STORE_MOVE_DETAIL = True


# ---------------------------------------------------------------------------
# Quality Score
# ---------------------------------------------------------------------------

def compute_quality_score(
    analyzed_moves: int,
    confidence: float,
    avg_c_final: float,
) -> float:
    """
    Returns quality_score in [0.0, 1.0].

    Short games contribute less but are NEVER discarded.
    Replaces the old binary n >= 44 filter.

    Weights: 50% sample size, 35% confidence, 15% positional complexity.
    """
    # Component 1: sample size (50%)
    if   analyzed_moves >= 44: w_n = 1.00
    elif analyzed_moves >= 30: w_n = 0.70
    elif analyzed_moves >= 20: w_n = 0.40
    elif analyzed_moves >= 10: w_n = 0.20
    else:                      w_n = 0.05  # stored but near-zero influence

    # Component 2: confidence (35%)
    w_conf = min(confidence / 0.70, 1.0) if confidence > 0 else 0.0

    # Component 3: positional complexity (15%)
    w_c = min(avg_c_final / 0.40, 1.0) if avg_c_final > 0 else 0.0

    return round(w_n * 0.50 + w_conf * 0.35 + w_c * 0.15, 4)


def is_baseline_eligible(
    quality_score: float,
    classification: str,
    p_cheat: float,
) -> bool:
    """
    Tier 2 eligibility — used when computing lambda_estimate.
    A game is eligible for the baseline pool iff it's high-quality and clearly human.
    """
    return (
        quality_score  >= 0.50
        and classification == "Human"
        and p_cheat        <  0.20
    )


# ---------------------------------------------------------------------------
# GameRecord dataclass (maps to `games` table + optional moves list)
# ---------------------------------------------------------------------------

@dataclass
class GameRecord:
    """
    Represents a single analysed game to be stored in the database.
    All optional fields default to None and are stored as NULL.
    """
    # ── Group A: Game Metadata ─────────────────────────────────────────────
    player_id:      str           # Platform user ID (use player_name for now)
    played_at:      str           # ISO-8601 timestamp string
    outcome:        str           # 'win' | 'loss' | 'draw' | 'unknown'
    total_moves:    int           # Total moves in full game
    analyzed_moves: int           # Moves actually analysed by Layer 1

    opponent_tier:  Optional[str]   = None  # Opponent tier if known
    avg_c_final:    Optional[float] = None  # Mean C_final — game positional complexity
    midgame_moves:  Optional[int]   = None  # Moves with C_final > 0.25

    # ── Group B: Layer 1 Metrics ───────────────────────────────────────────
    lambda_mle:        Optional[float] = None
    mean_delta:        Optional[float] = None
    near_optimal_pct:  Optional[float] = None  # Ratio of Δ ≤ 2% moves
    p_cheat:           Optional[float] = None  # P(Cheat) at end of game
    log_lr:            Optional[float] = None  # Log Likelihood Ratio
    confidence:        Optional[float] = None
    tau_mle:           Optional[float] = None  # Boltzmann temperature estimate
    classification:    Optional[str]   = None  # 'Human' | 'Suspicious' | 'Cheater'
    em_pi:             Optional[float] = None
    em_lambda_good:    Optional[float] = None
    em_lambda_blunder: Optional[float] = None
    best_dist_model:   Optional[str]   = None  # 'exponential'|'gamma'|'weibull'

    # ── Group C: Layer 2 Metrics (all NULL if Layer 2 skipped) ────────────
    l2_runs_z:            Optional[float] = None
    l2_runs_p:            Optional[float] = None
    l2_acf_rho1:          Optional[float] = None
    l2_acf_p:             Optional[float] = None
    l2_cusum_max:         Optional[float] = None
    l2_cusum_trigger:     Optional[bool]  = None
    l2_cusum_changepoint: Optional[int]   = None
    l2_cusum_p:           Optional[float] = None
    l2_cac:               Optional[float] = None
    l2_cac_p:             Optional[float] = None
    l2_entropy_h:         Optional[float] = None
    l2_entropy_p:         Optional[float] = None
    l2_fisher_chi2:       Optional[float] = None
    l2_vote_score:        Optional[float] = None
    l2_verdict:           Optional[str]   = None
    l2_tests_run:         int = 0

    # ── Group D: Quality & Eligibility (auto-computed on save) ────────────
    quality_score:        float = 0.0
    is_baseline_eligible: bool  = False
    phase_ratio:          Optional[float] = None

    # ── Auto-generated identifier ─────────────────────────────────────────
    game_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # ── Per-move detail (written to `moves` table if STORE_MOVE_DETAIL) ───
    moves: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# ProfileStore
# ---------------------------------------------------------------------------

class ProfileStore:
    """
    Two-Tier Profile Storage backed by SQLite.

    Tables:
        players         — identity (player_id, platform, skill_tier, …)
        games           — Tier 1 raw evidence (all games, all lengths)
        moves           — per-move detail (optional, toggled by STORE_MOVE_DETAIL)
        player_profiles — aggregate statistics updated after each game

    Usage:
        store = ProfileStore()
        game_id = store.save_game(record)   # always succeeds, any game length
        baseline = store.get_baseline_lambda(player_id)
    """

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._maybe_backup_old_format()
        self._init_schema()

    # ── Schema detection and migration ─────────────────────────────────────

    def _maybe_backup_old_format(self):
        """
        Detect the old single-table schema.
        If detected, rename the file to *.bak and start fresh.
        This preserves old data without blocking the new schema.
        """
        if not self.db_path.exists():
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
            # Old format has player_games but not the new `games` table
            if "player_games" in tables and "games" not in tables:
                bak = self.db_path.with_suffix(".db.bak")
                shutil.move(str(self.db_path), str(bak))
        except Exception:
            pass  # If we can't read it, just overwrite

    # ── Schema initialisation ───────────────────────────────────────────────

    _DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS players (
    player_id    TEXT PRIMARY KEY,
    platform     TEXT NOT NULL DEFAULT 'local',
    display_name TEXT,
    skill_tier   TEXT NOT NULL DEFAULT 'T3',
    created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS games (
    game_id              TEXT PRIMARY KEY,
    player_id            TEXT NOT NULL REFERENCES players(player_id),
    played_at            TIMESTAMP NOT NULL,
    outcome              TEXT NOT NULL DEFAULT 'unknown',
    total_moves          INTEGER NOT NULL,
    analyzed_moves       INTEGER NOT NULL,
    opponent_tier        TEXT,
    avg_c_final          REAL,
    midgame_moves        INTEGER,
    -- Layer 1
    lambda_mle           REAL,
    mean_delta           REAL,
    near_optimal_pct     REAL,
    p_cheat              REAL,
    log_lr               REAL,
    confidence           REAL,
    tau_mle              REAL,
    classification       TEXT,
    em_pi                REAL,
    em_lambda_good       REAL,
    em_lambda_blunder    REAL,
    best_dist_model      TEXT,
    -- Layer 2 (NULL if skipped)
    l2_runs_z            REAL,
    l2_runs_p            REAL,
    l2_acf_rho1          REAL,
    l2_acf_p             REAL,
    l2_cusum_max         REAL,
    l2_cusum_trigger     BOOLEAN,
    l2_cusum_changepoint INTEGER,
    l2_cusum_p           REAL,
    l2_cac               REAL,
    l2_cac_p             REAL,
    l2_entropy_h         REAL,
    l2_entropy_p         REAL,
    l2_fisher_chi2       REAL,
    l2_vote_score        REAL,
    l2_verdict           TEXT,
    l2_tests_run         INTEGER NOT NULL DEFAULT 0,
    -- Quality
    quality_score        REAL    NOT NULL DEFAULT 0.0,
    is_baseline_eligible BOOLEAN NOT NULL DEFAULT 0,
    phase_ratio          REAL
);

CREATE TABLE IF NOT EXISTS moves (
    move_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         TEXT    NOT NULL REFERENCES games(game_id),
    move_number     INTEGER NOT NULL,
    delta           REAL    NOT NULL,
    c_raw           REAL,
    f_opp           REAL,
    c_final         REAL    NOT NULL,
    best_wr         REAL    NOT NULL,
    played_wr       REAL    NOT NULL,
    opp_best_wr     REAL,
    is_trivial      BOOLEAN NOT NULL DEFAULT 0,
    posterior_after REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS player_profiles (
    player_id            TEXT    PRIMARY KEY REFERENCES players(player_id),
    total_games          INTEGER NOT NULL DEFAULT 0,
    total_analyzed_moves INTEGER NOT NULL DEFAULT 0,
    mean_p_cheat         REAL,
    mean_log_lr          REAL,
    suspicious_count     INTEGER NOT NULL DEFAULT 0,
    quick_win_count      INTEGER NOT NULL DEFAULT 0,
    quick_win_rate       REAL,
    baseline_games       INTEGER NOT NULL DEFAULT 0,
    lambda_estimate      REAL,
    lambda_ci_lower      REAL,
    lambda_ci_upper      REAL,
    lambda_std           REAL,
    weighted_lambda      REAL,
    mean_cac             REAL,
    mean_acf_rho1        REAL,
    l2_flag_rate         REAL,
    l2_tests_avg         REAL,
    z_score_lambda       REAL,
    profile_verdict      TEXT    NOT NULL DEFAULT 'Normal',
    last_updated         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    flagged_at           TIMESTAMP
);

-- Indices
CREATE INDEX IF NOT EXISTS idx_games_player  ON games(player_id, played_at DESC);
CREATE INDEX IF NOT EXISTS idx_games_class   ON games(classification);
CREATE INDEX IF NOT EXISTS idx_games_quality ON games(quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_moves_game    ON moves(game_id, move_number);
"""

    def _init_schema(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self._DDL)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _upsert_player(
        self,
        conn: sqlite3.Connection,
        player_id: str,
        *,
        platform: str = "local",
        display_name: Optional[str] = None,
        skill_tier: str = "T3",
    ):
        conn.execute(
            """
            INSERT OR IGNORE INTO players (player_id, platform, display_name, skill_tier)
            VALUES (?, ?, ?, ?)
            """,
            (player_id, platform, display_name, skill_tier),
        )

    def _insert_game(self, conn: sqlite3.Connection, r: GameRecord):
        conn.execute(
            """
            INSERT OR REPLACE INTO games (
                game_id, player_id, played_at, outcome, total_moves, analyzed_moves,
                opponent_tier, avg_c_final, midgame_moves,
                lambda_mle, mean_delta, near_optimal_pct, p_cheat, log_lr,
                confidence, tau_mle, classification, em_pi, em_lambda_good,
                em_lambda_blunder, best_dist_model,
                l2_runs_z, l2_runs_p, l2_acf_rho1, l2_acf_p,
                l2_cusum_max, l2_cusum_trigger, l2_cusum_changepoint, l2_cusum_p,
                l2_cac, l2_cac_p, l2_entropy_h, l2_entropy_p,
                l2_fisher_chi2, l2_vote_score, l2_verdict, l2_tests_run,
                quality_score, is_baseline_eligible, phase_ratio
            ) VALUES (
                ?,?,?,?,?,?,  ?,?,?,
                ?,?,?,?,?,  ?,?,?,?,?,  ?,?,
                ?,?,?,?,  ?,?,?,?,  ?,?,?,?,  ?,?,?,?,
                ?,?,?
            )
            """,
            (
                r.game_id, r.player_id, r.played_at, r.outcome,
                r.total_moves, r.analyzed_moves,
                r.opponent_tier, r.avg_c_final, r.midgame_moves,
                r.lambda_mle, r.mean_delta, r.near_optimal_pct, r.p_cheat, r.log_lr,
                r.confidence, r.tau_mle, r.classification, r.em_pi, r.em_lambda_good,
                r.em_lambda_blunder, r.best_dist_model,
                r.l2_runs_z, r.l2_runs_p, r.l2_acf_rho1, r.l2_acf_p,
                r.l2_cusum_max, r.l2_cusum_trigger, r.l2_cusum_changepoint, r.l2_cusum_p,
                r.l2_cac, r.l2_cac_p, r.l2_entropy_h, r.l2_entropy_p,
                r.l2_fisher_chi2, r.l2_vote_score, r.l2_verdict, r.l2_tests_run,
                r.quality_score, r.is_baseline_eligible, r.phase_ratio,
            ),
        )

    def _insert_moves(self, conn: sqlite3.Connection, game_id: str, moves: list):
        """Batch insert per-move detail rows."""
        rows = []
        for i, m in enumerate(moves):
            rows.append((
                game_id,
                i + 1,                            # move_number (1-based)
                m.get("delta", 0.0),
                m.get("c_raw"),
                m.get("f_opp"),
                m.get("c_final", 0.0),
                m.get("best_wr", 0.0),
                m.get("played_wr", 0.0),
                m.get("opp_best_wr"),
                int(m.get("is_trivial", False)),
                m.get("posterior_after", 0.0),
            ))
        conn.executemany(
            """
            INSERT INTO moves (
                game_id, move_number, delta,
                c_raw, f_opp, c_final,
                best_wr, played_wr, opp_best_wr,
                is_trivial, posterior_after
            ) VALUES (?,?,?, ?,?,?, ?,?,?, ?,?)
            """,
            rows,
        )

    def _update_profile(self, conn: sqlite3.Connection, player_id: str):
        """
        UPSERT player_profiles using aggregate query over games table.
        Runs entirely in SQL — no manual accumulation needed.
        """
        conn.execute(
            """
            INSERT INTO player_profiles (
                player_id, total_games, total_analyzed_moves,
                mean_p_cheat, mean_log_lr,
                suspicious_count, quick_win_count, quick_win_rate,
                baseline_games, weighted_lambda,
                mean_cac, mean_acf_rho1, l2_flag_rate, l2_tests_avg,
                last_updated
            )
            SELECT
                player_id,
                COUNT(*)                                        AS total_games,
                COALESCE(SUM(analyzed_moves), 0)               AS total_analyzed_moves,
                AVG(p_cheat)                                    AS mean_p_cheat,
                AVG(log_lr)                                     AS mean_log_lr,
                SUM(CASE WHEN classification IN ('Suspicious','Cheater','Suspicious (L2-flagged)',
                    'Suspicious (LRT-flagged)','Suspicious-Elevated','Manual Review')
                    THEN 1 ELSE 0 END)                          AS suspicious_count,
                SUM(CASE WHEN outcome='win' AND analyzed_moves < 20
                    THEN 1 ELSE 0 END)                          AS quick_win_count,
                CAST(
                    SUM(CASE WHEN outcome='win' AND analyzed_moves < 20 THEN 1.0 ELSE 0.0 END)
                    AS REAL
                ) / NULLIF(SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END), 0)
                                                                AS quick_win_rate,
                SUM(is_baseline_eligible)                       AS baseline_games,
                SUM(lambda_mle * quality_score)
                    / NULLIF(SUM(quality_score * is_baseline_eligible), 0)
                                                                AS weighted_lambda,
                AVG(CASE WHEN l2_cac IS NOT NULL THEN l2_cac END)      AS mean_cac,
                AVG(CASE WHEN l2_acf_rho1 IS NOT NULL THEN l2_acf_rho1 END)
                                                                AS mean_acf_rho1,
                AVG(CASE WHEN l2_verdict = 'Suspicious' THEN 1.0 ELSE 0.0 END)
                                                                AS l2_flag_rate,
                AVG(l2_tests_run)                               AS l2_tests_avg,
                CURRENT_TIMESTAMP
            FROM games
            WHERE player_id = ?
            ON CONFLICT(player_id) DO UPDATE SET
                total_games          = excluded.total_games,
                total_analyzed_moves = excluded.total_analyzed_moves,
                mean_p_cheat         = excluded.mean_p_cheat,
                mean_log_lr          = excluded.mean_log_lr,
                suspicious_count     = excluded.suspicious_count,
                quick_win_count      = excluded.quick_win_count,
                quick_win_rate       = excluded.quick_win_rate,
                baseline_games       = excluded.baseline_games,
                weighted_lambda      = excluded.weighted_lambda,
                mean_cac             = excluded.mean_cac,
                mean_acf_rho1        = excluded.mean_acf_rho1,
                l2_flag_rate         = excluded.l2_flag_rate,
                l2_tests_avg         = excluded.l2_tests_avg,
                last_updated         = CURRENT_TIMESTAMP
            """,
            (player_id,),
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def save_game(
        self,
        record: GameRecord,
        *,
        platform: str = "local",
        skill_tier: str = "T3",
    ) -> str:
        """
        Save game to Tier 1. Always succeeds regardless of game length.
        Computes quality_score and is_baseline_eligible, then:
            1. Upserts player row
            2. Inserts game row
            3. (Optional) inserts move rows
            4. Updates aggregate player_profiles row

        Returns:
            game_id (UUID string)
        """
        # Auto-compute derived fields
        record.quality_score = compute_quality_score(
            record.analyzed_moves,
            record.confidence or 0.0,
            record.avg_c_final or 0.30,
        )
        record.is_baseline_eligible = is_baseline_eligible(
            record.quality_score,
            record.classification or "",
            record.p_cheat or 1.0,
        )
        if record.analyzed_moves > 0 and record.midgame_moves is not None:
            record.phase_ratio = record.midgame_moves / record.analyzed_moves

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            self._upsert_player(
                conn, record.player_id,
                platform=platform,
                skill_tier=skill_tier,
            )
            self._insert_game(conn, record)
            if STORE_MOVE_DETAIL and record.moves:
                self._insert_moves(conn, record.game_id, record.moves)
            self._update_profile(conn, record.player_id)

        return record.game_id

    def get_baseline_lambda(self, player_id: str, window: int = 30) -> dict:
        """
        Returns weighted λ estimate from Tier 2 (baseline-eligible) games.

        Returns dict with keys:
            status:   'OK' | 'INSUFFICIENT_DATA'
            lambda:   float
            ci_lower: float
            ci_upper: float
            n_games:  int
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT lambda_mle, quality_score
                FROM   games
                WHERE  player_id = ?
                  AND  is_baseline_eligible = 1
                  AND  lambda_mle IS NOT NULL
                ORDER  BY played_at DESC
                LIMIT  ?
                """,
                (player_id, window),
            ).fetchall()

        n = len(rows)
        if n < 2:
            return {"status": "INSUFFICIENT_DATA", "n_games": n}

        lambdas = [r["lambda_mle"] for r in rows]
        weights = [r["quality_score"] for r in rows]
        total_w  = sum(weights) or 1.0
        w_lambda = sum(l * w for l, w in zip(lambdas, weights)) / total_w

        # Sample std-dev and t-CI (t_{0.025, n-1})
        mean_l = sum(lambdas) / n
        variance = sum((l - mean_l) ** 2 for l in lambdas) / (n - 1)
        std = variance ** 0.5

        # t critical values (conservative: use t_{0.025,29} = 2.045 for n >= 30, else 2.776 for n=5)
        import math
        t_crit = 2.045 if n >= 30 else 2.776 if n <= 5 else 2.0 + 0.8 / (n - 4)
        ci_half = t_crit * std / math.sqrt(n)

        return {
            "status":   "OK",
            "lambda":   round(w_lambda, 4),
            "ci_lower": round(w_lambda - ci_half, 4),
            "ci_upper": round(w_lambda + ci_half, 4),
            "std":      round(std, 4),
            "n_games":  n,
        }

    def get_player_profile(self, player_id: str) -> Optional[dict]:
        """Return current profile snapshot for a player."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM player_profiles WHERE player_id = ?",
                (player_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_all_games(
        self,
        player_id: str,
        limit: int = 100,
        baseline_only: bool = False,
    ) -> list[dict]:
        """
        Retrieve game history for a player.

        Args:
            player_id:     Player identifier (case-sensitive)
            limit:         Maximum rows to return
            baseline_only: If True, filter to Tier 2 (is_baseline_eligible=1)
        """
        where_extra = " AND is_baseline_eligible = 1" if baseline_only else ""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT * FROM games
                WHERE player_id = ?{where_extra}
                ORDER BY played_at DESC
                LIMIT ?
                """,
                (player_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_suspicious_players(self, days: int = 7) -> list[dict]:
        """
        Multi-signal anomaly detection (Plan Section 6.3).
        Returns players with ≥ 2 active signals, minimum 3 games.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    pp.player_id,
                    p.skill_tier,
                    pp.total_games,
                    pp.baseline_games,
                    pp.weighted_lambda,
                    pp.z_score_lambda,
                    pp.mean_p_cheat,
                    pp.quick_win_rate,
                    pp.l2_flag_rate,
                    pp.mean_cac,
                    pp.profile_verdict,
                    (
                        (COALESCE(pp.z_score_lambda, 0) > 2.5) +
                        (COALESCE(pp.mean_p_cheat,   0) > 0.15) +
                        (COALESCE(pp.quick_win_rate, 0) > 0.35) +
                        (COALESCE(pp.l2_flag_rate,   0) > 0.30) +
                        (COALESCE(pp.mean_cac,        0) > 0.35)
                    ) AS signal_count
                FROM player_profiles pp
                JOIN players p USING(player_id)
                WHERE (
                    pp.z_score_lambda > 2.5
                    OR pp.mean_p_cheat  > 0.15
                    OR pp.quick_win_rate > 0.35
                    OR pp.l2_flag_rate   > 0.30
                    OR pp.mean_cac       > 0.35
                )
                  AND pp.total_games >= 3
                ORDER BY signal_count DESC, pp.mean_p_cheat DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def get_quick_win_flagged(self, days: int = 30) -> list[dict]:
        """
        Quick-win rate detection (Plan Section 6.1).
        Returns players with quick_win_rate > 40% and ≥ 5 total games.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT
                    p.player_id,
                    p.skill_tier,
                    COUNT(*)                                                  AS total_games,
                    ROUND(AVG(g.analyzed_moves), 1)                          AS avg_analyzed,
                    SUM(CASE WHEN g.outcome='win' AND g.analyzed_moves < 20
                             THEN 1 ELSE 0 END)                              AS quick_wins,
                    ROUND(
                        CAST(SUM(CASE WHEN g.outcome='win' AND g.analyzed_moves < 20
                             THEN 1 ELSE 0 END) AS REAL)
                        / NULLIF(SUM(CASE WHEN g.outcome='win' THEN 1 ELSE 0 END), 0),
                    3)                                                        AS quick_win_rate,
                    ROUND(AVG(g.p_cheat), 4)                                 AS mean_p_cheat,
                    ROUND(AVG(g.log_lr),  2)                                 AS mean_log_lr
                FROM games g
                JOIN players p USING(player_id)
                WHERE g.played_at > datetime('now', '-{days} days')
                GROUP BY p.player_id
                HAVING quick_win_rate > 0.40 AND total_games >= 5
                ORDER BY quick_win_rate DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]
