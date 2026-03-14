"""
tests/test_profile_store.py — Sprint 2 Profile Database Tests

Covers Sprint 2A/2B/2C acceptance criteria:
  T-DB-01 to T-TS-05 from the implementation plan.

Uses Python 3.13. No external dependencies (sqlite3, unittest, tempfile only).
"""

import os
import sys
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.profile_store import (
    ProfileStore,
    GameRecord,
    compute_quality_score,
    is_baseline_eligible,
    STORE_MOVE_DETAIL,
)
from src.core.profile_analyzer import ProfileAnalyzer, ProfileResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(**overrides) -> GameRecord:
    """Return a minimal valid GameRecord with sensible defaults."""
    defaults = dict(
        player_id      = "test_player",
        played_at      = "2025-01-01T12:00:00",
        outcome        = "win",
        total_moves    = 50,
        analyzed_moves = 40,
        avg_c_final    = 0.35,
        midgame_moves  = 20,
        lambda_mle     = 30.0,
        mean_delta     = 0.03,
        near_optimal_pct = 0.70,
        p_cheat        = 0.05,
        log_lr         = -1.2,
        confidence     = 0.80,
        tau_mle        = 0.10,
        classification = "Human",
        best_dist_model = "exponential",
    )
    defaults.update(overrides)
    return GameRecord(**defaults)


def _temp_store() -> ProfileStore:
    """Return a ProfileStore backed by a fresh temp-file DB."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return ProfileStore(path)


# ---------------------------------------------------------------------------
# T-DB-02: compute_quality_score()
# ---------------------------------------------------------------------------

class TestComputeQualityScore(unittest.TestCase):
    """Sprint 2A — T-DB-02: quality score formula with 5 representative cases."""

    def _score(self, n: int, conf: float = 0.80, c: float = 0.35) -> float:
        return compute_quality_score(n, conf, c)

    def test_n5_very_low(self):
        """n=5: w_n=0.05 → score must be below 0.60 and well below n=10 score."""
        s = self._score(5)
        self.assertLess(s, 0.60)
        # n=5 score must be less than n=15 score (monotone)
        self.assertLess(s, self._score(15))

    def test_n15_low(self):
        """n=15: w_n=0.20 → score clearly below n=30 score."""
        s = self._score(15)
        self.assertGreater(s, 0.10)
        self.assertLess(s, self._score(30))

    def test_n25_medium(self):
        """n=25: w_n=0.40 → score above n=10 but below n=44."""
        s = self._score(25)
        self.assertGreater(s, self._score(10))
        self.assertLess(s, self._score(44))

    def test_n35_high(self):
        """n=35: w_n=0.70 → score roughly 0.65+."""
        s = self._score(35)
        self.assertGreater(s, 0.60)

    def test_n50_max_eligible(self):
        """n=50: w_n=1.00, conf=0.80, c=0.35 → near-max score."""
        s = self._score(50)
        self.assertGreater(s, 0.85)
        self.assertLessEqual(s, 1.0)

    def test_zero_confidence_does_not_blow_up(self):
        """conf=0 → zero confidence component, score still valid."""
        s = compute_quality_score(44, 0.0, 0.35)
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)

    def test_monotone_in_n(self):
        """Score must be non-decreasing as n jumps across thresholds."""
        ns = [3, 10, 15, 25, 35, 44, 60]
        scores = [self._score(n) for n in ns]
        for i in range(len(scores) - 1):
            self.assertLessEqual(scores[i], scores[i + 1] + 1e-9)


# ---------------------------------------------------------------------------
# T-DB-02 (B): is_baseline_eligible()
# ---------------------------------------------------------------------------

class TestIsBaselineEligible(unittest.TestCase):

    def test_fully_eligible(self):
        self.assertTrue(is_baseline_eligible(0.80, "Human", 0.05))

    def test_not_human_ineligible(self):
        self.assertFalse(is_baseline_eligible(0.80, "Suspicious", 0.05))

    def test_high_p_cheat_ineligible(self):
        self.assertFalse(is_baseline_eligible(0.80, "Human", 0.25))

    def test_low_quality_ineligible(self):
        self.assertFalse(is_baseline_eligible(0.40, "Human", 0.05))

    def test_borderline_quality(self):
        # Exactly 0.50 should be eligible
        self.assertTrue(is_baseline_eligible(0.50, "Human", 0.10))

    def test_borderline_p_cheat(self):
        # p_cheat == 0.20 is NOT eligible (strict <)
        self.assertFalse(is_baseline_eligible(0.80, "Human", 0.20))


# ---------------------------------------------------------------------------
# T-DB-03: save_game() — always succeeds regardless of game length
# ---------------------------------------------------------------------------

class TestSaveGame(unittest.TestCase):

    def setUp(self):
        self.store = _temp_store()

    def test_5_move_game_does_not_raise(self):
        """T-DB-03: 5-move game must save without exception."""
        r = _make_record(analyzed_moves=5, total_moves=5)
        game_id = self.store.save_game(r)
        self.assertIsInstance(game_id, str)
        self.assertEqual(len(game_id), 36)  # UUID form

    def test_returns_uuid(self):
        r = _make_record()
        game_id = self.store.save_game(r)
        import uuid
        uuid.UUID(game_id)  # must not raise

    def test_short_game_quality_is_low_relative(self):
        """n=5 game has lower quality than n=44 game."""
        r_short = _make_record(analyzed_moves=5, confidence=0.90)
        r_long  = _make_record(analyzed_moves=44, confidence=0.90)
        self.store.save_game(r_short)
        self.store.save_game(r_long)
        games = self.store.get_all_games("test_player")
        qs = sorted([g["quality_score"] for g in games])
        self.assertLess(qs[0], qs[1])  # short < long

    def test_long_human_game_is_baseline_eligible(self):
        """High-quality Human game must be marked is_baseline_eligible=1."""
        r = _make_record(
            analyzed_moves=50,
            confidence=0.85,
            avg_c_final=0.40,
            p_cheat=0.03,
            classification="Human",
        )
        self.store.save_game(r)
        games = self.store.get_all_games("test_player")
        self.assertEqual(games[0]["is_baseline_eligible"], 1)

    def test_cheater_game_not_baseline_eligible(self):
        r = _make_record(
            analyzed_moves=50,
            confidence=0.80,
            p_cheat=0.75,
            classification="Cheater",
        )
        self.store.save_game(r)
        games = self.store.get_all_games("test_player")
        self.assertEqual(games[0]["is_baseline_eligible"], 0)

    def test_multiple_players_isolated(self):
        """Games for different players must not bleed into each other."""
        self.store.save_game(_make_record(player_id="alice", lambda_mle=20.0))
        self.store.save_game(_make_record(player_id="bob",   lambda_mle=80.0))
        alice = self.store.get_all_games("alice")
        bob   = self.store.get_all_games("bob")
        self.assertEqual(len(alice), 1)
        self.assertEqual(len(bob),   1)
        self.assertAlmostEqual(alice[0]["lambda_mle"], 20.0)
        self.assertAlmostEqual(bob[0]["lambda_mle"],   80.0)


# ---------------------------------------------------------------------------
# T-DB-04: _update_profile() incrementally tracks total_games
# ---------------------------------------------------------------------------

class TestUpdateProfile(unittest.TestCase):

    def setUp(self):
        self.store = _temp_store()

    def test_total_games_increments(self):
        """T-DB-04: total_games in player_profiles must grow with each save."""
        for i in range(3):
            self.store.save_game(_make_record())
        profile = self.store.get_player_profile("test_player")
        self.assertEqual(profile["total_games"], 3)

    def test_suspicious_count_accumulates(self):
        self.store.save_game(_make_record(classification="Human"))
        self.store.save_game(_make_record(classification="Suspicious"))
        self.store.save_game(_make_record(classification="Cheater"))
        p = self.store.get_player_profile("test_player")
        self.assertEqual(p["suspicious_count"], 2)

    def test_quick_win_count(self):
        """Win with analyzed_moves < 20 counts as quick_win."""
        self.store.save_game(_make_record(outcome="win", analyzed_moves=15))
        self.store.save_game(_make_record(outcome="win", analyzed_moves=40))
        p = self.store.get_player_profile("test_player")
        self.assertEqual(p["quick_win_count"], 1)

    def test_baseline_games_count(self):
        """baseline_games = number of is_baseline_eligible games."""
        # This game IS eligible
        self.store.save_game(_make_record(
            analyzed_moves=50, confidence=0.85, avg_c_final=0.40,
            p_cheat=0.03, classification="Human",
        ))
        # This one is NOT — force confidence=0 so quality is very low and ineligible
        self.store.save_game(_make_record(
            analyzed_moves=5, confidence=0.0, avg_c_final=0.0,
            p_cheat=0.01, classification="Human",
        ))
        p = self.store.get_player_profile("test_player")
        self.assertEqual(p["baseline_games"], 1)


# ---------------------------------------------------------------------------
# T-DB-05: get_baseline_lambda() — CI computation
# ---------------------------------------------------------------------------

class TestGetBaselineLambda(unittest.TestCase):

    def setUp(self):
        self.store = _temp_store()

    def _add_eligible(self, lam: float):
        self.store.save_game(_make_record(
            analyzed_moves = 50,
            confidence     = 0.85,
            avg_c_final    = 0.40,
            p_cheat        = 0.03,
            classification = "Human",
            lambda_mle     = lam,
        ))

    def test_insufficient_with_one_game(self):
        """T-DB-05: Returns INSUFFICIENT_DATA when n_baseline < 2."""
        self._add_eligible(30.0)
        result = self.store.get_baseline_lambda("test_player")
        self.assertEqual(result["status"], "INSUFFICIENT_DATA")

    def test_ok_with_two_games(self):
        """With n=2 eligible games, status must be 'OK'."""
        self._add_eligible(28.0)
        self._add_eligible(32.0)
        result = self.store.get_baseline_lambda("test_player")
        self.assertEqual(result["status"], "OK")
        self.assertAlmostEqual(result["lambda"], 30.0, places=1)

    def test_ci_bounds_contain_mean(self):
        """CI lower < lambda < CI upper."""
        for lam in [25.0, 28.0, 30.0, 32.0, 35.0]:
            self._add_eligible(lam)
        result = self.store.get_baseline_lambda("test_player")
        self.assertLess(result["ci_lower"], result["lambda"])
        self.assertGreater(result["ci_upper"], result["lambda"])

    def test_non_eligible_games_excluded(self):
        """Short/Suspicious games must not influence baseline lambda."""
        self._add_eligible(30.0)
        self._add_eligible(30.0)
        # Non-eligible game with very different lambda
        self.store.save_game(_make_record(
            analyzed_moves=5, classification="Cheater", lambda_mle=200.0
        ))
        result = self.store.get_baseline_lambda("test_player")
        self.assertEqual(result["status"], "OK")
        self.assertAlmostEqual(result["lambda"], 30.0, places=1)

    def test_missing_player_returns_insufficient(self):
        result = self.store.get_baseline_lambda("nobody")
        self.assertEqual(result["status"], "INSUFFICIENT_DATA")


# ---------------------------------------------------------------------------
# T-DB-06: get_player_profile() — weighted_lambda
# ---------------------------------------------------------------------------

class TestGetPlayerProfile(unittest.TestCase):

    def setUp(self):
        self.store = _temp_store()

    def test_returns_none_for_unknown_player(self):
        self.assertIsNone(self.store.get_player_profile("ghost"))

    def test_profile_created_after_first_game(self):
        self.store.save_game(_make_record())
        profile = self.store.get_player_profile("test_player")
        self.assertIsNotNone(profile)
        self.assertEqual(profile["total_games"], 1)


# ---------------------------------------------------------------------------
# T-DB-07: moves table — rows inserted iff STORE_MOVE_DETAIL=True
# ---------------------------------------------------------------------------

class TestMovesTable(unittest.TestCase):

    def setUp(self):
        self.store = _temp_store()

    def test_moves_inserted_when_enabled(self):
        """Moves list → rows in `moves` table when STORE_MOVE_DETAIL=True."""
        moves = [
            {"delta": 0.02, "c_final": 0.35, "best_wr": 0.70,
             "played_wr": 0.68, "is_trivial": False, "posterior_after": 0.05},
            {"delta": 0.01, "c_final": 0.40, "best_wr": 0.72,
             "played_wr": 0.71, "is_trivial": False, "posterior_after": 0.04},
        ]
        r = _make_record(moves=moves)
        game_id = self.store.save_game(r)

        with sqlite3.connect(self.store.db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM moves WHERE game_id = ?", (game_id,)
            ).fetchone()[0]

        if STORE_MOVE_DETAIL:
            self.assertEqual(count, 2)
        else:
            self.assertEqual(count, 0)

    def test_move_numbers_are_1_indexed(self):
        moves = [
            {"delta": 0.01, "c_final": 0.3, "best_wr": 0.6,
             "played_wr": 0.59, "is_trivial": False, "posterior_after": 0.05},
        ]
        r = _make_record(moves=moves)
        game_id = self.store.save_game(r)

        if not STORE_MOVE_DETAIL:
            return  # not relevant

        with sqlite3.connect(self.store.db_path) as conn:
            mn = conn.execute(
                "SELECT move_number FROM moves WHERE game_id = ?", (game_id,)
            ).fetchone()[0]
        self.assertEqual(mn, 1)


# ---------------------------------------------------------------------------
# T-TS-05: ProfileAnalyzer — verdict escalation
# ---------------------------------------------------------------------------

class TestProfileAnalyzer(unittest.TestCase):

    def _history(self, lambdas: list[float]) -> list[dict]:
        return [{"lambda_mle": l} for l in lambdas]

    def test_insufficient_data_below_min(self):
        result = ProfileAnalyzer.analyze_suspect_game(50.0, [])
        self.assertEqual(result.status, "INSUFFICIENT_DATA")

    def test_normal_within_baseline(self):
        history = self._history([28.0, 30.0, 32.0, 29.0, 31.0])
        result  = ProfileAnalyzer.analyze_suspect_game(30.0, history)
        self.assertEqual(result.status, "NORMAL")

    def test_suspicious_extreme_lambda(self):
        """Very high λ (engine-like, near-zero deltas) → SUSPICIOUS."""
        history = self._history([28.0, 30.0, 32.0, 29.0, 31.0] * 4)
        result  = ProfileAnalyzer.analyze_suspect_game(300.0, history)
        self.assertEqual(result.status, "SUSPICIOUS")

    def test_p_value_between_0_and_1(self):
        history = self._history([28.0, 30.0, 32.0, 29.0, 31.0])
        result  = ProfileAnalyzer.analyze_suspect_game(35.0, history)
        if result.p_value is not None:
            self.assertGreaterEqual(result.p_value, 0.0)
            self.assertLessEqual(result.p_value,    1.0)

    def test_z_test_used_for_m_ge_30(self):
        """m >= 30 uses Z-test → critical_05 == 1.645."""
        history = self._history([30.0] * 30)
        result  = ProfileAnalyzer.analyze_suspect_game(31.0, history)
        self.assertAlmostEqual(result.critical_05, 1.645, places=2)

    # compute_profile_verdict
    def test_verdict_normal_no_signals(self):
        profile = {"z_score_lambda": 0.5, "mean_p_cheat": 0.02}
        v = ProfileAnalyzer.compute_profile_verdict(profile)
        self.assertEqual(v, "Normal")

    def test_verdict_monitoring_two_signals(self):
        profile = {
            "z_score_lambda": 2.6,   # active (> 2.5)
            "mean_p_cheat":   0.20,  # active (> 0.15)
            "quick_win_rate": 0.10,
            "l2_flag_rate":   0.10,
            "mean_cac":       0.10,
        }
        v = ProfileAnalyzer.compute_profile_verdict(profile)
        self.assertEqual(v, "Monitoring")

    def test_verdict_flagged_three_signals(self):
        profile = {
            "z_score_lambda": 2.6,   # active
            "mean_p_cheat":   0.20,  # active
            "quick_win_rate": 0.40,  # active
            "l2_flag_rate":   0.10,
            "mean_cac":       0.10,
        }
        v = ProfileAnalyzer.compute_profile_verdict(profile)
        self.assertEqual(v, "Flagged")

    def test_verdict_flagged_extreme_z(self):
        profile = {"z_score_lambda": 4.0}  # z > 3.0 → Flagged alone
        v = ProfileAnalyzer.compute_profile_verdict(profile)
        self.assertEqual(v, "Flagged")

    def test_verdict_none_profile(self):
        self.assertEqual(ProfileAnalyzer.compute_profile_verdict(None), "Normal")


# ---------------------------------------------------------------------------
# Old-format detection: backup and start fresh
# ---------------------------------------------------------------------------

class TestOldFormatAutoBackup(unittest.TestCase):

    def test_old_format_renamed_to_bak(self):
        """If an old-style player_games table is found, it should be renamed .bak."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        path = Path(path)

        # Create old-format DB
        with sqlite3.connect(path) as conn:
            conn.execute("""
                CREATE TABLE player_games (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_name TEXT, lambda_mle REAL, is_baseline BOOLEAN
                )
            """)
            conn.execute("INSERT INTO player_games VALUES (NULL,'alice',30.0,1)")

        # Opening the new ProfileStore should detect and backup
        store = ProfileStore(path)

        bak = path.with_suffix(".db.bak")
        self.assertTrue(bak.exists(), "Old DB should have been renamed to .bak")

        # New DB must have the new schema tables
        with sqlite3.connect(path) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        self.assertIn("games",          tables)
        self.assertIn("player_profiles", tables)
        self.assertNotIn("player_games", tables)

        # Cleanup
        path.unlink(missing_ok=True)
        bak.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
