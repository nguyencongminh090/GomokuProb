import unittest
import math
from src.core.profile_analyzer import ProfileAnalyzer, ProfileResult
from src.core.profile_store import ProfileRecord

class TestProfileAnalyzer(unittest.TestCase):
    
    def setUp(self):
        # Helper to create dummy records quickly
        self.create_record = lambda l: ProfileRecord(
            player_name="test_player",
            timestamp="2024-01-01T12:00:00",
            lambda_mle=l,
            mean_delta=0.01,
            num_moves=45,
            skill_tier="T_Unassigned",
            lambda_human=10.0,
            classification="Human",
            confidence=0.9,
            is_baseline=True
        )

    def test_insufficient_data(self):
        # m = 0
        res = ProfileAnalyzer.analyze_suspect_game(50.0, [])
        self.assertEqual(res.status, "INSUFFICIENT_DATA")
        
        # m = 1
        res = ProfileAnalyzer.analyze_suspect_game(50.0, [self.create_record(15.0)])
        self.assertEqual(res.status, "INSUFFICIENT_DATA")

    def test_t_test_small_sample(self):
        # m = 5, which < 30 (T-test)
        # lambdas = [8, 9, 10, 11, 12] -> lambda_bar = 10, s_lambda = 1.58
        history = [self.create_record(float(v)) for v in [8, 9, 10, 11, 12]]
        
        # Suspect lambda = 50 (Huge anomaly)
        res = ProfileAnalyzer.analyze_suspect_game(50.0, history)
        
        self.assertEqual(res.status, "SUSPICIOUS")
        self.assertIsNotNone(res.test_statistic)
        self.assertTrue(res.test_statistic > res.critical_01)
        self.assertEqual(res.m_games, 5)
        
        # Normal game: suspect = 10.5
        res_normal = ProfileAnalyzer.analyze_suspect_game(10.5, history)
        self.assertEqual(res_normal.status, "NORMAL")
        self.assertTrue(res_normal.test_statistic <= res_normal.critical_05)

    def test_z_test_large_sample(self):
        # m = 30, which >= 30 (Z-test)
        # Create a mix of values to get a real variance: 15 zeros, 15 twenties => mean=10, variance ~ 103.4, std_dev ~ 10.17
        history = [self.create_record(0.0) for _ in range(15)] + [self.create_record(20.0) for _ in range(15)]
        
        # Suspect lambda = 50. 
        # Mean = 10. s_lambda = 10.17. m = 30. std_err = 10.17 / sqrt(30) = 1.85
        # Z = (50 - 10) / 1.85 = 21.6. Expected to be > critical_01 (2.326)
        res = ProfileAnalyzer.analyze_suspect_game(50.0, history)
        self.assertEqual(res.m_games, 30)
        self.assertEqual(res.status, "SUSPICIOUS")
        
        # check critical values for Z test
        self.assertAlmostEqual(res.critical_05, 1.645, places=3)
        self.assertAlmostEqual(res.critical_01, 2.326, places=3)
        self.assertTrue(res.test_statistic > 20.0)

    def test_t_vs_z_critical_values(self):
        # Ensure that T-test is actually used for m=2
        history_2 = [self.create_record(10.0), self.create_record(20.0)]
        res_t = ProfileAnalyzer.analyze_suspect_game(15.0, history_2)
        # df = 1. T-critical for 95% 1-sided is ~6.314
        self.assertAlmostEqual(res_t.critical_05, 6.3137, places=3)
        
        # Ensure Z-test is used for m=30
        history_30 = [self.create_record(10.0)] * 30
        res_z = ProfileAnalyzer.analyze_suspect_game(15.0, history_30)
        self.assertAlmostEqual(res_z.critical_05, 1.645, places=3)


import os
import sqlite3
from src.core.profile_store import ProfileStore

class TestProfileStore(unittest.TestCase):
    
    def setUp(self):
        self.test_db = "test_profiles.db"
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        self.store = ProfileStore(self.test_db)
        
    def tearDown(self):
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
            
    def test_num_moves_filter(self):
        # 1. Add game with num_moves < 44
        rec_short = ProfileRecord(
            player_name="short_game_player",
            timestamp="2024-01-01T12:00:00",
            lambda_mle=15.0,
            mean_delta=0.02,
            num_moves=30,  # < 44
            skill_tier="T_Intermediate",
            lambda_human=15.0,
            classification="Human",
            confidence=0.90,
            is_baseline=True # Even if true, get_baseline_games should filter it
        )
        self.store.add_game(rec_short)
        
        # 2. Add game with num_moves >= 44
        rec_long = ProfileRecord(
            player_name="long_game_player",
            timestamp="2024-01-01T12:00:00",
            lambda_mle=12.0,
            mean_delta=0.015,
            num_moves=50,  # >= 44
            skill_tier="T_Advanced",
            lambda_human=20.0,
            classification="Human",
            confidence=0.85,
            is_baseline=True
        )
        self.store.add_game(rec_long)
        
        # Verify filtering
        short_baselines = self.store.get_baseline_games("short_game_player")
        self.assertEqual(len(short_baselines), 0, "Game with num_moves < 44 should be filtered out")
        
        long_baselines = self.store.get_baseline_games("long_game_player")
        self.assertEqual(len(long_baselines), 1, "Game with num_moves >= 44 should be included")


if __name__ == '__main__':
    unittest.main()
