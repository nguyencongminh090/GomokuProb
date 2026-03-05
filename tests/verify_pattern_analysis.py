import sys
import os

# Add project root to path
# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "PyGomo", "src"))

from src.core.pattern_analysis import PatternAnalyzer
from src.core.board import Move
from src.core.rapfi_patterns import RapfiPattern

def test_pattern_analysis():
    print("=== Testing PatternAnalyzer ===")
    analyzer = PatternAnalyzer()
    
    # Mock Rapfi JSON Output (15x15 board)
    # Let's say at (5,5) we have a FLEX3 (Code 6) for Black
    # and nothing for White.
    
    rows_black = [[0]*15 for _ in range(15)]
    rows_white = [[0]*15 for _ in range(15)]
    
    # Set pattern at 5,5 (Row 5, Col 5)
    rows_black[5][5] = RapfiPattern.H_FLEX3 # 6
    rows_white[5][5] = RapfiPattern.NONE    # 0
    
    mock_static_eval = {
        "static_eval": 100,
        "patterns": {
            "black": rows_black,
            "white": rows_white
        }
    }
    
    # Case 1: Black moves at 5,5
    # Should detect Offensive FLEX3
    move1 = Move(5, 5, 1, "F6") # 1=Black
    print(f"Analyzing Move: {move1.notation} (Black)")
    
    ctx1 = analyzer.analyze_move(mock_static_eval, move1, is_black=True)
    
    print("Context 1:", ctx1)
    
    assert ctx1['own_pattern'] == RapfiPattern.H_FLEX3
    assert ctx1['own_desc'] == "H_FLEX3"
    assert ctx1['opp_pattern'] == RapfiPattern.NONE
    
    # Case 2: White moves at 5,5 (Blocking)
    # Should detect Defensive FLEX3 (Block)
    print(f"\nAnalyzing Move: {move1.notation} (White blocking)")
    
    # For White, "own" is White patterns (0), "opp" is Black patterns (6)
    ctx2 = analyzer.analyze_move(mock_static_eval, move1, is_black=False)
    
    print("Context 2:", ctx2)
    
    assert ctx2['own_pattern'] == RapfiPattern.NONE
    assert ctx2['opp_pattern'] == RapfiPattern.H_FLEX3
    assert ctx2['opp_desc'] == "H_FLEX3"
    
    # Check Stats
    print("\nSummary:")
    print(analyzer.get_summary())
    
    print("\n✅ Verification Successful!")

if __name__ == "__main__":
    test_pattern_analysis()
