#!/usr/bin/env python3
"""
PyGomo Position Test

Test starting engine from a specific position using BOARD command.

Usage:
    python position_test.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pygomo import EngineClient, SearchInfo, Move, BoardPosition


def main():
    """Test position setup and engine response."""
    
    # Resolve engine path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    engine_path = os.path.join(project_dir, "engine", "rapfi")
    
    print("=" * 50)
    print("  PyGomo Position Test")
    print("=" * 50)
    
    # Create a test position
    # Classic opening: black h8, white i9, black h9
    position = BoardPosition()
    position.add_move(Move("h8"), BoardPosition.SELF)      # Black
    position.add_move(Move("i9"), BoardPosition.OPPONENT)  # White
    position.add_move(Move("h9"), BoardPosition.SELF)      # Black
    
    print("\nTest Position:")
    print("  1. Black: h8")
    print("  2. White: i9")
    print("  3. Black: h9")
    print("\nEngine to play as White...\n")
    
    # Start engine
    engine = EngineClient(
        engine_path,
        working_directory=os.path.dirname(engine_path),
    )
    
    try:
        engine.connect()
        print(f"[*] Connected to engine (PID: {engine.process_id})")
        
        # Start game
        engine.start(15)
        engine.set_rule(1)  # Standard
        engine.set_time(turn_time_ms=10000)
        
        print("[*] Sending position to engine...")
        
        def on_info(info: SearchInfo):
            pv = " ".join(str(m) for m in info.pv[:5])
            print(f"  depth {info.depth:2} | eval {info.eval.raw_value:>6} | "
                  f"WR {info.winrate_percent:5.1f}% | pv {pv}")
        
        # Send position and get response
        result = engine.board(
            position,
            start_thinking=True,
            timeout=15.0,
            on_info=on_info,
        )
        
        if result:
            print(f"\n[*] Engine response: {result.move}")
            print(f"[*] Final eval: {result.eval.raw_value if result.eval else 'N/A'}")
            print(f"[*] Final winrate: {result.winrate:.1%}" if result.winrate else "")
        else:
            print("[!] No response from engine")
            
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        engine.quit()
        print("\n[*] Done.")


if __name__ == "__main__":
    main()
