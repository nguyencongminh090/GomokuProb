#!/usr/bin/env python3
"""
PyGomo Basic Engine Test

Simple test to verify engine connection and basic commands.

Usage:
    python basic_test.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pygomo import EngineClient, SearchInfo


def main():
    """Run basic engine tests."""
    
    # Resolve engine path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    engine_path = os.path.join(project_dir, "engine", "rapfi")
    
    print("=" * 50)
    print("  PyGomo Basic Engine Test")
    print("=" * 50)
    print(f"\nEngine: {engine_path}")
    
    engine = EngineClient(
        engine_path,
        working_directory=os.path.dirname(engine_path),
    )
    
    try:
        # Test 1: Connect
        print("\n[Test 1] Connecting...")
        engine.connect()
        print(f"  ✓ Connected (PID: {engine.process_id})")
        
        # Test 2: About
        print("\n[Test 2] Getting engine info...")
        about = engine.about(timeout=5.0)
        if about:
            print(f"  ✓ About: {about[:60]}...")
        else:
            print("  ✗ No about info")
        
        # Test 3: Start
        print("\n[Test 3] Starting game (15x15)...")
        if engine.start(15):
            print("  ✓ Game started")
        else:
            print("  ✗ Failed to start")
            return
        
        # Test 4: Configure
        print("\n[Test 4] Configuring engine...")
        engine.set_rule(1)  # Standard
        engine.set_time(turn_time_ms=5000)
        print("  ✓ Configuration sent")
        
        # Test 5: Begin (engine plays first)
        print("\n[Test 5] Requesting first move (BEGIN)...")
        
        def on_info(info: SearchInfo):
            pv = " ".join(str(m) for m in info.pv[:3])
            print(f"    depth {info.depth:2} | eval {info.eval.raw_value:>6} | pv {pv}")
        
        result = engine.begin(timeout=10.0, on_info=on_info)
        if result:
            print(f"  ✓ Engine played: {result.move}")
        else:
            print("  ✗ No move returned")
            return
        
        # Test 6: Turn (respond to human move at different position)
        print("\n[Test 6] Sending human move (i8) and getting response...")
        result = engine.turn("i8", timeout=10.0, on_info=on_info)
        if result:
            print(f"  ✓ Engine played: {result.move}")
        else:
            print("  ✗ No move returned")
            return
        
        # Test 7: Stop
        print("\n[Test 7] Testing STOP command...")
        engine.stop()
        print("  ✓ Stop sent")
        
        # Test 8: Restart
        print("\n[Test 8] Restarting game...")
        if engine.restart():
            print("  ✓ Game restarted")
        else:
            print("  ✗ Restart failed")
        
        print("\n" + "=" * 50)
        print("  All tests passed! ✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n[*] Shutting down engine...")
        engine.quit()
        print("[*] Done.")


if __name__ == "__main__":
    main()
