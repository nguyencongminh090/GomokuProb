#!/usr/bin/env python3
"""Debug script to capture raw engine output."""

import sys
import os
import subprocess
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
engine_path = os.path.join(project_dir, "engine", "rapfi")

print("Starting engine to capture raw output...")

# Start engine directly
process = subprocess.Popen(
    [engine_path],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=os.path.dirname(engine_path),
    text=True,
    bufsize=1,
)

def send(cmd):
    print(f">>> {cmd}")
    process.stdin.write(cmd + "\n")
    process.stdin.flush()

def read_for(seconds):
    print(f"--- Reading for {seconds}s ---")
    start = time.time()
    while time.time() - start < seconds:
        line = ""
        try:
            import select
            if select.select([process.stdout], [], [], 0.1)[0]:
                line = process.stdout.readline()
                if line:
                    print(f"<<< {repr(line.strip())}")
        except:
            time.sleep(0.1)

# Sequence
send("START 15")
read_for(1)

send("INFO timeout_turn 5000")
send("INFO rule 1")
read_for(0.5)

send("BEGIN")
read_for(6)

# Play a move and see engine's thinking
send("TURN 8,7")  # Play near center
read_for(8)

send("END")
time.sleep(0.5)
process.terminate()
print("\nDone!")
