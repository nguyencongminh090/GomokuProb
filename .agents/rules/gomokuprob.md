---
description: GomokuProb project rules — Gomoku Anti-Cheating Analysis Tool
---

# GomokuProb Agent Rules

You are working on **GomokuProb**, a Gomoku anti-cheating analysis desktop application.
This tool determines whether a player used engine assistance by applying **Bayesian statistical inference** to move quality data.

## Project Identity

- **Purpose:** Detect engine-assisted play in Gomoku games using probability & statistics
- **Stack:** Python 3.11+ / PyQt6 (GUI) / Rapfi engine (C++ AI) / PyGomo (protocol library)
- **Architecture:** MVVM — `src/core/` (models) → `src/services/` (infrastructure) → `src/ui/` (presentation)

## Critical Architecture Rules

### Layer Isolation
- `src/core/` modules MUST NOT import from `src/ui/` or `src/services/` (except `v2_worker.py` which bridges core↔services)
- `src/ui/` communicates with `src/core/` ONLY through the `MainViewModel` in `src/ui/viewmodel.py`
- `src/services/engine_service.py` is the ONLY module that communicates with the Rapfi engine process

### Threading Model
- The **analysis worker** (`v2_worker.py`) runs in a QThread. It emits `pyqtSignal` to communicate results
- Engine I/O is blocking and wrapped in the worker thread — NEVER call engine methods from the main Qt thread
- UI updates from engine callbacks MUST go through `sig_log_message` signal for thread safety

### Engine Communication
- Engine communicates via **Gomocup protocol** (stdin/stdout text commands)
- Positions are sent via `YXBOARD` + move list + `DONE`
- Candidate moves use `nbest_time_limited()` which sends `YXNBEST` and auto-sends `STOP` after timeout
- The engine's winrate is from the opponent's perspective — always invert: `player_wr = 1.0 - opponent_wr`
- After each `YXBOARD`/`DONE`, always call `_reset_engine()` before the next search
- Static pattern data uses `YXSTATICJSON` which returns a single JSON line

## Mathematical Model — Do NOT Break These Invariants

### Delta Model (delta_model.py)
- Δ_i = W_best - W_played, ALWAYS ≥ 0 (enforced by `max(0.0, delta)`)
- Δ ~ Exponential(λ), MLE: λ̂ = 1/mean(Δ)
- Use epsilon = 0.001 for zero-delta handling to avoid division by zero
- Higher λ = smaller deltas = more suspicious

### Bayesian Updates (player_model.py)
- P(Cheat | D) uses log-space computation with log-sum-exp normalization
- Online updates: posterior becomes next prior, weighted by complexity
- Complexity < 0.2 → SKIP update entirely (trivial positions don't count)
- Forcing/winning moves → reduce P(Cheat) weight retroactively (tactical = less suspicious)

### Complexity (complexity.py)
- 4 factors: Impact(0.15) + Variance(0.40) + Criticality(0.20) + Phase(0.25)
- Variance is the most important factor — it determines decision difficulty
- Trivially won/lost positions (WR > 95% AND variance < 5%) → complexity = 0.10

### Accuracy (complexity.py)
- Base: accuracy = exp(-5 × Δ × complexity_factor)
- Near-optimal (Δ ≤ 0.02) → accuracy = 1.0
- Strategic bonus up to 15% for creating opponent complexity (only if base > 0.4)

### Information Theory (information_theory.py)
- Shannon entropy uses log base 2 (bits)
- KL divergence: D_KL(Player || Engine) — low = suspicious
- Always use epsilon = 1e-10 to avoid log(0)

### Temporal Model (temporal_model.py)
- Moving average window for smooth trend analysis
- Switch-point = delta_before - delta_after exceeds threshold
- Positive change = player improved (suspicious if sudden)
- Merge nearby switch points within min_gap = 5 moves

## Coding Conventions

### Data Classes
- ALL model data uses `@dataclass` — never raw dicts for domain objects
- `MoveAnalysis`, `GameAnalysis`, `V2MoveResult`, `V2GameResult` are the key types
- Config uses `AnalysisConfig` dataclass with JSON serialization

### Naming
- `wr` = winrate (0.0 to 1.0, NOT percentage)
- `delta` = winrate loss (always ≥ 0)
- `p_cheat` = posterior probability of cheating
- `lambda_*` = Exponential distribution rate parameter
- Pattern codes follow `RapfiPattern` IntEnum (0=NONE through 13=A_FIVE)
- Move notation is algebraic: lowercase letter + number (e.g., "h8")

### Error Handling
- Engine calls can fail (process crash, timeout) — always wrap in try/except
- Return sensible defaults: winrate = 0.5, delta = 0.0, complexity = 0.5
- Check `self.is_running` in loops to support graceful cancellation

### Logging
- Use `self.engine._log(type_, msg)` with types: "INFO", "CMD", "STAGE", "ERROR", "DEBUG"
- "STAGE" messages route to Analysis Flow tab — use `===` headers for section breaks
- ALL logging goes through thread-safe signal, never print() to console

## File Formats
- Game input: SGF (`;B[hh];W[ih]...`), raw algebraic strings ("h8 i9 h10"), or coordinate pairs
- Config: `config.json` at project root
- Logs: `logs/analysis_YYYYMMDD_HHMMSS.txt`
- Engine binaries: `engine/` directory with `.bin.lz4` neural network weights

## Testing
- Tests in `tests/` — run with `python -m pytest tests/`
- `test_math_prob.py` tests Kalman filter and delta calculations
- `test_parser.py` tests coordinate parsing and format detection
- `verify_pattern_analysis.py` validates pattern analysis against known positions
- When modifying math modules, always verify against test_math_prob.py

## What NOT To Do
1. **Never change the Bayesian update formula** without understanding the mathematical implications — the log-sum-exp normalization is critical for numerical stability
2. **Never call engine methods from the UI thread** — this will freeze the entire application
3. **Never modify `lib/PyGomo/`** without updating `docs/` — it's a shared library
4. **Never use raw threading** — always use QThread + pyqtSignal for Qt compatibility
5. **Never assume winrate perspective** — engine returns OPPONENT's winrate, always invert
6. **Never skip the complexity check** when doing Bayesian updates — trivial positions create false signals
7. **Never use numpy/scipy** in core modules — the project deliberately uses pure Python for portability
