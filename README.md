# GomokuProb

**GomokuProb** is a Gomoku anti-cheating analysis desktop application. This tool determines whether a player used engine assistance by applying **Bayesian statistical inference** to move quality data.  It analyzes matches, determines move complexities, evaluates winrate drops (deltas), and estimates the probability of cheating.

## Features

- **Bayesian Statistical Inference:** Advanced mathematical modeling of player moves vs. engine moves.
- **Rapfi Engine Integration:** Uses the state-of-the-art C++ Rapfi AI engine to evaluate position winrates and find candidate moves.
- **Robust Mathematical Model:**
  - Delta Model (calculating winrate loss)
  - Bayesian Updates (online learning of player behavior)
  - Complexity Evaluation (impact, variance, criticality, phase)
  - Information Theory metrics (Shannon entropy, KL divergence)
  - Temporal Model (moving average window, switch-point detection)
- **GUI Application:** Built with PyQt6, featuring an MVVM architecture for clean presentation.

---

## Folder Structure

The project follows a clean Model-View-ViewModel (MVVM) architecture with distinct layer isolation.

```text
GomokuProb/
├── .agents/             # Agent rules and configuration for AI assistants
├── docs/                # Project documentation (API, protocol, etc.)
├── engine/              # Pre-compiled Rapfi engine binaries and neural network weights
├── lib/                 # Shared libraries (e.g., PyGomo protocol library)
├── src/                 # Main Python source code
│   ├── core/            # Core business logic and mathematical models (Delta, Bayesian, Complexity)
│   ├── services/        # Infrastructure and engine communication (e.g., engine_service.py)
│   ├── ui/              # User Interface (PyQt6 views and MainViewModel)
│   ├── utils/           # Helper scripts and utilities
│   └── main.py          # Application entry point
├── tests/               # Pytest suite for mathematical models and models validation
├── tools/               # Useful scripts like DB migration and DB viewing
├── config.json          # Main configuration file
└── README.md            # This file
```

---

## Requirements

- **Python:** 3.11 or higher
- **UI Framework:** `PyQt6`
- **Other libraries:** `pytest` (for testing)

*(Note: Mathematical computations deliberately use pure Python for portability, avoiding numpy/scipy in core modules).*

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd GomokuProb
   ```

2. **Set up a virtual environment (Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   Since this project uses PyQt6, you can install it via pip:
   ```bash
   pip install PyQt6 pytest
   ```
   *(If there's an additional requirement file, install via `pip install -r requirements.txt`)*

---

## Usage

### Running the Application

To launch the GomokuProb desktop GUI, run the `main.py` file from the project root:

```bash
python src/main.py
```

### Running Tests

The project includes an extensive test suite focused on mathematical logic and parsers. To run the tests, use `pytest`:

```bash
python -m pytest tests/
```

### Configuration
You can find settings like the window layout, paths, and engine variables inside `config.json` at the root of the project.

---

## Architecture Rules & Notes

- **Layer Isolation:** `src/core/` does not import from `src/ui/` or `src/services/`.
- **Threading Model:** Analysis occurs in a background `QThread` (`v2_worker.py`). Engine I/O is blocking within this thread, keeping the UI responsive.
- **Engine Protocol:** Uses the Gomocup protocol to communicate via stdin/stdout with the Rapfi engine.
