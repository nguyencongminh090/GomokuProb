"""
Configuration Model.
"""
import json
import dataclasses
import os
from dataclasses import dataclass

@dataclass
class AnalysisConfig:
    # Engine Settings
    engine_path: str = "engine/rapfi"
    rule: int = 0 # 0=Freestyle, 1=Standard, 4=Renju
    time_limit_ms: int = 1000 # 1 second per move default for fast analysis
    node_limit: int = 0 # 0 = No limit
    start_move: int = 1 # Analyze from move 1
    analyze_side: str = "both"  # "both", "black", or "white"
    
    # V2: Bayesian Model Parameters
    prior_cheat: float = 0.01           # Prior P(Cheat) - 1% base rate
    lambda_human: float = 5.0           # λ for human players (avg 20% winrate loss)
    lambda_cheater: float = 50.0        # λ for cheaters (avg 2% winrate loss)
    
    # V2: Classification Thresholds
    threshold_suspicious: float = 0.3   # P(Cheat) > this → "Suspicious"
    threshold_cheater: float = 0.7      # P(Cheat) > this → "Cheater"
    near_optimal_threshold: float = 0.02  # Δ < this counts as "near-optimal"
    
    # V2: Temporal Analysis
    temporal_window_size: int = 5       # Moving average window
    switch_detection_threshold: float = 0.1  # Min Δ change to detect switch
    
    # V4: Mixture Model Parameters (Paper Section 3.3)
    mixture_pi: float = 0.75            # Weight of "good play" component
    mixture_lambda_good: float = 20.0   # Rate for good plays
    mixture_lambda_blunder: float = 3.0 # Rate for blunders
    
    # V4: Analysis Options
    enable_model_selection: bool = True  # Run Gamma/Weibull comparison (AIC/BIC)
    enable_em_fitting: bool = True       # Fit mixture parameters via EM
    
    # LLM Report Generation Settings
    llm_provider: str = "gemini"         # e.g., "gemini", "groq", or "openai"
    llm_model: str = "gemini-2.5-flash"
    llm_api_key: str = "AIzaSyC4AukNo7Xm5lYEGnK5dA3g-qrF-XCMX1c"
    llm_endpoint_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    
    @property
    def rule_name(self) -> str:
        rules = {0: "Freestyle", 1: "Standard", 4: "Renju"}
        return rules.get(self.rule, "Unknown")

    def save(self, file_path: str = "config.json") -> None:
        """Save configuration to a JSON file."""
        data = dataclasses.asdict(self)
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Failed to save config: {e}")

    @classmethod
    def load(cls, file_path: str = "config.json") -> "AnalysisConfig":
        """Load configuration from a JSON file."""
        if not os.path.exists(file_path):
            return cls() # Return default config
        
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            # Filter out keys that might not exist in current version
            valid_keys = cls.__dataclass_fields__.keys()
            filtered_data = {k: v for k, v in data.items() if k in valid_keys}
            return cls(**filtered_data)
        except Exception as e:
            print(f"Failed to load config: {e}")
            return cls()
