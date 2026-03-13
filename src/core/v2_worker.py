"""
GomoProb V2: Analysis Worker

Replacement for the V1 AnalysisWorker that uses the new Bayesian framework.
"""

from PyQt6.QtCore import QObject, pyqtSignal
from typing import List, Optional
from dataclasses import dataclass, field

from src.core.board import BoardState, Move
from src.core.config import AnalysisConfig
from src.services.engine_service import EngineService, MoveEvaluation

# V2 Imports
from src.core.delta_model import (
    DeltaModel, MoveAnalysis, GameAnalysis, SoftmaxChoiceModel,
    ModelSelector, FitResult,
)
from src.core.player_model import BayesianPlayerModel, PlayerClassification
from src.core.temporal_model import TemporalModel, TemporalAnalysis
from src.core.feature_extractor import FeatureExtractor, FeatureVector
from src.core.pattern_analysis import PatternAnalyzer
from src.core.profile_store import ProfileStore, ProfileRecord
from src.core.profile_analyzer import ProfileAnalyzer, ProfileResult
from src.core.information_theory import InformationTheory
from src.core.complexity import (
    calculate_complexity, calculate_accuracy, ComplexityResult,
    OpponentAnalysis, calculate_opponent_metrics
)


@dataclass
class V2MoveResult:
    """Result for a single move under V2 framework."""
    move_index: int
    move_notation: str
    played_winrate: float
    best_winrate: float
    delta: float                    # Δ = best - played
    
    # Running Bayesian update
    p_cheat_cumulative: float       # P(Cheat | all moves so far)
    
    # V1 compatibility (for UI)
    accuracy_score: float = 0.0     # Converted from delta for UI
    regret: float = 0.0
    
    # Position info
    is_forced: bool = False
    position_complexity: float = 1.0
    
    # Opponent analysis (after human move)
    opponent_analysis: OpponentAnalysis = None
    
    # V3: Softmax choice probability
    softmax_choice_prob: float = 0.0  # P(this choice | candidates, τ)
    
    # Static Pattern Context (V3)
    pattern_context: dict = None


@dataclass
class V2GameResult:
    """Complete game analysis result under V2 framework."""
    moves: List[V2MoveResult] = field(default_factory=list)
    
    # Final classification
    classification: PlayerClassification = None
    
    # Feature vector
    features: FeatureVector = None
    
    # Temporal analysis
    temporal: TemporalAnalysis = None
    
    # Game metadata
    total_moves: int = 0
    analyzed_moves: int = 0
    
    # V3 Statistics
    pattern_stats: str = ""
    temperature_mle: float = 0.1  # Softmax temperature estimate
    
    # V4 Statistics (Paper alignment)
    temperature_score: float = 0.0     # S_τ = 1/τ̂
    log_likelihood_ratio: float = 0.0  # log Λ = L₁ - L₀
    
    # V4: Distribution model comparison (Paper Section 3.2.4)
    distribution_fits: List = field(default_factory=list)  # List[FitResult]
    best_distribution: str = "exponential"  # Best model by BIC
    
    # V4: EM-fitted mixture parameters (Paper Section 3.3)
    em_pi: float = 0.0
    em_lambda_good: float = 0.0
    em_lambda_blunder: float = 0.0
    
    # V4: Profile Analysis results (Paper Section 10.2)
    profile_result: Optional[ProfileResult] = None



class V2AnalysisWorker(QObject):
    """
    V2 Analysis Worker using Bayesian framework.
    
    Key differences from V1:
    - Uses Winrate Delta model (Δ ~ Exponential)
    - Computes running P(Cheat | data) after each move
    - Performs temporal analysis for switch-point detection
    - Builds comprehensive feature vector
    """
    
    # Signals
    progress = pyqtSignal(int, int)          # current, total
    move_result = pyqtSignal(object)         # V2MoveResult
    game_complete = pyqtSignal(object)       # V2GameResult
    finished = pyqtSignal()
    
    def __init__(
        self,
        engine_service: EngineService,
        board: BoardState,
        config: AnalysisConfig,
        player_name: str = "",
        save_to_profile: bool = False
    ):
        super().__init__()
        self.engine = engine_service
        self.board = board
        self.config = config
        self.player_name = player_name.strip()
        self.save_to_profile = save_to_profile
        self.is_running = True
        
        # Profile DB
        self.profile_store = ProfileStore()
        
        # Initialize V2 models from config
        self.delta_model = DeltaModel(
            near_optimal_threshold=config.near_optimal_threshold
        )
        
        self.player_model = BayesianPlayerModel(
            prior_cheat=config.prior_cheat,
            lambda_human=config.lambda_human,
            lambda_cheater=config.lambda_cheater,
            threshold_suspicious=config.threshold_suspicious,
            threshold_cheater=config.threshold_cheater,
            # V4: Mixture model from config
            mixture_pi=config.mixture_pi,
            mixture_lambda_good=config.mixture_lambda_good,
            mixture_lambda_blunder=config.mixture_lambda_blunder,
        )
        
        self.temporal_model = TemporalModel(
            window_size=config.temporal_window_size,
            switch_threshold=config.switch_detection_threshold
        )
        
        self.feature_extractor = FeatureExtractor(
            near_optimal_threshold=config.near_optimal_threshold
        )
        
        # V3: Pattern Analyzer
        from src.core.pattern_analysis import PatternAnalyzer
        self.pattern_analyzer = PatternAnalyzer()
        
        # V3: Softmax Choice Model
        self.softmax_model = SoftmaxChoiceModel()
    
    def run(self):
        """Main analysis loop."""
        # Start engine
        self.engine.engine_path = self.config.engine_path
        self.engine.start(rule=self.config.rule)
        
        total_moves = len(self.board.moves)
        
        # Setup board state
        current_board = BoardState(self.board.size)
        start_idx = max(0, self.config.start_move - 1)
        
        # Pre-fill board
        for i in range(start_idx):
            move = self.board.moves[i]
            current_board.add_move(move.x, move.y, move.color, move.notation)
        
        # Collect data for analysis
        move_analyses: List[MoveAnalysis] = []
        move_results: List[V2MoveResult] = []
        p_cheat_running = self.config.prior_cheat  # Start with prior
        
        # V3: Collect softmax observations for temperature MLE
        softmax_observations = []  # List of (candidate_winrates, chosen_index)
        
        # Track previous winrate for context-aware complexity
        prev_winrate = 0.5  # Initial balanced assumption
        prev_opp_best = 0.5  # Opponent's best WR before human's current move
        
        time_limit_sec = self.config.time_limit_ms / 1000.0
        
        # Analysis loop
        for i in range(start_idx, total_moves):
            if not self.is_running:
                break
            
            # Check game over
            winner = current_board.check_win()
            if winner != 0:
                self.engine._log("STAGE", f"Game Over: Winner = {'Black' if winner == 1 else 'White'}")
                break
            
            self.progress.emit(i + 1, total_moves)
            
            move = self.board.moves[i]
            move_num = i + 1
            
            # Determine if we should analyze this move
            is_black = (i % 2 == 0)
            should_analyze = True
            if self.config.analyze_side == "black" and not is_black:
                should_analyze = False
            elif self.config.analyze_side == "white" and is_black:
                should_analyze = False
            
            if should_analyze:
                self.engine._log("STAGE", f"")
                self.engine._log("STAGE", f"{'='*50}")
                self.engine._log("STAGE", f"V2 ANALYZING MOVE #{move_num}: {move.notation}")
                self.engine._log("STAGE", f"{'='*50}")
                
                # === V3: PATTERN ANALYSIS (STATIC) ===
                # Analyze the static patterns BEFORE the move is made to see what the user reacted to/created
                static_data = self.engine.get_static_eval(current_board)
                pattern_ctx = self.pattern_analyzer.analyze_move(static_data, move, is_black)
                
                if pattern_ctx:
                    self.engine._log("STAGE", f"  Pattern Context: Own={pattern_ctx['own_desc']} | Opp={pattern_ctx['opp_desc']}")
                
                # Get candidates from engine
                candidates = self.engine.analyze(
                    current_board, 
                    top_n=5,
                    time_limit=time_limit_sec,
                    node_limit=self.config.node_limit
                )
                
                # Find best move
                best_eval = max(candidates, key=lambda x: x.winrate) if candidates else None
                best_wr = best_eval.winrate if best_eval else 0.5
                
                # Find played move evaluation
                played_eval = None
                move_wins = current_board.check_win_after_move(move.x, move.y, move.color) == move.color
                
                if move_wins:
                    played_eval = MoveEvaluation(move.notation, 1.0, is_best=True)
                else:
                    # Search in candidates
                    for c in candidates:
                        if c.move_notation.lower() == move.notation.lower():
                            played_eval = c
                            break
                    
                    if not played_eval:
                        # Evaluate separately
                        played_eval = self.engine.evaluate_move(
                            current_board, move, time_limit=time_limit_sec
                        )
                
                played_wr = played_eval.winrate if played_eval else 0.5
                
                # Calculate delta
                delta = self.delta_model.calculate_delta(best_wr, played_wr)
                
                # Determine if forced move
                reasonable_moves = [c for c in candidates if c.winrate >= 0.4]
                is_forced = len(reasonable_moves) <= 1
                
                # Context-Aware Complexity using all candidates
                candidate_winrates = [c.winrate for c in candidates]
                complexity_result = calculate_complexity(candidate_winrates, prev_winrate, w_opp_best=prev_opp_best)
                position_complexity = complexity_result.final_complexity
                
                # Log complexity details
                self.engine._log("STAGE", f"  Complexity (Raw): {complexity_result.complexity:.2f} (Impact={complexity_result.impact_factor:.2f}, Var={complexity_result.variance_factor:.2f})")
                self.engine._log("STAGE", f"  Opponent Adj: F_opp={complexity_result.opp_factor:.2f} | C_adj={complexity_result.adjusted_complexity:.2f} | C_final={complexity_result.final_complexity:.2f}")
                self.engine._log("STAGE", f"  Prev WR: {prev_winrate*100:.1f}% → Best WR: {best_wr*100:.1f}%")
                
                # Create MoveAnalysis
                move_analysis = MoveAnalysis(
                    move_index=i,
                    move_notation=move.notation,
                    best_winrate=best_wr,
                    played_winrate=played_wr,
                    delta=delta,
                    is_forced=is_forced,
                    position_complexity=position_complexity
                )
                move_analyses.append(move_analysis)
                
                # V3: Update running P(Cheat) (tempered Bayesian update)
                p_cheat_running = self.player_model.update_online(p_cheat_running, delta, position_complexity)
                
                # V3: Softmax choice probability
                candidate_wrs = [c.winrate for c in candidates]
                chosen_idx = -1
                for ci, c in enumerate(candidates):
                    if c.move_notation.lower() == move.notation.lower():
                        chosen_idx = ci
                        break
                
                softmax_prob = 0.0
                if chosen_idx >= 0 and candidate_wrs:
                    softmax_observations.append((candidate_wrs, chosen_idx))
                    # Use a reference temperature of 0.1 for per-move display
                    import math as _math
                    softmax_prob = _math.exp(
                        SoftmaxChoiceModel.log_prob_of_choice(candidate_wrs, chosen_idx, 0.1)
                    )
                
                # Calculate context-aware accuracy
                accuracy = calculate_accuracy(delta, position_complexity)
                
                result = V2MoveResult(
                    move_index=i,
                    move_notation=move.notation,
                    played_winrate=played_wr,
                    best_winrate=best_wr,
                    delta=delta,
                    p_cheat_cumulative=p_cheat_running,
                    accuracy_score=accuracy,
                    regret=delta,
                    is_forced=is_forced,
                    position_complexity=position_complexity,
                    opponent_analysis=None,  # Will be filled after move added
                    softmax_choice_prob=softmax_prob,
                    pattern_context=pattern_ctx  # V3 Pattern Data
                )
                move_results.append(result)
                
                # Log before opponent analysis
                self.engine._log("STAGE", f"  Best: {best_eval.move_notation if best_eval else '?'} | WR {best_wr*100:.1f}%")
                self.engine._log("STAGE", f"  Played: {move.notation} | WR {played_wr*100:.1f}%")
                self.engine._log("STAGE", f"  Δ = {delta*100:.1f}%")
                self.engine._log("STAGE", f"  P(Cheat) running = {p_cheat_running*100:.1f}%")
            
            # Update previous winrate for next move's complexity calculation (Fix Issue #4)
            # Use the actual played winrate as the new baseline
            if should_analyze and played_eval:
                prev_winrate = played_wr
            
            # Add move to board
            current_board.add_move(move.x, move.y, move.color, move.notation)
            
            # === OPPONENT FUTURE ANALYSIS ===
            # After human's move, analyze opponent's options (NBEST 5)
            if should_analyze:
                winner = current_board.check_win()
                if winner == 0 and self.is_running:
                    self.engine._log("STAGE", f"  --- Opponent Analysis ---")
                    
                    # Get opponent's candidates
                    opp_candidates = self.engine.analyze(
                        current_board,
                        top_n=5,
                        time_limit=time_limit_sec * 0.5,  # Half time for speed
                        node_limit=self.config.node_limit // 2 if self.config.node_limit else None
                    )
                    
                    if opp_candidates:
                        opp_winrates = [e.winrate for e in opp_candidates]
                        opp_analysis = calculate_opponent_metrics(opp_winrates, prev_opp_best)
                        
                        # V3 TASK-04 Update prev_opp_best for the next turn's F_opp calculation
                        prev_opp_best = opp_analysis.opp_best
                        
                        # Update the result with opponent analysis
                        result.opponent_analysis = opp_analysis
                        
                        # V3: Retroactive hack REMOVED — tempered likelihood handles
                        # complexity/forcing naturally. No more ad-hoc posterior manipulation.
                        
                        # === STRATEGIC ACCURACY BONUS (Fix Issue #3) ===
                        # Compute C_opp correctly from opponent's perspective
                        opp_complex_result = calculate_complexity(
                            opp_winrates,
                            prev_winrate=1.0 - prev_winrate # Convert human's prev_wr to opponent's perspective
                        )
                        c_opp = opp_complex_result.complexity # Raw complexity
                        
                        # Boost accuracy score if move created complexity for opponent
                        new_accuracy = calculate_accuracy(
                            delta, 
                            position_complexity, 
                            opponent_complexity=c_opp
                        )
                        if new_accuracy > result.accuracy_score:
                             self.engine._log("STAGE", f"  Strategic Bonus: Acc {result.accuracy_score*100:.1f}% → {new_accuracy*100:.1f}%")
                             result.accuracy_score = new_accuracy
                        
                        # Log opponent analysis - all 5 metrics
                        self.engine._log("STAGE", f"  Opponent NBEST: {[f'{wr*100:.1f}%' for wr in opp_winrates]}")
                        self.engine._log("STAGE", f"  Opp Var: {opp_analysis.opp_variance:.2f} | Opp Best: {opp_analysis.opp_best*100:.1f}%")
                        self.engine._log("STAGE", f"  Forcing: {opp_analysis.forcing_level*100:.0f}% | Pressure: {opp_analysis.pressure*100:.1f}% | Viable: {opp_analysis.viable_count}")
                        self.engine._log("STAGE", f"  Quality: {opp_analysis.move_quality}")
                    else:
                        # Fix Bug #2: Decay stale prev_opp_best if no candidates 
                        prev_opp_best = prev_opp_best * 0.7 + 0.5 * 0.3
                        self.engine._log("STAGE", f"  (No opponent candidates)")
                
                self.engine._log("STAGE", f"")
                
                # Emit per-move result (after opponent analysis added)
                self.move_result.emit(result)
            else:
                # Bug #1: We are skipping analysis for this move (opponent's turn).
                # However, we MUST compute the opponent's *actual* best option from the
                # position AFTER they moved, so that F_opp is accurate for the human's NEXT turn.
                winner = current_board.check_win()
                if winner == 0 and self.is_running:
                    # Quick engine call to evaluate state after opponent played
                    quick_candidates = self.engine.analyze(
                        current_board,
                        top_n=3,
                        time_limit=time_limit_sec * 0.2, # Very fast
                        node_limit=self.config.node_limit // 4 if self.config.node_limit else None
                    )
                    if quick_candidates:
                        human_best = max(c.winrate for c in quick_candidates)
                        prev_opp_best = 1.0 - human_best
                    else:
                        prev_opp_best = prev_opp_best * 0.7 + 0.5 * 0.3
        
        # ----------------------------
        # POST-GAME ANALYSIS
        # ----------------------------
        
        # Build GameAnalysis
        game_analysis = self.delta_model.analyze_game(move_analyses)
        
        # V3: Use online posterior as the authoritative result
        # (batch classify_game uses mixture model but without per-move tempering)
        classification = self.player_model.classify_game(game_analysis)
        
        # Online posterior IS the final result (consistent single pathway)
        classification.p_cheat = p_cheat_running
        classification.p_human = 1.0 - p_cheat_running
        
        if classification.p_cheat >= self.player_model.threshold_cheater:
            classification.classification = "Cheater"
        elif classification.p_cheat >= self.player_model.threshold_suspicious:
            classification.classification = "Suspicious"
        else:
            classification.classification = "Human"
        
        # V3: Estimate softmax temperature from all collected observations
        temperature_mle = 0.1  # Default
        if softmax_observations:
            temperature_mle = self.softmax_model.estimate_temperature(softmax_observations)
        
        # V4: Compute temperature score S_τ = 1/τ̂ (Paper Eq. tau_score)
        temp_score = SoftmaxChoiceModel.temperature_score(temperature_mle)
        
        # V4: Compute Likelihood Ratio Test (Paper Section 5.5)
        deltas_lrt = [m.delta for m in move_analyses]
        log_lambda, _, _ = self.player_model.compute_lrt(deltas_lrt)
        
        # V4: Compute prior sensitivity analysis (Paper Section 5.6)
        sensitivity = self.player_model.sensitivity_analysis(deltas_lrt)
        
        # Store V4 metrics in classification
        classification.log_likelihood_ratio = log_lambda
        classification.temperature_score = temp_score
        classification.sensitivity_results = sensitivity
        
        # V4: LRT Escalation Rule (Paper Section 5.5)
        if (classification.classification == "Human" and
            log_lambda > 10.0 and
            len(move_analyses) >= 10):
            classification.classification = "Suspicious (LRT-flagged)"
            classification.lrt_warning = True
        elif (classification.classification in ["Human", "Suspicious"] and
              log_lambda > 5.0 and
              len(move_analyses) >= 10):
            # Flag but do not escalate classification
            classification.lrt_warning = True
        
        # V4: Distribution model comparison (Paper Section 3.2.4)
        # Fit Exponential, Gamma, Weibull and select best by BIC
        dist_fits = []
        best_dist_name = "exponential"
        if self.config.enable_model_selection and len(deltas_lrt) >= 5:
            try:
                dist_fits = ModelSelector.fit_and_compare(deltas_lrt)
                if dist_fits:
                    best_dist_name = dist_fits[0].distribution
            except Exception:
                pass
        
        # V4: EM-fitted mixture parameters (Paper Section 3.3)
        # Fit mixture model from actual game data
        from src.core.mixture_model import MixtureModel as MixtureFitter
        em_pi, em_lg, em_lb = 0.0, 0.0, 0.0
        if self.config.enable_em_fitting and len(deltas_lrt) >= 10:
            try:
                fitter = MixtureFitter(
                    pi=self.player_model.mixture_model.pi,
                    lambda_good=self.player_model.mixture_model.lambda_good,
                    lambda_blunder=self.player_model.mixture_model.lambda_blunder,
                )
                em_pi, em_lg, em_lb = fitter.fit_em(deltas_lrt)
            except Exception:
                pass
        
        # Temporal analysis
        deltas = [m.delta for m in move_analyses]
        temporal = self.temporal_model.analyze(deltas)
        
        # Feature extraction
        features = self.feature_extractor.extract(game_analysis)
        features.p_cheat = classification.p_cheat
        features.classification = classification.classification
        
        # ----------------------------
        # V4 PROFILE ANALYSIS & ESCALATION
        # ----------------------------
        profile_res = None
        if self.player_name:
            history = self.profile_store.get_baseline_games(self.player_name)
            profile_res = ProfileAnalyzer.analyze_suspect_game(features.lambda_mle, history)
            
            # Escalation Rules (Paper Section 10.2)
            if profile_res.status != "INSUFFICIENT_DATA":
                z_stat = profile_res.test_statistic
                crit_05 = profile_res.critical_05
                crit_01 = profile_res.critical_01
                
                if classification.classification == "Suspicious" and z_stat > crit_05:
                    classification.classification = "Suspicious-Elevated"
                elif classification.classification == "Human":
                    if z_stat > crit_01:
                        classification.classification = "Manual Review"
                    elif z_stat > crit_05:
                        # Paper says confirmed Human requires Z <= z_0.05
                        classification.classification = "Inconsistent (Z-score)"
                        
        # ----------------------------
        
        # Build final result
        game_result = V2GameResult(
            moves=move_results,
            classification=classification,
            features=features,
            temporal=temporal,
            total_moves=len(self.board.moves),
            analyzed_moves=len(move_analyses),
            pattern_stats=self.pattern_analyzer.get_summary(),
            temperature_mle=temperature_mle,
            temperature_score=temp_score,
            log_likelihood_ratio=log_lambda,
            distribution_fits=dist_fits,
            best_distribution=best_dist_name,
            em_pi=em_pi,
            em_lambda_good=em_lg,
            em_lambda_blunder=em_lb,
            profile_result=profile_res
        )
        
        # V4 Profile Saving (Paper Section 8 strict baseline rules)
        if self.player_name and self.save_to_profile:
            # Strict logic: must be Human, confident >= 0.7, p_cheat < 0.20, and enough moves
            is_baseline_game = (
                classification.classification == "Human" and
                classification.confidence >= 0.70 and
                classification.p_cheat < 0.20 and
                len(move_analyses) >= 44
            )
            
            record = ProfileRecord(
                player_name=self.player_name,
                timestamp="", # Auto-generated
                lambda_mle=features.lambda_mle,
                mean_delta=features.mean_delta,
                num_moves=len(move_analyses),
                skill_tier="T_Unassigned", # Could map this via lambda_human
                lambda_human=self.player_model.mixture_model.lambda_good,
                classification=classification.classification,
                confidence=classification.confidence,
                is_baseline=is_baseline_game
            )
            self.profile_store.add_game(record)
            self.engine._log("INFO", f"Saved game to profile for '{self.player_name}' (Baseline: {is_baseline_game})")

        
        # Log final results
        self.engine._log("STAGE", f"")
        self.engine._log("STAGE", f"{'='*50}")
        self.engine._log("STAGE", f"V2 GAME ANALYSIS COMPLETE")
        self.engine._log("STAGE", f"{'='*50}")
        self.engine._log("STAGE", f"  Analyzed Moves: {game_result.analyzed_moves}")
        self.engine._log("STAGE", f"  Mean Δ: {features.mean_delta*100:.1f}%")
        self.engine._log("STAGE", f"  λ (MLE): {features.lambda_mle:.2f}")
        self.engine._log("STAGE", f"  Near-optimal %: {features.near_optimal_ratio*100:.1f}%")
        self.engine._log("STAGE", f"  P(Cheat): {classification.p_cheat*100:.1f}%")
        self.engine._log("STAGE", f"  CLASSIFICATION: {classification.classification}")
        self.engine._log("STAGE", f"  Confidence: {classification.confidence*100:.1f}%")
        self.engine._log("STAGE", f"  Softmax τ (MLE): {temperature_mle:.4f}")
        self.engine._log("STAGE", f"  Temperature Score S_τ: {temp_score:.2f}")
        self.engine._log("STAGE", f"  Log LR (Λ): {log_lambda:.2f}")
        
        # Log sensitivity analysis
        if sensitivity:
            sens_str = " | ".join(f"P(H1)={p:.3f}→{post*100:.1f}%" for p, post in sensitivity.items())
            self.engine._log("STAGE", f"  Sensitivity: {sens_str}")
        
        # V4: Log distribution model comparison
        if dist_fits:
            self.engine._log("STAGE", f"  --- Distribution Model Comparison ---")
            self.engine._log("STAGE", f"  Best Model (BIC): {best_dist_name.upper()}")
            for fit in dist_fits:
                ks_status = '✓' if fit.ks_pvalue > 0.05 else '✗'
                self.engine._log("STAGE",
                    f"    {fit.distribution:12s} | AIC={fit.aic:.1f} | BIC={fit.bic:.1f}"
                    f" | KS p={fit.ks_pvalue:.3f} {ks_status}"
                    f" | params={fit.params}"
                )
        
        # V4: Log EM-fitted mixture parameters
        if em_pi > 0:
            self.engine._log("STAGE", f"  --- EM-Fitted Mixture Parameters ---")
            self.engine._log("STAGE", f"    π={em_pi:.3f} (default={self.player_model.mixture_model.pi:.3f})")
            self.engine._log("STAGE", f"    λ_good={em_lg:.2f} (default={self.player_model.mixture_model.lambda_good:.2f})")
            self.engine._log("STAGE", f"    λ_blunder={em_lb:.2f} (default={self.player_model.mixture_model.lambda_blunder:.2f})")
            
        # V4: Log Profile Analysis
        if profile_res:
            self.engine._log("STAGE", f"  --- Profile History Analysis ({self.player_name}) ---")
            if profile_res.status == "INSUFFICIENT_DATA":
                self.engine._log("STAGE", f"    Status: Profile building (m={profile_res.m_games}). Need m>=2.")
            else:
                self.engine._log("STAGE", f"    Based on {profile_res.m_games} historical games")
                self.engine._log("STAGE", f"    Historical Baseline: λ_bar = {profile_res.lambda_bar:.2f} (s_λ = {profile_res.s_lambda:.2f})")
                self.engine._log("STAGE", f"    Game MLE λ: {features.lambda_mle:.2f}")
                test_type = "Z-test" if profile_res.m_games >= 30 else "T-test"
                self.engine._log("STAGE", f"    {test_type} Statistic: {profile_res.test_statistic:.2f} (p={profile_res.p_value:.4f})")
                self.engine._log("STAGE", f"    Critical Limits: 5%={profile_res.critical_05:.2f}, 1%={profile_res.critical_01:.2f}")
                self.engine._log("STAGE", f"    Result: {profile_res.status} - {profile_res.message}")
        
        if temporal.is_suspicious:
            self.engine._log("STAGE", f"  ⚠️ SUSPICIOUS: Switch-point detected!")
            for sp in temporal.switch_points:
                self.engine._log("STAGE", f"    Move {sp.move_index}: {sp.direction} by {sp.magnitude*100:.1f}%")
        
        self.engine._log("STAGE", f"")
        
        # Emit final results
        self.game_complete.emit(game_result)
        
        self.engine.stop_search()
        self.finished.emit()
    
    def stop(self):
        """Stop the analysis."""
        self.is_running = False
