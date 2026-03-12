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
from src.core.delta_model import DeltaModel, MoveAnalysis, GameAnalysis, SoftmaxChoiceModel
from src.core.player_model import BayesianPlayerModel, PlayerClassification
from src.core.temporal_model import TemporalModel, TemporalAnalysis
from src.core.feature_extractor import FeatureExtractor, FeatureVector
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
        config: AnalysisConfig
    ):
        super().__init__()
        self.engine = engine_service
        self.board = board
        self.config = config
        self.is_running = True
        
        # Initialize V2 models from config
        self.delta_model = DeltaModel(
            near_optimal_threshold=config.near_optimal_threshold
        )
        
        self.player_model = BayesianPlayerModel(
            prior_cheat=config.prior_cheat,
            lambda_human=config.lambda_human,
            lambda_cheater=config.lambda_cheater,
            threshold_suspicious=config.threshold_suspicious,
            threshold_cheater=config.threshold_cheater
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
                complexity_result = calculate_complexity(candidate_winrates, prev_winrate)
                position_complexity = complexity_result.complexity
                
                # Log complexity details
                self.engine._log("STAGE", f"  Complexity: {position_complexity:.2f} (Impact={complexity_result.impact_factor:.2f}, Var={complexity_result.variance_factor:.2f})")
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
            
            # Update previous winrate for next move's complexity calculation
            # Keep SAME perspective: prev_wr = best achievable WR from this analyzed move
            if should_analyze and best_eval:
                prev_winrate = best_wr
                prev_opp_best = 1.0 - played_wr  # Opponent's WR before their response
            
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
                        
                        # Update the result with opponent analysis
                        result.opponent_analysis = opp_analysis
                        
                        # V3: Retroactive hack REMOVED — tempered likelihood handles
                        # complexity/forcing naturally. No more ad-hoc posterior manipulation.
                        
                        # === STRATEGIC ACCURACY BONUS ===
                        # Boost accuracy score if move created complexity for opponent
                        new_accuracy = calculate_accuracy(
                            delta, 
                            position_complexity, 
                            opponent_complexity=opp_analysis.opp_variance
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
                        self.engine._log("STAGE", f"  (No opponent candidates)")
                
                self.engine._log("STAGE", f"")
                
                # Emit per-move result (after opponent analysis added)
                self.move_result.emit(result)
        
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
        
        # Temporal analysis
        deltas = [m.delta for m in move_analyses]
        temporal = self.temporal_model.analyze(deltas)
        
        # Feature extraction
        features = self.feature_extractor.extract(game_analysis)
        features.p_cheat = classification.p_cheat
        features.classification = classification.classification
        
        # Build final result
        game_result = V2GameResult(
            moves=move_results,
            classification=classification,
            features=features,
            temporal=temporal,
            total_moves=len(self.board.moves),
            analyzed_moves=len(move_analyses),
            pattern_stats=self.pattern_analyzer.get_summary(),
            temperature_mle=temperature_mle
        )
        
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
