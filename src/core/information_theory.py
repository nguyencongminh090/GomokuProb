"""
GomoProb V2: Information Theory Metrics

Implements information-theoretic measures for anti-cheating detection.
- Shannon Entropy: Measures "randomness" of player choices
- Cross-Entropy: Compares player distribution to engine distribution
- KL Divergence: D_KL(Player || Engine) - lower = more engine-like

Core insight:
- Humans have higher entropy (more varied choices)
- Cheaters have lower entropy (always near-optimal)
"""

import math
from typing import List, Optional


class InformationTheory:
    """
    Information-theoretic metrics for player behavior analysis.
    """
    
    @staticmethod
    def shannon_entropy(probabilities: List[float]) -> float:
        """
        Calculate Shannon entropy of a probability distribution.
        
        H(P) = -Σ p_i * log(p_i)
        
        Higher entropy = more uncertainty/randomness.
        A player who always picks the same type of move has low entropy.
        
        Args:
            probabilities: List of probabilities (should sum to 1)
            
        Returns:
            Entropy in bits (using log base 2)
        """
        if not probabilities:
            return 0.0
        
        epsilon = 1e-10  # Avoid log(0)
        
        entropy = 0.0
        for p in probabilities:
            if p > epsilon:
                entropy -= p * math.log2(p)
        
        return entropy
    
    @staticmethod
    def cross_entropy(p_true: List[float], p_model: List[float]) -> float:
        """
        Calculate cross-entropy between true distribution and model distribution.
        
        H(P, Q) = -Σ p_i * log(q_i)
        
        Measures how well Q predicts P.
        Lower = Q is a better model of P.
        
        Args:
            p_true: True probability distribution (player choices)
            p_model: Model distribution (engine policy)
            
        Returns:
            Cross-entropy value
        """
        if len(p_true) != len(p_model):
            raise ValueError("Distributions must have same length")
        
        if not p_true:
            return 0.0
        
        epsilon = 1e-10
        
        cross_ent = 0.0
        for p, q in zip(p_true, p_model):
            if p > epsilon:
                q_safe = max(q, epsilon)
                cross_ent -= p * math.log2(q_safe)
        
        return cross_ent
    
    @staticmethod
    def kl_divergence(p_true: List[float], p_model: List[float]) -> float:
        """
        Calculate KL Divergence D_KL(P || Q).
        
        D_KL(P || Q) = Σ p_i * log(p_i / q_i) = H(P, Q) - H(P)
        
        Measures how different P is from Q.
        D_KL = 0 means P == Q (player = engine).
        Higher D_KL = player deviates more from engine.
        
        For anti-cheating:
        - Low D_KL = suspicious (too engine-like)
        - High D_KL = human-like
        
        Args:
            p_true: Player choice distribution
            p_model: Engine policy distribution
            
        Returns:
            KL Divergence value (always >= 0)
        """
        if len(p_true) != len(p_model):
            raise ValueError("Distributions must have same length")
        
        if not p_true:
            return 0.0
        
        epsilon = 1e-10
        
        kl_div = 0.0
        for p, q in zip(p_true, p_model):
            if p > epsilon:
                q_safe = max(q, epsilon)
                kl_div += p * math.log2(p / q_safe)
        
        return max(0.0, kl_div)  # KL is always non-negative
    
    @staticmethod
    def entropy_from_deltas(deltas: List[float], bin_count: int = 10) -> float:
        """
        Calculate entropy of delta distribution using histogram binning.
        
        This gives a measure of how varied the player's mistakes are.
        - Low entropy: Consistent mistake pattern (or no mistakes)
        - High entropy: Random/varied mistakes
        
        Args:
            deltas: List of Δ values
            bin_count: Number of histogram bins
            
        Returns:
            Entropy of binned delta distribution
        """
        if not deltas:
            return 0.0
        
        # Create histogram bins from 0 to max(delta)
        max_delta = max(deltas) if deltas else 1.0
        bin_width = max(max_delta / bin_count, 0.01)
        
        # Count deltas in each bin
        bins = [0] * bin_count
        for d in deltas:
            bin_idx = min(int(d / bin_width), bin_count - 1)
            bins[bin_idx] += 1
        
        # Convert to probabilities
        total = sum(bins)
        if total == 0:
            return 0.0
        
        probs = [b / total for b in bins]
        
        return InformationTheory.shannon_entropy(probs)
    
    @staticmethod
    def move_choice_entropy(softmax_vectors: List[List[float]]) -> float:
        """
        Calculate average entropy of move choices across the game.
        
        Each softmax_vector represents P(move | position) for candidates.
        Higher entropy = position had many reasonable options.
        
        Args:
            softmax_vectors: List of probability vectors per move
            
        Returns:
            Average entropy across all positions
        """
        if not softmax_vectors:
            return 0.0
        
        entropies = []
        for vec in softmax_vectors:
            if vec:
                entropies.append(InformationTheory.shannon_entropy(vec))
        
        return sum(entropies) / len(entropies) if entropies else 0.0


def analyze_player_information(
    player_choices: List[int],        # Index of move player chose (0-indexed)
    engine_policies: List[List[float]]  # P(move) from engine per position
) -> dict:
    """
    Full information-theoretic analysis of player behavior.
    
    Args:
        player_choices: For each position, which candidate did player pick (0-indexed)
        engine_policies: For each position, engine's policy distribution
        
    Returns:
        Dictionary with entropy, cross-entropy, and KL divergence metrics
    """
    it = InformationTheory()
    
    if len(player_choices) != len(engine_policies):
        raise ValueError("Mismatched lengths")
    
    n = len(player_choices)
    if n == 0:
        return {"entropy": 0.0, "cross_entropy": 0.0, "kl_divergence": 0.0}
    
    # Build player "policy" from observed choices
    # This is just a 1-hot vector per position (what they actually picked)
    total_kl = 0.0
    total_cross_ent = 0.0
    
    for choice_idx, engine_policy in zip(player_choices, engine_policies):
        if not engine_policy:
            continue
        
        # Player's actual choice as distribution (1-hot)
        player_dist = [0.0] * len(engine_policy)
        if 0 <= choice_idx < len(player_dist):
            player_dist[choice_idx] = 1.0
        
        # KL divergence for this position
        kl = it.kl_divergence(player_dist, engine_policy)
        total_kl += kl
        
        # Cross-entropy
        cross_ent = it.cross_entropy(player_dist, engine_policy)
        total_cross_ent += cross_ent
    
    return {
        "avg_kl_divergence": total_kl / n,
        "avg_cross_entropy": total_cross_ent / n,
        "total_kl_divergence": total_kl
    }
