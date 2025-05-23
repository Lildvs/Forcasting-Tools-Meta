"""
Ensemble Methods Module

This module provides methods for creating and managing ensemble predictions
with a focus on personality-aware weighting strategies.
"""

import logging
import math
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable, Tuple

from forecasting_tools.data_models.forecast_report import (
    NumericDistribution,
    PredictedOptionList,
    ReasonedPrediction,
    BinaryReport
)
from forecasting_tools.personality_management.config import PersonalityConfig


logger = logging.getLogger(__name__)


class EnsembleMethod:
    """Base class for ensemble methods."""
    
    def __init__(self, name: str):
        """
        Initialize the ensemble method.
        
        Args:
            name: Name of the method
        """
        self.name = name
    
    def compute_weights(
        self, 
        forecaster_ids: List[str],
        performance_history: Optional[Dict[str, List[float]]] = None,
        personality_configs: Optional[Dict[str, PersonalityConfig]] = None,
    ) -> Dict[str, float]:
        """
        Compute weights for ensemble members.
        
        Args:
            forecaster_ids: List of forecaster identifiers
            performance_history: Optional performance history for each forecaster
            personality_configs: Optional personality configurations for each forecaster
            
        Returns:
            Dictionary mapping forecaster IDs to weights
        """
        # Base implementation uses equal weights
        return {fid: 1.0 for fid in forecaster_ids}


class EqualWeightEnsemble(EnsembleMethod):
    """Equal weighting for all ensemble members."""
    
    def __init__(self):
        """Initialize the equal weight ensemble method."""
        super().__init__("equal_weight")
    
    def compute_weights(
        self, 
        forecaster_ids: List[str],
        performance_history: Optional[Dict[str, List[float]]] = None,
        personality_configs: Optional[Dict[str, PersonalityConfig]] = None,
    ) -> Dict[str, float]:
        """
        Compute equal weights for all forecasters.
        
        Args:
            forecaster_ids: List of forecaster identifiers
            performance_history: Optional performance history for each forecaster
            personality_configs: Optional personality configurations for each forecaster
            
        Returns:
            Dictionary mapping forecaster IDs to equal weights
        """
        return {fid: 1.0 for fid in forecaster_ids}


class PerformanceWeightEnsemble(EnsembleMethod):
    """Weight ensemble members based on past performance."""
    
    def __init__(
        self, 
        learning_rate: float = 0.1, 
        recency_decay: float = 0.9,
        min_weight: float = 0.2
    ):
        """
        Initialize the performance-weighted ensemble method.
        
        Args:
            learning_rate: Rate at which weights are adjusted based on performance
            recency_decay: Decay factor for older performance records
            min_weight: Minimum weight for any forecaster
        """
        super().__init__("performance_weight")
        self.learning_rate = learning_rate
        self.recency_decay = recency_decay
        self.min_weight = min_weight
    
    def compute_weights(
        self, 
        forecaster_ids: List[str],
        performance_history: Optional[Dict[str, List[float]]] = None,
        personality_configs: Optional[Dict[str, PersonalityConfig]] = None,
    ) -> Dict[str, float]:
        """
        Compute weights based on past performance.
        
        Args:
            forecaster_ids: List of forecaster identifiers
            performance_history: Performance history for each forecaster
            personality_configs: Optional personality configurations for each forecaster
            
        Returns:
            Dictionary mapping forecaster IDs to weights
        """
        # If no performance history, use equal weights
        if not performance_history:
            return {fid: 1.0 for fid in forecaster_ids}
        
        weights = {}
        
        for fid in forecaster_ids:
            if fid in performance_history and performance_history[fid]:
                # Apply recency decay to past performance
                history = performance_history[fid]
                decay_factors = [self.recency_decay ** i for i in range(len(history) - 1, -1, -1)]
                
                # Calculate weighted average performance
                weighted_performance = sum(p * d for p, d in zip(history, decay_factors)) / sum(decay_factors)
                
                # Convert performance to weight
                # Use a logistic function to map performance to weights
                weight = 1.0 / (1.0 + math.exp(-self.learning_rate * weighted_performance))
                
                # Ensure minimum weight
                weights[fid] = max(weight, self.min_weight)
            else:
                # No history, use default weight
                weights[fid] = 1.0
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {fid: w / total_weight for fid, w in weights.items()}
        
        return weights


class PersonalityDiversityEnsemble(EnsembleMethod):
    """Weight ensemble members to maximize personality diversity."""
    
    def __init__(
        self, 
        diversity_weight: float = 0.5,
        performance_weight: float = 0.5,
    ):
        """
        Initialize the personality diversity ensemble method.
        
        Args:
            diversity_weight: Weight given to personality diversity
            performance_weight: Weight given to past performance
        """
        super().__init__("personality_diversity")
        self.diversity_weight = diversity_weight
        self.performance_weight = performance_weight
    
    def compute_weights(
        self, 
        forecaster_ids: List[str],
        performance_history: Optional[Dict[str, List[float]]] = None,
        personality_configs: Optional[Dict[str, PersonalityConfig]] = None,
    ) -> Dict[str, float]:
        """
        Compute weights based on personality diversity and performance.
        
        Args:
            forecaster_ids: List of forecaster identifiers
            performance_history: Optional performance history for each forecaster
            personality_configs: Personality configurations for each forecaster
            
        Returns:
            Dictionary mapping forecaster IDs to weights
        """
        # If no personality configs, use equal weights
        if not personality_configs:
            return {fid: 1.0 for fid in forecaster_ids}
        
        # Compute diversity scores
        diversity_scores = self._compute_diversity_scores(forecaster_ids, personality_configs)
        
        # Compute performance scores
        if performance_history:
            performance_scores = self._compute_performance_scores(forecaster_ids, performance_history)
        else:
            # If no performance history, use equal performance scores
            performance_scores = {fid: 1.0 for fid in forecaster_ids}
        
        # Combine diversity and performance scores
        weights = {}
        for fid in forecaster_ids:
            diversity_score = diversity_scores.get(fid, 1.0)
            performance_score = performance_scores.get(fid, 1.0)
            
            # Weighted combination
            weights[fid] = (
                self.diversity_weight * diversity_score +
                self.performance_weight * performance_score
            )
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {fid: w / total_weight for fid, w in weights.items()}
        
        return weights
    
    def _compute_diversity_scores(
        self, 
        forecaster_ids: List[str],
        personality_configs: Dict[str, PersonalityConfig]
    ) -> Dict[str, float]:
        """
        Compute diversity scores for each forecaster.
        
        Args:
            forecaster_ids: List of forecaster identifiers
            personality_configs: Personality configurations for each forecaster
            
        Returns:
            Dictionary mapping forecaster IDs to diversity scores
        """
        diversity_scores = {}
        
        for fid in forecaster_ids:
            if fid not in personality_configs:
                diversity_scores[fid] = 1.0
                continue
                
            # Calculate how different this personality is from others
            total_distance = 0.0
            personality = personality_configs[fid]
            
            for other_fid in forecaster_ids:
                if other_fid != fid and other_fid in personality_configs:
                    other_personality = personality_configs[other_fid]
                    distance = self._calculate_personality_distance(personality, other_personality)
                    total_distance += distance
            
            # Average distance from others (higher is more diverse)
            if len(forecaster_ids) > 1:
                avg_distance = total_distance / (len(forecaster_ids) - 1)
                diversity_scores[fid] = avg_distance
            else:
                diversity_scores[fid] = 1.0
        
        # Normalize diversity scores
        max_score = max(diversity_scores.values()) if diversity_scores else 1.0
        if max_score > 0:
            diversity_scores = {fid: s / max_score for fid, s in diversity_scores.items()}
        
        return diversity_scores
    
    def _calculate_personality_distance(
        self, 
        personality1: PersonalityConfig,
        personality2: PersonalityConfig
    ) -> float:
        """
        Calculate the distance between two personalities.
        
        Args:
            personality1: First personality configuration
            personality2: Second personality configuration
            
        Returns:
            Distance measure between 0.0 and 1.0
        """
        # Initialize distance
        distance = 0.0
        feature_count = 0
        
        # Compare thinking style
        if personality1.thinking_style != personality2.thinking_style:
            distance += 1.0
            feature_count += 1
        
        # Compare uncertainty approach
        if personality1.uncertainty_approach != personality2.uncertainty_approach:
            distance += 1.0
            feature_count += 1
        
        # Compare reasoning depth
        if personality1.reasoning_depth != personality2.reasoning_depth:
            distance += 1.0
            feature_count += 1
            
        # Compare temperature (normalize to 0-1 range)
        if hasattr(personality1, "temperature") and hasattr(personality2, "temperature"):
            temp_diff = abs(personality1.temperature - personality2.temperature) / 1.0  # Assuming max diff is 1.0
            distance += temp_diff
            feature_count += 1
        
        # Compare traits
        common_traits = set(personality1.traits.keys()).intersection(set(personality2.traits.keys()))
        for trait_name in common_traits:
            trait1 = personality1.traits[trait_name]
            trait2 = personality2.traits[trait_name]
            
            if isinstance(trait1.value, (int, float)) and isinstance(trait2.value, (int, float)):
                # Numeric trait
                trait_diff = abs(trait1.value - trait2.value)
                distance += trait_diff
                feature_count += 1
            elif trait1.value != trait2.value:
                # Categorical trait
                distance += 1.0
                feature_count += 1
        
        # Normalize distance
        if feature_count > 0:
            return distance / feature_count
        else:
            return 0.0
    
    def _compute_performance_scores(
        self, 
        forecaster_ids: List[str],
        performance_history: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Compute performance scores for each forecaster.
        
        Args:
            forecaster_ids: List of forecaster identifiers
            performance_history: Performance history for each forecaster
            
        Returns:
            Dictionary mapping forecaster IDs to performance scores
        """
        performance_scores = {}
        
        for fid in forecaster_ids:
            if fid in performance_history and performance_history[fid]:
                # Use average performance
                performance_scores[fid] = sum(performance_history[fid]) / len(performance_history[fid])
            else:
                # No history, use default score
                performance_scores[fid] = 1.0
        
        # Normalize performance scores
        max_score = max(performance_scores.values()) if performance_scores else 1.0
        if max_score > 0:
            performance_scores = {fid: s / max_score for fid, s in performance_scores.items()}
        
        return performance_scores


class DomainSpecificEnsemble(EnsembleMethod):
    """Weight ensemble members based on the domain of the question."""
    
    def __init__(
        self,
        domain_weights: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """
        Initialize the domain-specific ensemble method.
        
        Args:
            domain_weights: Optional dictionary mapping domains to forecaster weights
        """
        super().__init__("domain_specific")
        self.domain_weights = domain_weights or {}
        
        # Default domain-personality mapping if none provided
        if not domain_weights:
            self.domain_weights = {
                "economics": {"economist_": 2.0, "bayesian_": 1.5},
                "finance": {"economist_": 2.0, "bayesian_": 1.5},
                "politics": {"bayesian_": 1.5, "creative_": 1.2},
                "technology": {"creative_": 1.5, "balanced_": 1.2},
                "science": {"bayesian_": 1.5, "balanced_": 1.2},
                "health": {"cautious_": 1.5, "balanced_": 1.2},
                "sports": {"bayesian_": 1.5, "balanced_": 1.2},
                "entertainment": {"creative_": 1.5, "balanced_": 1.2},
                "geopolitics": {"bayesian_": 1.5, "cautious_": 1.2},
                "environment": {"cautious_": 1.5, "balanced_": 1.2},
                "energy": {"economist_": 1.5, "balanced_": 1.2},
                "social": {"creative_": 1.5, "balanced_": 1.2},
            }
    
    def compute_weights(
        self, 
        forecaster_ids: List[str],
        performance_history: Optional[Dict[str, List[float]]] = None,
        personality_configs: Optional[Dict[str, PersonalityConfig]] = None,
        domain: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Compute weights based on the question domain.
        
        Args:
            forecaster_ids: List of forecaster identifiers
            performance_history: Optional performance history for each forecaster
            personality_configs: Optional personality configurations for each forecaster
            domain: Optional domain for the question
            
        Returns:
            Dictionary mapping forecaster IDs to weights
        """
        weights = {fid: 1.0 for fid in forecaster_ids}
        
        # If no domain specified, use equal weights
        if not domain or domain not in self.domain_weights:
            return weights
        
        # Get domain-specific weights
        domain_weights = self.domain_weights[domain]
        
        # Apply domain-specific weights to matching forecasters
        for fid in forecaster_ids:
            for pattern, weight in domain_weights.items():
                if pattern in fid:
                    weights[fid] *= weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {fid: w / total_weight for fid, w in weights.items()}
        
        return weights


class AdaptiveEnsemble(EnsembleMethod):
    """
    Adaptively adjust ensemble weights based on performance feedback.
    
    This method uses the "Hedge" algorithm, which multiplicatively
    updates weights based on performance.
    """
    
    def __init__(
        self, 
        learning_rate: float = 0.1,
        min_weight: float = 0.1
    ):
        """
        Initialize the adaptive ensemble method.
        
        Args:
            learning_rate: Rate at which weights are adjusted based on performance
            min_weight: Minimum weight for any forecaster
        """
        super().__init__("adaptive")
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        self.current_weights: Dict[str, float] = {}
    
    def compute_weights(
        self, 
        forecaster_ids: List[str],
        performance_history: Optional[Dict[str, List[float]]] = None,
        personality_configs: Optional[Dict[str, PersonalityConfig]] = None,
    ) -> Dict[str, float]:
        """
        Compute weights adaptively based on past performance.
        
        Args:
            forecaster_ids: List of forecaster identifiers
            performance_history: Performance history for each forecaster
            personality_configs: Optional personality configurations for each forecaster
            
        Returns:
            Dictionary mapping forecaster IDs to weights
        """
        # Initialize weights if not already set
        if not self.current_weights:
            self.current_weights = {fid: 1.0 for fid in forecaster_ids}
        
        # Add any new forecasters
        for fid in forecaster_ids:
            if fid not in self.current_weights:
                self.current_weights[fid] = 1.0
        
        # If no performance history, use current weights
        if not performance_history:
            return self.current_weights.copy()
        
        # Update weights based on most recent performance
        for fid in forecaster_ids:
            if fid in performance_history and performance_history[fid]:
                # Get most recent performance
                latest_performance = performance_history[fid][-1]
                
                # Update weight using multiplicative update rule
                # Higher performance = higher weight
                self.current_weights[fid] *= math.exp(self.learning_rate * latest_performance)
                
                # Ensure minimum weight
                self.current_weights[fid] = max(self.current_weights[fid], self.min_weight)
        
        # Normalize weights
        total_weight = sum(self.current_weights.values())
        if total_weight > 0:
            self.current_weights = {fid: w / total_weight for fid, w in self.current_weights.items()}
        
        return self.current_weights.copy()


def weighted_aggregate_binary(
    predictions: List[float],
    weights: List[float]
) -> float:
    """
    Compute weighted average of binary predictions.
    
    Args:
        predictions: List of probability predictions
        weights: List of weights
        
    Returns:
        Weighted average probability
    """
    if not predictions:
        raise ValueError("No predictions to aggregate")
    
    if len(predictions) != len(weights):
        raise ValueError("Number of predictions must match number of weights")
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight <= 0:
        # If all weights are zero, use equal weights
        normalized_weights = [1.0 / len(weights)] * len(weights)
    else:
        normalized_weights = [w / total_weight for w in weights]
    
    # Compute weighted average
    weighted_avg = sum(p * w for p, w in zip(predictions, normalized_weights))
    
    return weighted_avg


def weighted_aggregate_numeric(
    means: List[float],
    stdevs: List[float],
    weights: List[float]
) -> Tuple[float, float]:
    """
    Compute weighted aggregation of numeric predictions.
    
    Args:
        means: List of means
        stdevs: List of standard deviations
        weights: List of weights
        
    Returns:
        Tuple of (weighted_mean, combined_stdev)
    """
    if not means or not stdevs:
        raise ValueError("No predictions to aggregate")
    
    if len(means) != len(stdevs) or len(means) != len(weights):
        raise ValueError("Number of means, stdevs, and weights must match")
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight <= 0:
        # If all weights are zero, use equal weights
        normalized_weights = [1.0 / len(weights)] * len(weights)
    else:
        normalized_weights = [w / total_weight for w in weights]
    
    # Compute weighted mean
    weighted_mean = sum(m * w for m, w in zip(means, normalized_weights))
    
    # Compute combined standard deviation
    # This accounts for both the variance within each prediction
    # and the variance between predictions
    within_variance = sum((s**2) * w for s, w in zip(stdevs, normalized_weights))
    between_variance = sum(w * (m - weighted_mean)**2 for m, w in zip(means, normalized_weights))
    combined_variance = within_variance + between_variance
    combined_stdev = math.sqrt(combined_variance)
    
    return weighted_mean, combined_stdev


def weighted_aggregate_multiple_choice(
    option_probabilities: List[List[float]],
    weights: List[float]
) -> List[float]:
    """
    Compute weighted aggregation of multiple choice predictions.
    
    Args:
        option_probabilities: List of probability lists
        weights: List of weights
        
    Returns:
        List of aggregated probabilities
    """
    if not option_probabilities:
        raise ValueError("No predictions to aggregate")
    
    if len(option_probabilities) != len(weights):
        raise ValueError("Number of predictions must match number of weights")
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight <= 0:
        # If all weights are zero, use equal weights
        normalized_weights = [1.0 / len(weights)] * len(weights)
    else:
        normalized_weights = [w / total_weight for w in weights]
    
    # Check that all probability lists have the same length
    num_options = len(option_probabilities[0])
    if not all(len(probs) == num_options for probs in option_probabilities):
        raise ValueError("All option probability lists must have the same length")
    
    # Compute weighted probabilities
    weighted_probabilities = [0.0] * num_options
    
    for probs, weight in zip(option_probabilities, normalized_weights):
        for i, prob in enumerate(probs):
            weighted_probabilities[i] += prob * weight
    
    return weighted_probabilities 