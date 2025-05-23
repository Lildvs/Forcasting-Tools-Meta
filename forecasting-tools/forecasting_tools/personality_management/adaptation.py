"""
Personality Adaptation Module

This module provides mechanisms for dynamically adapting personality traits
based on forecast performance.
"""

import logging
import math
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

from forecasting_tools.personality_management.config import (
    PersonalityConfig, 
    PersonalityTrait,
    ReasoningDepth,
    UncertaintyApproach,
    ThinkingStyle
)


logger = logging.getLogger(__name__)


class PersonalityAdapter:
    """
    Adapts personality traits based on forecast performance.
    
    This class provides mechanisms to:
    - Track performance of personality traits across different domains
    - Adjust traits based on feedback
    - Create hybrid personalities optimized for specific domains
    """
    
    def __init__(
        self, 
        base_personality: PersonalityConfig,
        learning_rate: float = 0.1,
        save_dir: Optional[str] = None
    ):
        """
        Initialize the personality adapter.
        
        Args:
            base_personality: Base personality configuration to adapt
            learning_rate: Rate at which traits are adjusted (0.0-1.0)
            save_dir: Optional directory to save adapted personalities
        """
        self.base_personality = base_personality
        self.learning_rate = learning_rate
        
        if save_dir:
            self.save_dir = Path(save_dir)
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = None
            
        # Performance history by domain
        self.performance_history: Dict[str, List[float]] = {}
        
        # Trait effectiveness by domain
        self.trait_effectiveness: Dict[str, Dict[str, float]] = {}
    
    def adapt_to_feedback(
        self, 
        domain: str, 
        performance: float,
        trait_updates: Optional[Dict[str, Any]] = None
    ) -> PersonalityConfig:
        """
        Adapt the personality based on feedback for a specific domain.
        
        Args:
            domain: The domain of the forecast
            performance: Performance metric (-1.0 to 1.0 scale)
            trait_updates: Optional specific trait updates to apply
            
        Returns:
            Adapted personality configuration
        """
        # Update performance history
        if domain not in self.performance_history:
            self.performance_history[domain] = []
        self.performance_history[domain].append(performance)
        
        # Create a copy of the base personality to adapt
        adapted_config = self._clone_personality(self.base_personality)
        
        # Apply specific trait updates if provided
        if trait_updates:
            self._apply_trait_updates(adapted_config, trait_updates)
        
        # Apply adaptive adjustments based on performance
        self._apply_adaptive_adjustments(adapted_config, domain, performance)
        
        # Update the adaptation name
        adapted_config.name = f"{self.base_personality.name}_adapted_{domain}"
        adapted_config.description = f"Adaptation of {self.base_personality.name} for {domain} domain"
        
        # Save the adapted personality if a save directory is specified
        if self.save_dir:
            self._save_adapted_personality(adapted_config)
        
        return adapted_config
    
    def create_hybrid_personality(
        self,
        personality_weights: Dict[str, float],
        personality_configs: Dict[str, PersonalityConfig],
        name: str = "hybrid_personality",
        description: Optional[str] = None
    ) -> PersonalityConfig:
        """
        Create a hybrid personality by combining multiple personalities.
        
        Args:
            personality_weights: Dictionary mapping personality names to weights
            personality_configs: Dictionary mapping personality names to configurations
            name: Name for the hybrid personality
            description: Optional description for the hybrid personality
            
        Returns:
            Hybrid personality configuration
        """
        # Normalize weights
        total_weight = sum(personality_weights.values())
        if total_weight <= 0:
            raise ValueError("Sum of personality weights must be positive")
            
        normalized_weights = {
            p_name: weight / total_weight
            for p_name, weight in personality_weights.items()
        }
        
        # Start with an empty personality
        hybrid = PersonalityConfig(
            name=name,
            description=description or f"Hybrid personality combining {', '.join(personality_weights.keys())}"
        )
        
        # Combine core traits
        self._combine_core_traits(hybrid, personality_configs, normalized_weights)
        
        # Combine custom traits
        self._combine_custom_traits(hybrid, personality_configs, normalized_weights)
        
        # Combine template variables
        self._combine_template_variables(hybrid, personality_configs, normalized_weights)
        
        # Save the hybrid personality if a save directory is specified
        if self.save_dir:
            self._save_adapted_personality(hybrid)
        
        return hybrid
    
    def _clone_personality(self, personality: PersonalityConfig) -> PersonalityConfig:
        """
        Create a deep copy of a personality configuration.
        
        Args:
            personality: The personality configuration to clone
            
        Returns:
            Cloned personality configuration
        """
        # Convert to dict and back to create a deep copy
        personality_dict = personality.to_dict()
        
        # Create a new personality from the dict
        cloned = PersonalityConfig.from_dict(personality_dict)
        
        return cloned
    
    def _apply_trait_updates(
        self, 
        personality: PersonalityConfig, 
        trait_updates: Dict[str, Any]
    ) -> None:
        """
        Apply specific trait updates to a personality.
        
        Args:
            personality: The personality configuration to update
            trait_updates: Dictionary of trait updates to apply
        """
        # Update core traits
        if "reasoning_depth" in trait_updates:
            personality.reasoning_depth = ReasoningDepth(trait_updates["reasoning_depth"])
            
        if "uncertainty_approach" in trait_updates:
            personality.uncertainty_approach = UncertaintyApproach(trait_updates["uncertainty_approach"])
            
        if "thinking_style" in trait_updates:
            personality.thinking_style = ThinkingStyle(trait_updates["thinking_style"])
            
        if "temperature" in trait_updates:
            personality.temperature = float(trait_updates["temperature"])
            
        if "expert_persona" in trait_updates:
            personality.expert_persona = trait_updates["expert_persona"]
        
        # Update custom traits
        for trait_name, trait_value in trait_updates.items():
            if trait_name in ["reasoning_depth", "uncertainty_approach", "thinking_style", "temperature", "expert_persona"]:
                continue
                
            if trait_name in personality.traits:
                # Update existing trait
                personality.traits[trait_name].value = trait_value
            else:
                # Create new trait
                personality.traits[trait_name] = PersonalityTrait(
                    name=trait_name,
                    description=f"Adapted trait: {trait_name}",
                    value=trait_value
                )
    
    def _apply_adaptive_adjustments(
        self, 
        personality: PersonalityConfig, 
        domain: str, 
        performance: float
    ) -> None:
        """
        Apply adaptive adjustments based on performance.
        
        Args:
            personality: The personality configuration to adjust
            domain: The domain of the forecast
            performance: Performance metric (-1.0 to 1.0 scale)
        """
        # Skip if performance is neutral
        if abs(performance) < 0.1:
            return
            
        # Initialize trait effectiveness for this domain if needed
        if domain not in self.trait_effectiveness:
            self.trait_effectiveness[domain] = {}
        
        # Apply domain-specific adjustments
        if domain in ["economics", "finance", "business"]:
            # Economic domains benefit from analytical thinking and moderate reasoning depth
            if performance > 0:
                # Reinforce traits that work well
                if personality.thinking_style == ThinkingStyle.ANALYTICAL:
                    self._update_trait_effectiveness(domain, "analytical_thinking", performance)
                if personality.reasoning_depth == ReasoningDepth.MODERATE:
                    self._update_trait_effectiveness(domain, "moderate_depth", performance)
            else:
                # Adjust traits that don't work well
                if personality.thinking_style != ThinkingStyle.ANALYTICAL:
                    personality.thinking_style = ThinkingStyle.ANALYTICAL
                if personality.reasoning_depth == ReasoningDepth.SHALLOW:
                    personality.reasoning_depth = ReasoningDepth.MODERATE
                    
        elif domain in ["politics", "geopolitics", "international_relations"]:
            # Political domains benefit from creative thinking and deeper reasoning
            if performance > 0:
                if personality.thinking_style == ThinkingStyle.CREATIVE:
                    self._update_trait_effectiveness(domain, "creative_thinking", performance)
                if personality.reasoning_depth in [ReasoningDepth.DEEP, ReasoningDepth.EXHAUSTIVE]:
                    self._update_trait_effectiveness(domain, "deep_reasoning", performance)
            else:
                if personality.thinking_style != ThinkingStyle.CREATIVE:
                    personality.thinking_style = ThinkingStyle.CREATIVE
                if personality.reasoning_depth in [ReasoningDepth.SHALLOW, ReasoningDepth.MODERATE]:
                    personality.reasoning_depth = ReasoningDepth.DEEP
                    
        elif domain in ["science", "technology", "health"]:
            # Scientific domains benefit from analytical thinking and cautious uncertainty
            if performance > 0:
                if personality.thinking_style == ThinkingStyle.ANALYTICAL:
                    self._update_trait_effectiveness(domain, "analytical_thinking", performance)
                if personality.uncertainty_approach == UncertaintyApproach.CAUTIOUS:
                    self._update_trait_effectiveness(domain, "cautious_uncertainty", performance)
            else:
                if personality.thinking_style != ThinkingStyle.ANALYTICAL:
                    personality.thinking_style = ThinkingStyle.ANALYTICAL
                if personality.uncertainty_approach != UncertaintyApproach.CAUTIOUS:
                    personality.uncertainty_approach = UncertaintyApproach.CAUTIOUS
        
        # Apply numeric trait adjustments
        for trait_name, trait in personality.traits.items():
            if isinstance(trait.value, (int, float)):
                # Get the optimal value for this trait in this domain
                optimal_value = self._get_optimal_trait_value(domain, trait_name)
                
                if optimal_value is not None:
                    # Adjust toward optimal value
                    current_value = trait.value
                    adjustment = self.learning_rate * (optimal_value - current_value) * abs(performance)
                    new_value = current_value + adjustment
                    
                    # Ensure value stays in valid range (0-1 for most traits)
                    if trait_name in ["risk_tolerance", "creativity", "skepticism", "data_reliance"]:
                        new_value = max(0.0, min(1.0, new_value))
                    
                    trait.value = new_value
    
    def _update_trait_effectiveness(
        self, 
        domain: str, 
        trait_key: str, 
        performance: float
    ) -> None:
        """
        Update the effectiveness score for a trait in a domain.
        
        Args:
            domain: The domain to update
            trait_key: The trait key to update
            performance: Performance score for this trait
        """
        current_score = self.trait_effectiveness[domain].get(trait_key, 0.0)
        # Exponential moving average
        updated_score = current_score * 0.8 + performance * 0.2
        self.trait_effectiveness[domain][trait_key] = updated_score
    
    def _get_optimal_trait_value(
        self, 
        domain: str, 
        trait_name: str
    ) -> Optional[float]:
        """
        Get the optimal value for a trait in a domain.
        
        Args:
            domain: The domain to check
            trait_name: The trait name to check
            
        Returns:
            Optimal value if available, None otherwise
        """
        # Use domain-specific knowledge for common traits
        if trait_name == "data_reliance":
            if domain in ["economics", "finance", "science", "health"]:
                return 0.8  # High data reliance
            elif domain in ["politics", "entertainment", "sports"]:
                return 0.6  # Moderate data reliance
        
        elif trait_name == "risk_tolerance":
            if domain in ["finance", "health", "security"]:
                return 0.3  # Low risk tolerance
            elif domain in ["technology", "entertainment", "sports"]:
                return 0.7  # Higher risk tolerance
        
        elif trait_name == "creativity":
            if domain in ["technology", "entertainment", "arts"]:
                return 0.8  # High creativity
            elif domain in ["finance", "law", "security"]:
                return 0.4  # Lower creativity
        
        # If no domain-specific knowledge, return None
        return None
    
    def _combine_core_traits(
        self,
        hybrid: PersonalityConfig,
        personality_configs: Dict[str, PersonalityConfig],
        weights: Dict[str, float]
    ) -> None:
        """
        Combine core traits from multiple personalities.
        
        Args:
            hybrid: The hybrid personality to update
            personality_configs: Dictionary of personality configurations
            weights: Dictionary of normalized weights
        """
        # Combine reasoning depth
        depth_scores = {
            ReasoningDepth.SHALLOW: 0,
            ReasoningDepth.MODERATE: 1,
            ReasoningDepth.DEEP: 2,
            ReasoningDepth.EXHAUSTIVE: 3
        }
        
        weighted_depth = 0.0
        for p_name, weight in weights.items():
            if p_name in personality_configs:
                depth = personality_configs[p_name].reasoning_depth
                weighted_depth += depth_scores[depth] * weight
                
        # Round to nearest depth
        rounded_depth = round(weighted_depth)
        depth_values = list(depth_scores.keys())
        hybrid.reasoning_depth = depth_values[min(rounded_depth, len(depth_values) - 1)]
        
        # Combine uncertainty approach
        approach_votes = {
            UncertaintyApproach.CAUTIOUS: 0.0,
            UncertaintyApproach.BALANCED: 0.0,
            UncertaintyApproach.BOLD: 0.0
        }
        
        for p_name, weight in weights.items():
            if p_name in personality_configs:
                approach = personality_configs[p_name].uncertainty_approach
                approach_votes[approach] += weight
                
        # Choose approach with highest vote
        hybrid.uncertainty_approach = max(approach_votes.items(), key=lambda x: x[1])[0]
        
        # Combine thinking style
        style_votes = {
            ThinkingStyle.ANALYTICAL: 0.0,
            ThinkingStyle.CREATIVE: 0.0,
            ThinkingStyle.BALANCED: 0.0,
            ThinkingStyle.BAYESIAN: 0.0
        }
        
        for p_name, weight in weights.items():
            if p_name in personality_configs:
                style = personality_configs[p_name].thinking_style
                style_votes[style] += weight
                
        # Choose style with highest vote
        hybrid.thinking_style = max(style_votes.items(), key=lambda x: x[1])[0]
        
        # Combine temperature
        weighted_temp = 0.0
        for p_name, weight in weights.items():
            if p_name in personality_configs:
                temp = personality_configs[p_name].temperature
                weighted_temp += temp * weight
                
        hybrid.temperature = weighted_temp
        
        # Combine expert persona
        persona_votes = {}
        for p_name, weight in weights.items():
            if p_name in personality_configs:
                persona = personality_configs[p_name].expert_persona
                if persona:
                    if persona not in persona_votes:
                        persona_votes[persona] = 0.0
                    persona_votes[persona] += weight
                    
        if persona_votes:
            # Choose persona with highest vote
            hybrid.expert_persona = max(persona_votes.items(), key=lambda x: x[1])[0]
    
    def _combine_custom_traits(
        self,
        hybrid: PersonalityConfig,
        personality_configs: Dict[str, PersonalityConfig],
        weights: Dict[str, float]
    ) -> None:
        """
        Combine custom traits from multiple personalities.
        
        Args:
            hybrid: The hybrid personality to update
            personality_configs: Dictionary of personality configurations
            weights: Dictionary of normalized weights
        """
        # Find all unique traits
        all_traits = set()
        for p_name in weights:
            if p_name in personality_configs:
                all_traits.update(personality_configs[p_name].traits.keys())
        
        # Combine numeric traits with weighted average
        for trait_name in all_traits:
            trait_values = []
            trait_weights = []
            
            for p_name, weight in weights.items():
                if p_name in personality_configs:
                    traits = personality_configs[p_name].traits
                    if trait_name in traits:
                        trait = traits[trait_name]
                        if isinstance(trait.value, (int, float)):
                            trait_values.append(trait.value)
                            trait_weights.append(weight)
                        elif trait_name not in hybrid.traits:
                            # For non-numeric traits, use the first one encountered
                            hybrid.traits[trait_name] = PersonalityTrait(
                                name=trait_name,
                                description=trait.description,
                                value=trait.value
                            )
            
            if trait_values:
                # Calculate weighted average for numeric traits
                weighted_avg = sum(v * w for v, w in zip(trait_values, trait_weights)) / sum(trait_weights)
                
                # Add to hybrid
                if trait_name not in hybrid.traits:
                    # Find a description from any personality that has this trait
                    description = ""
                    for p_config in personality_configs.values():
                        if trait_name in p_config.traits:
                            description = p_config.traits[trait_name].description
                            break
                            
                    hybrid.traits[trait_name] = PersonalityTrait(
                        name=trait_name,
                        description=description,
                        value=weighted_avg
                    )
                else:
                    hybrid.traits[trait_name].value = weighted_avg
    
    def _combine_template_variables(
        self,
        hybrid: PersonalityConfig,
        personality_configs: Dict[str, PersonalityConfig],
        weights: Dict[str, float]
    ) -> None:
        """
        Combine template variables from multiple personalities.
        
        Args:
            hybrid: The hybrid personality to update
            personality_configs: Dictionary of personality configurations
            weights: Dictionary of normalized weights
        """
        # Find all unique template variables
        all_vars = set()
        for p_name in weights:
            if p_name in personality_configs:
                all_vars.update(personality_configs[p_name].template_variables.keys())
        
        # For each variable, choose the one from the highest-weighted personality
        for var_name in all_vars:
            best_p_name = None
            best_weight = -1
            
            for p_name, weight in weights.items():
                if p_name in personality_configs:
                    p_config = personality_configs[p_name]
                    if var_name in p_config.template_variables and weight > best_weight:
                        best_p_name = p_name
                        best_weight = weight
            
            if best_p_name:
                value = personality_configs[best_p_name].template_variables[var_name]
                hybrid.template_variables[var_name] = value
    
    def _save_adapted_personality(self, personality: PersonalityConfig) -> None:
        """
        Save an adapted personality to a file.
        
        Args:
            personality: The personality configuration to save
        """
        if not self.save_dir:
            return
            
        # Create a filename
        filename = f"{personality.name}.json"
        file_path = self.save_dir / filename
        
        # Convert to dict
        personality_dict = personality.to_dict()
        
        # Save to file
        with open(file_path, "w") as f:
            json.dump(personality_dict, f, indent=2)
            
        logger.info(f"Saved adapted personality to {file_path}")


class PersonalityFeedbackLoop:
    """
    Implements a feedback loop for personality adaptation.
    
    This class tracks forecast performance and uses it to adapt
    personalities over time.
    """
    
    def __init__(
        self,
        base_personality_configs: Dict[str, PersonalityConfig],
        save_dir: Optional[str] = None,
        learning_rate: float = 0.1
    ):
        """
        Initialize the feedback loop.
        
        Args:
            base_personality_configs: Dictionary of base personality configurations
            save_dir: Optional directory to save adapted personalities
            learning_rate: Rate at which traits are adjusted (0.0-1.0)
        """
        self.base_personalities = base_personality_configs
        
        # Create adapters for each personality
        self.adapters = {
            name: PersonalityAdapter(
                base_personality=config,
                learning_rate=learning_rate,
                save_dir=save_dir
            )
            for name, config in base_personality_configs.items()
        }
        
        # Track performance history
        self.performance_history: Dict[str, Dict[str, List[float]]] = {}
        
        # Track domain-specific personalities
        self.domain_personalities: Dict[str, Dict[str, PersonalityConfig]] = {}
    
    def record_performance(
        self,
        personality_name: str,
        domain: str,
        performance: float
    ) -> None:
        """
        Record performance for a personality in a domain.
        
        Args:
            personality_name: Name of the personality
            domain: Domain of the forecast
            performance: Performance metric (-1.0 to 1.0 scale)
        """
        # Initialize history if needed
        if personality_name not in self.performance_history:
            self.performance_history[personality_name] = {}
            
        if domain not in self.performance_history[personality_name]:
            self.performance_history[personality_name][domain] = []
            
        # Record performance
        self.performance_history[personality_name][domain].append(performance)
        
        # Adapt personality if needed
        self._adapt_personality(personality_name, domain, performance)
    
    def get_optimal_personality(
        self,
        domain: str,
        create_if_missing: bool = True
    ) -> Optional[PersonalityConfig]:
        """
        Get the optimal personality for a domain.
        
        Args:
            domain: Domain to get the optimal personality for
            create_if_missing: Whether to create a new personality if none exists
            
        Returns:
            Optimal personality configuration or None if not available
        """
        # Check if we have a domain-specific personality
        if domain in self.domain_personalities:
            # Return the best performing personality
            personalities = self.domain_personalities[domain]
            if personalities:
                return max(
                    personalities.values(),
                    key=lambda p: self._get_average_performance(p.name, domain)
                )
        
        if not create_if_missing:
            return None
            
        # Create a new domain-specific personality
        return self._create_domain_personality(domain)
    
    def _adapt_personality(
        self,
        personality_name: str,
        domain: str,
        performance: float
    ) -> None:
        """
        Adapt a personality based on performance.
        
        Args:
            personality_name: Name of the personality
            domain: Domain of the forecast
            performance: Performance metric (-1.0 to 1.0 scale)
        """
        if personality_name not in self.adapters:
            return
            
        # Get the adapter
        adapter = self.adapters[personality_name]
        
        # Adapt the personality
        adapted = adapter.adapt_to_feedback(domain, performance)
        
        # Store the adapted personality
        if domain not in self.domain_personalities:
            self.domain_personalities[domain] = {}
            
        self.domain_personalities[domain][personality_name] = adapted
    
    def _create_domain_personality(self, domain: str) -> PersonalityConfig:
        """
        Create a new domain-specific personality.
        
        Args:
            domain: Domain to create a personality for
            
        Returns:
            Domain-specific personality configuration
        """
        # Find the best performing personalities for this domain
        personality_scores = {}
        
        for p_name, history in self.performance_history.items():
            if domain in history:
                scores = history[domain]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    personality_scores[p_name] = avg_score
        
        # If no history, use all personalities with equal weight
        if not personality_scores:
            personality_scores = {p_name: 1.0 for p_name in self.base_personalities}
        
        # Create a hybrid personality
        if domain not in self.domain_personalities:
            self.domain_personalities[domain] = {}
            
        # Get an adapter (use the first one)
        adapter = next(iter(self.adapters.values()))
        
        # Create the hybrid personality
        hybrid = adapter.create_hybrid_personality(
            personality_weights=personality_scores,
            personality_configs=self.base_personalities,
            name=f"hybrid_{domain}",
            description=f"Hybrid personality optimized for {domain} domain"
        )
        
        # Store the hybrid personality
        self.domain_personalities[domain]["hybrid"] = hybrid
        
        return hybrid
    
    def _get_average_performance(
        self,
        personality_name: str,
        domain: str
    ) -> float:
        """
        Get the average performance for a personality in a domain.
        
        Args:
            personality_name: Name of the personality
            domain: Domain to check
            
        Returns:
            Average performance or 0.0 if no history
        """
        if personality_name in self.performance_history:
            history = self.performance_history[personality_name]
            if domain in history:
                scores = history[domain]
                if scores:
                    return sum(scores) / len(scores)
        
        return 0.0 