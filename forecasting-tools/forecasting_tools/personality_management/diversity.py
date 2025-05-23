"""
Personality Diversity Module

This module provides tools for measuring and analyzing diversity among
personality configurations in forecasting ensembles.
"""

import logging
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any, Union
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from io import BytesIO
import base64

from forecasting_tools.personality_management.config import (
    PersonalityConfig,
    ThinkingStyle,
    UncertaintyApproach,
    ReasoningDepth,
    PersonalityTrait
)


logger = logging.getLogger(__name__)


class PersonalityDiversityScorer:
    """
    Tool for measuring and analyzing diversity among personality configurations.
    
    This class provides methods to:
    - Calculate diversity scores for ensembles of personalities
    - Identify gaps in personality coverage
    - Recommend personality additions for optimal diversity
    - Visualize personality distributions
    """
    
    def __init__(self):
        """Initialize the diversity scorer."""
        pass
    
    def calculate_ensemble_diversity(
        self, personalities: Dict[str, PersonalityConfig]
    ) -> Dict[str, Any]:
        """
        Calculate overall diversity metrics for an ensemble of personalities.
        
        Args:
            personalities: Dictionary mapping identifiers to personality configurations
            
        Returns:
            Dictionary of diversity metrics
        """
        if not personalities:
            return {
                "overall_diversity": 0.0,
                "trait_diversity": {},
                "coverage_gaps": [],
                "uniqueness_scores": {}
            }
        
        # Calculate pairwise distances
        distances = self._calculate_pairwise_distances(personalities)
        
        # Calculate overall diversity (average pairwise distance)
        overall_diversity = self._calculate_overall_diversity(distances)
        
        # Calculate trait-specific diversity
        trait_diversity = self._calculate_trait_diversity(personalities)
        
        # Identify coverage gaps
        coverage_gaps = self._identify_coverage_gaps(personalities)
        
        # Calculate uniqueness scores
        uniqueness_scores = self._calculate_uniqueness_scores(personalities, distances)
        
        return {
            "overall_diversity": overall_diversity,
            "trait_diversity": trait_diversity,
            "coverage_gaps": coverage_gaps,
            "uniqueness_scores": uniqueness_scores
        }
    
    def recommend_additions(
        self, 
        current_personalities: Dict[str, PersonalityConfig],
        available_personalities: Dict[str, PersonalityConfig],
        num_additions: int = 1
    ) -> List[str]:
        """
        Recommend personalities to add to increase ensemble diversity.
        
        Args:
            current_personalities: Dictionary of current personality configurations
            available_personalities: Dictionary of available personality configurations
            num_additions: Number of personalities to recommend adding
            
        Returns:
            List of recommended personality identifiers
        """
        if not current_personalities or not available_personalities:
            # If no current personalities, recommend the most diverse set from available
            if not current_personalities:
                return self._select_diverse_subset(available_personalities, num_additions)
            return []
        
        # Calculate current coverage gaps
        gaps = self._identify_coverage_gaps(current_personalities)
        
        # Calculate how each available personality would fill gaps
        gap_filling_scores = {}
        for pid, personality in available_personalities.items():
            if pid in current_personalities:
                continue
                
            # Check how well this personality fills gaps
            score = self._calculate_gap_filling_score(personality, gaps)
            gap_filling_scores[pid] = score
        
        # Sort by gap filling score
        sorted_personalities = sorted(
            gap_filling_scores.items(), key=lambda x: x[1], reverse=True
        )
        
        # Return top N recommendations
        return [pid for pid, _ in sorted_personalities[:num_additions]]
    
    def create_diversity_visualization(
        self, personalities: Dict[str, PersonalityConfig]
    ) -> Union[str, Figure]:
        """
        Create a visualization of personality diversity.
        
        Args:
            personalities: Dictionary mapping identifiers to personality configurations
            
        Returns:
            Either a base64-encoded PNG image or a matplotlib Figure
        """
        if not personalities:
            # Create empty figure with message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No personalities provided", ha="center", va="center", fontsize=14)
            ax.axis("off")
            return fig
        
        # Create figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot thinking style distribution
        self._plot_thinking_style_distribution(personalities, axs[0, 0])
        
        # Plot uncertainty approach distribution
        self._plot_uncertainty_approach_distribution(personalities, axs[0, 1])
        
        # Plot reasoning depth distribution
        self._plot_reasoning_depth_distribution(personalities, axs[1, 0])
        
        # Plot personality trait radar chart
        self._plot_personality_trait_radar(personalities, axs[1, 1])
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert to base64 encoded string
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        buffer.seek(0)
        image_str = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close(fig)
        
        return f"data:image/png;base64,{image_str}"
    
    def _calculate_pairwise_distances(
        self, personalities: Dict[str, PersonalityConfig]
    ) -> Dict[Tuple[str, str], float]:
        """
        Calculate pairwise distances between all personalities.
        
        Args:
            personalities: Dictionary mapping identifiers to personality configurations
            
        Returns:
            Dictionary mapping personality pairs to distances
        """
        distances = {}
        personality_ids = list(personalities.keys())
        
        for i, pid1 in enumerate(personality_ids):
            for j, pid2 in enumerate(personality_ids):
                if i < j:  # Calculate each pair only once
                    p1 = personalities[pid1]
                    p2 = personalities[pid2]
                    distance = self._calculate_personality_distance(p1, p2)
                    distances[(pid1, pid2)] = distance
                    distances[(pid2, pid1)] = distance
        
        return distances
    
    def _calculate_personality_distance(
        self, personality1: PersonalityConfig, personality2: PersonalityConfig
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
    
    def _calculate_overall_diversity(
        self, distances: Dict[Tuple[str, str], float]
    ) -> float:
        """
        Calculate overall diversity from pairwise distances.
        
        Args:
            distances: Dictionary mapping personality pairs to distances
            
        Returns:
            Overall diversity score between 0.0 and 1.0
        """
        if not distances:
            return 0.0
            
        # Calculate average distance
        return sum(distances.values()) / len(distances)
    
    def _calculate_trait_diversity(
        self, personalities: Dict[str, PersonalityConfig]
    ) -> Dict[str, float]:
        """
        Calculate diversity for each trait across personalities.
        
        Args:
            personalities: Dictionary mapping identifiers to personality configurations
            
        Returns:
            Dictionary mapping trait names to diversity scores
        """
        trait_diversity = {}
        
        # Core traits
        thinking_styles = [p.thinking_style for p in personalities.values()]
        uncertainty_approaches = [p.uncertainty_approach for p in personalities.values()]
        reasoning_depths = [p.reasoning_depth for p in personalities.values()]
        
        # Calculate entropy-based diversity for each core trait
        trait_diversity["thinking_style"] = self._calculate_categorical_diversity(thinking_styles)
        trait_diversity["uncertainty_approach"] = self._calculate_categorical_diversity(uncertainty_approaches)
        trait_diversity["reasoning_depth"] = self._calculate_categorical_diversity(reasoning_depths)
        
        # Find all custom traits across personalities
        all_traits = set()
        for p in personalities.values():
            all_traits.update(p.traits.keys())
        
        # Calculate diversity for each custom trait
        for trait_name in all_traits:
            trait_values = []
            for p in personalities.values():
                if trait_name in p.traits:
                    trait_values.append(p.traits[trait_name].value)
            
            # Calculate diversity based on trait type
            if trait_values and all(isinstance(v, (int, float)) for v in trait_values):
                # Numeric trait
                trait_diversity[trait_name] = self._calculate_numeric_diversity(trait_values)
            else:
                # Categorical trait
                trait_diversity[trait_name] = self._calculate_categorical_diversity(trait_values)
        
        return trait_diversity
    
    def _calculate_categorical_diversity(self, values: List[Any]) -> float:
        """
        Calculate diversity for categorical values using entropy.
        
        Args:
            values: List of categorical values
            
        Returns:
            Diversity score between 0.0 and 1.0
        """
        if not values:
            return 0.0
            
        # Count occurrences of each value
        value_counts = {}
        for value in values:
            value_str = str(value)  # Convert to string for consistent counting
            if value_str in value_counts:
                value_counts[value_str] += 1
            else:
                value_counts[value_str] = 1
        
        # Calculate entropy
        entropy = 0.0
        total_count = len(values)
        for count in value_counts.values():
            probability = count / total_count
            entropy -= probability * math.log2(probability)
        
        # Normalize entropy to [0, 1] range
        max_entropy = math.log2(len(value_counts))
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 0.0
        
        return normalized_entropy
    
    def _calculate_numeric_diversity(self, values: List[Union[int, float]]) -> float:
        """
        Calculate diversity for numeric values using coefficient of variation.
        
        Args:
            values: List of numeric values
            
        Returns:
            Diversity score between 0.0 and 1.0
        """
        if not values:
            return 0.0
            
        # Calculate mean and standard deviation
        mean = sum(values) / len(values)
        if mean == 0:
            return 0.0
            
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = math.sqrt(variance)
        
        # Calculate coefficient of variation
        cv = std_dev / abs(mean)
        
        # Normalize to [0, 1] range using 1 - 1/(1+cv)
        normalized_cv = 1.0 - 1.0 / (1.0 + cv)
        
        return normalized_cv
    
    def _identify_coverage_gaps(
        self, personalities: Dict[str, PersonalityConfig]
    ) -> List[Dict[str, Any]]:
        """
        Identify gaps in personality trait coverage.
        
        Args:
            personalities: Dictionary mapping identifiers to personality configurations
            
        Returns:
            List of gap descriptions with trait name and missing values
        """
        gaps = []
        
        # Check core traits
        thinking_styles = [p.thinking_style for p in personalities.values()]
        all_thinking_styles = set(ThinkingStyle)
        missing_thinking_styles = all_thinking_styles - set(thinking_styles)
        if missing_thinking_styles:
            gaps.append({
                "trait": "thinking_style",
                "missing_values": [ts.value for ts in missing_thinking_styles]
            })
        
        uncertainty_approaches = [p.uncertainty_approach for p in personalities.values()]
        all_uncertainty_approaches = set(UncertaintyApproach)
        missing_uncertainty_approaches = all_uncertainty_approaches - set(uncertainty_approaches)
        if missing_uncertainty_approaches:
            gaps.append({
                "trait": "uncertainty_approach",
                "missing_values": [ua.value for ua in missing_uncertainty_approaches]
            })
        
        reasoning_depths = [p.reasoning_depth for p in personalities.values()]
        all_reasoning_depths = set(ReasoningDepth)
        missing_reasoning_depths = all_reasoning_depths - set(reasoning_depths)
        if missing_reasoning_depths:
            gaps.append({
                "trait": "reasoning_depth",
                "missing_values": [rd.value for rd in missing_reasoning_depths]
            })
        
        # For numeric traits, check if we have good coverage across the range
        numeric_traits = {}
        for p in personalities.values():
            for trait_name, trait in p.traits.items():
                if isinstance(trait.value, (int, float)):
                    if trait_name not in numeric_traits:
                        numeric_traits[trait_name] = []
                    numeric_traits[trait_name].append(trait.value)
        
        # Analyze coverage of numeric traits (looking for clustered values)
        for trait_name, values in numeric_traits.items():
            if len(values) >= 2:  # Need at least 2 values to analyze distribution
                values.sort()
                
                # Check for large gaps in the distribution
                max_gap = 0
                for i in range(1, len(values)):
                    gap = values[i] - values[i-1]
                    max_gap = max(max_gap, gap)
                
                # If the max gap is large relative to the range, report it
                value_range = values[-1] - values[0]
                if value_range > 0 and max_gap / value_range > 0.4:  # Gap is > 40% of range
                    gaps.append({
                        "trait": trait_name,
                        "gap_type": "large_gap",
                        "gap_size": max_gap,
                        "range": value_range,
                        "suggestion": f"Add personality with {trait_name} around {(values[0] + values[-1]) / 2}"
                    })
        
        return gaps
    
    def _calculate_uniqueness_scores(
        self, 
        personalities: Dict[str, PersonalityConfig],
        distances: Dict[Tuple[str, str], float]
    ) -> Dict[str, float]:
        """
        Calculate uniqueness score for each personality.
        
        Args:
            personalities: Dictionary mapping identifiers to personality configurations
            distances: Dictionary mapping personality pairs to distances
            
        Returns:
            Dictionary mapping personality IDs to uniqueness scores
        """
        uniqueness_scores = {}
        
        for pid in personalities:
            # Calculate average distance to all other personalities
            other_pids = [other_pid for other_pid in personalities if other_pid != pid]
            if other_pids:
                avg_distance = sum(distances[(pid, other_pid)] for other_pid in other_pids) / len(other_pids)
                uniqueness_scores[pid] = avg_distance
            else:
                uniqueness_scores[pid] = 1.0  # Only one personality, so maximally unique
        
        return uniqueness_scores
    
    def _calculate_gap_filling_score(
        self, personality: PersonalityConfig, gaps: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate how well a personality fills identified gaps.
        
        Args:
            personality: Personality configuration to evaluate
            gaps: List of identified gaps
            
        Returns:
            Gap filling score between 0.0 and 1.0
        """
        if not gaps:
            return 0.0
            
        gap_filling_points = 0.0
        total_possible_points = 0.0
        
        for gap in gaps:
            if gap["trait"] == "thinking_style" and personality.thinking_style.value in gap["missing_values"]:
                gap_filling_points += 1.0
                
            elif gap["trait"] == "uncertainty_approach" and personality.uncertainty_approach.value in gap["missing_values"]:
                gap_filling_points += 1.0
                
            elif gap["trait"] == "reasoning_depth" and personality.reasoning_depth.value in gap["missing_values"]:
                gap_filling_points += 1.0
                
            elif gap["trait"] in personality.traits and gap.get("gap_type") == "large_gap":
                # For numeric trait gaps, check if the value helps fill the gap
                trait_value = personality.traits[gap["trait"]].value
                if isinstance(trait_value, (int, float)):
                    # Calculate how centered this value is in the gap
                    suggestion_value = float(gap["suggestion"].split("around ")[1])
                    distance_to_center = abs(trait_value - suggestion_value)
                    # Normalize by gap size
                    normalized_distance = distance_to_center / (gap["gap_size"] / 2)
                    # Convert to score (closer to center = higher score)
                    gap_filling_points += max(0, 1.0 - normalized_distance)
                    
            total_possible_points += 1.0
        
        if total_possible_points > 0:
            return gap_filling_points / total_possible_points
        else:
            return 0.0
    
    def _select_diverse_subset(
        self, personalities: Dict[str, PersonalityConfig], subset_size: int
    ) -> List[str]:
        """
        Select a diverse subset of personalities.
        
        Args:
            personalities: Dictionary mapping identifiers to personality configurations
            subset_size: Number of personalities to select
            
        Returns:
            List of selected personality identifiers
        """
        if not personalities:
            return []
            
        if len(personalities) <= subset_size:
            return list(personalities.keys())
        
        # Calculate pairwise distances
        distances = self._calculate_pairwise_distances(personalities)
        
        # Greedy algorithm to select diverse subset
        personality_ids = list(personalities.keys())
        selected_ids = []
        
        # Start with the first personality
        selected_ids.append(personality_ids[0])
        
        # Greedily add personalities that maximize minimum distance to already selected
        while len(selected_ids) < subset_size:
            best_id = None
            best_min_distance = -1
            
            for pid in personality_ids:
                if pid not in selected_ids:
                    # Calculate minimum distance to already selected personalities
                    min_distance = min(distances[(pid, selected_pid)] for selected_pid in selected_ids)
                    
                    if min_distance > best_min_distance:
                        best_min_distance = min_distance
                        best_id = pid
            
            if best_id:
                selected_ids.append(best_id)
            else:
                break
        
        return selected_ids
    
    def _plot_thinking_style_distribution(
        self, personalities: Dict[str, PersonalityConfig], ax
    ) -> None:
        """
        Plot distribution of thinking styles.
        
        Args:
            personalities: Dictionary mapping identifiers to personality configurations
            ax: Matplotlib axis to plot on
        """
        # Count thinking styles
        thinking_styles = {}
        for p in personalities.values():
            style = p.thinking_style.value
            thinking_styles[style] = thinking_styles.get(style, 0) + 1
        
        # Plot
        labels = list(thinking_styles.keys())
        sizes = list(thinking_styles.values())
        
        ax.pie(
            sizes, 
            labels=labels, 
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.Paired(np.linspace(0, 1, len(labels)))
        )
        ax.axis('equal')
        ax.set_title('Thinking Style Distribution')
    
    def _plot_uncertainty_approach_distribution(
        self, personalities: Dict[str, PersonalityConfig], ax
    ) -> None:
        """
        Plot distribution of uncertainty approaches.
        
        Args:
            personalities: Dictionary mapping identifiers to personality configurations
            ax: Matplotlib axis to plot on
        """
        # Count uncertainty approaches
        approaches = {}
        for p in personalities.values():
            approach = p.uncertainty_approach.value
            approaches[approach] = approaches.get(approach, 0) + 1
        
        # Plot
        labels = list(approaches.keys())
        sizes = list(approaches.values())
        
        ax.pie(
            sizes, 
            labels=labels, 
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.Paired(np.linspace(0, 1, len(labels)))
        )
        ax.axis('equal')
        ax.set_title('Uncertainty Approach Distribution')
    
    def _plot_reasoning_depth_distribution(
        self, personalities: Dict[str, PersonalityConfig], ax
    ) -> None:
        """
        Plot distribution of reasoning depths.
        
        Args:
            personalities: Dictionary mapping identifiers to personality configurations
            ax: Matplotlib axis to plot on
        """
        # Count reasoning depths
        depths = {}
        for p in personalities.values():
            depth = p.reasoning_depth.value
            depths[depth] = depths.get(depth, 0) + 1
        
        # Plot
        labels = list(depths.keys())
        sizes = list(depths.values())
        
        ax.pie(
            sizes, 
            labels=labels, 
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.Paired(np.linspace(0, 1, len(labels)))
        )
        ax.axis('equal')
        ax.set_title('Reasoning Depth Distribution')
    
    def _plot_personality_trait_radar(
        self, personalities: Dict[str, PersonalityConfig], ax
    ) -> None:
        """
        Plot radar chart of common numeric personality traits.
        
        Args:
            personalities: Dictionary mapping identifiers to personality configurations
            ax: Matplotlib axis to plot on
        """
        # Find common numeric traits
        common_traits = set()
        for p in personalities.values():
            for trait_name, trait in p.traits.items():
                if isinstance(trait.value, (int, float)):
                    common_traits.add(trait_name)
        
        if not common_traits:
            ax.text(0.5, 0.5, "No numeric traits found", ha="center", va="center")
            ax.axis('off')
            return
        
        # Convert to list and sort for consistency
        common_traits = sorted(list(common_traits))
        
        # Create radar chart
        num_traits = len(common_traits)
        angles = np.linspace(0, 2*np.pi, num_traits, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        # Plot each personality
        for pid, personality in personalities.items():
            values = []
            for trait_name in common_traits:
                if trait_name in personality.traits and isinstance(personality.traits[trait_name].value, (int, float)):
                    values.append(personality.traits[trait_name].value)
                else:
                    values.append(0.0)
            
            # Close the polygon
            values += values[:1]
            
            # Plot
            ax.plot(angles, values, linewidth=1, label=pid)
            ax.fill(angles, values, alpha=0.1)
        
        # Set chart properties
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(common_traits)
        ax.set_title('Personality Trait Radar')
        
        # Add legend if not too many personalities
        if len(personalities) <= 5:
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))


def calculate_personality_coverage(
    personalities: Dict[str, PersonalityConfig]
) -> Dict[str, float]:
    """
    Calculate coverage metrics for a set of personalities.
    
    Args:
        personalities: Dictionary mapping identifiers to personality configurations
        
    Returns:
        Dictionary of coverage metrics
    """
    diversity_scorer = PersonalityDiversityScorer()
    diversity_metrics = diversity_scorer.calculate_ensemble_diversity(personalities)
    
    # Extract core trait coverage
    coverage = {}
    
    # Calculate coverage of thinking styles
    all_thinking_styles = set(style.value for style in ThinkingStyle)
    present_thinking_styles = set(p.thinking_style.value for p in personalities.values())
    coverage["thinking_style_coverage"] = len(present_thinking_styles) / len(all_thinking_styles)
    
    # Calculate coverage of uncertainty approaches
    all_uncertainty_approaches = set(approach.value for approach in UncertaintyApproach)
    present_uncertainty_approaches = set(p.uncertainty_approach.value for p in personalities.values())
    coverage["uncertainty_approach_coverage"] = len(present_uncertainty_approaches) / len(all_uncertainty_approaches)
    
    # Calculate coverage of reasoning depths
    all_reasoning_depths = set(depth.value for depth in ReasoningDepth)
    present_reasoning_depths = set(p.reasoning_depth.value for p in personalities.values())
    coverage["reasoning_depth_coverage"] = len(present_reasoning_depths) / len(all_reasoning_depths)
    
    # Overall coverage (average of core trait coverages)
    coverage["overall_coverage"] = (
        coverage["thinking_style_coverage"] +
        coverage["uncertainty_approach_coverage"] +
        coverage["reasoning_depth_coverage"]
    ) / 3.0
    
    # Add overall diversity
    coverage["overall_diversity"] = diversity_metrics["overall_diversity"]
    
    return coverage


def calculate_ensemble_diversity_score(personalities: Dict[str, PersonalityConfig]) -> float:
    """
    Calculate a single diversity score for an ensemble of personalities.
    
    Args:
        personalities: Dictionary mapping identifiers to personality configurations
        
    Returns:
        Diversity score between 0.0 and 1.0
    """
    coverage = calculate_personality_coverage(personalities)
    
    # Combine coverage and diversity with equal weights
    return (coverage["overall_coverage"] + coverage["overall_diversity"]) / 2.0 