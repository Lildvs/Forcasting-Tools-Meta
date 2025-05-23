import logging
from typing import Optional, Dict, Any, List

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64

from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.config import PersonalityConfig
from forecasting_tools.forecast_helpers.competition import CompetitionTracker, CompetitionMetric
from forecasting_tools.personality_management.diversity import PersonalityDiversityScorer

logger = logging.getLogger(__name__)


class PersonalityDetails:
    """
    A component for displaying detailed information about personalities.
    
    This component provides:
    - Visual representation of personality traits
    - Performance metrics across domains
    - Trait comparison and analysis
    - Domain-specific effectiveness visualization
    """
    
    @staticmethod
    def display_personality_profile(
        personality_name: str,
        show_performance: bool = True,
        show_radar: bool = True,
        show_domain_effectiveness: bool = True
    ) -> None:
        """
        Display a comprehensive profile of a personality.
        
        Args:
            personality_name: Name of the personality to display
            show_performance: Whether to show performance metrics
            show_radar: Whether to show the radar chart
            show_domain_effectiveness: Whether to show domain effectiveness
        """
        try:
            # Load personality configuration
            personality_manager = PersonalityManager()
            personality = personality_manager.load_personality(personality_name)
            
            # Display basic info
            st.header(f"{personality.name} Profile")
            
            if personality.description:
                st.markdown(f"*{personality.description}*")
            
            # Create three columns for core trait visualization
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Thinking Style")
                st.markdown(f"**{personality.thinking_style.value}**")
                
                # Visualize thinking style distribution
                thinking_styles = ["Analytical", "Creative", "Balanced", "Bayesian"]
                values = [1.0 if ts.lower() == personality.thinking_style.value.lower() else 0.2 for ts in thinking_styles]
                
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.bar(thinking_styles, values, color='skyblue')
                ax.set_ylim(0, 1.2)
                ax.set_title("Thinking Style")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            with col2:
                st.subheader("Uncertainty Approach")
                st.markdown(f"**{personality.uncertainty_approach.value}**")
                
                # Visualize uncertainty approach
                uncertainty_approaches = ["Cautious", "Balanced", "Bold"]
                values = [1.0 if ua.lower() == personality.uncertainty_approach.value.lower() else 0.2 for ua in uncertainty_approaches]
                
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.bar(uncertainty_approaches, values, color='lightgreen')
                ax.set_ylim(0, 1.2)
                ax.set_title("Uncertainty Approach")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            with col3:
                st.subheader("Reasoning Depth")
                st.markdown(f"**{personality.reasoning_depth.value}**")
                
                # Visualize reasoning depth
                reasoning_depths = ["Shallow", "Moderate", "Deep", "Exhaustive"]
                values = [1.0 if rd.lower() == personality.reasoning_depth.value.lower() else 0.2 for rd in reasoning_depths]
                
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.bar(reasoning_depths, values, color='salmon')
                ax.set_ylim(0, 1.2)
                ax.set_title("Reasoning Depth")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            # Display radar chart of traits
            if show_radar and personality.traits:
                st.subheader("Personality Traits")
                
                # Get numeric traits
                numeric_traits = {}
                for trait_name, trait in personality.traits.items():
                    if isinstance(trait.value, (int, float)):
                        numeric_traits[trait_name] = trait.value
                
                if numeric_traits:
                    # Create radar chart
                    PersonalityDetails._create_trait_radar_chart(numeric_traits)
                else:
                    st.info("No numeric traits available for radar chart visualization")
            
            # Display domain-specific effectiveness
            if show_domain_effectiveness:
                st.subheader("Domain Effectiveness")
                
                # In a real implementation, this would use actual performance data
                # For now, use mock data for demonstration
                domains = [
                    "Economics", "Politics", "Science", "Technology", 
                    "Health", "Sports", "Entertainment", "Geopolitics"
                ]
                
                # Generate mock effectiveness scores based on personality traits
                effectiveness_scores = PersonalityDetails._generate_mock_domain_effectiveness(
                    personality, domains
                )
                
                # Create effectiveness visualization
                PersonalityDetails._create_domain_effectiveness_chart(domains, effectiveness_scores)
            
            # Display performance metrics if requested
            if show_performance:
                st.subheader("Performance Metrics")
                
                # In a real implementation, this would fetch actual performance data
                # For now, use mock data for demonstration
                metrics = {
                    "Accuracy": 0.78,
                    "Calibration": 0.82,
                    "Information Score": 0.65,
                    "Relative Score": 0.71
                }
                
                # Display metrics
                cols = st.columns(len(metrics))
                for i, (metric_name, value) in enumerate(metrics.items()):
                    with cols[i]:
                        st.metric(
                            label=metric_name,
                            value=f"{value:.2f}",
                            delta=f"{(value - 0.5):.2f}",
                            delta_color="normal"
                        )
                
                st.info("Note: These metrics are simulated for demonstration purposes.")
        
        except Exception as e:
            st.error(f"Error displaying personality profile: {str(e)}")
    
    @staticmethod
    def compare_personalities(
        personality_names: List[str],
        comparison_traits: Optional[List[str]] = None
    ) -> None:
        """
        Display a comparison between multiple personalities.
        
        Args:
            personality_names: List of personality names to compare
            comparison_traits: Optional list of specific traits to compare
        """
        if not personality_names:
            st.warning("No personalities selected for comparison")
            return
            
        try:
            # Load personality configurations
            personality_manager = PersonalityManager()
            personalities = {}
            
            for name in personality_names:
                try:
                    personalities[name] = personality_manager.load_personality(name)
                except Exception as e:
                    st.warning(f"Could not load personality '{name}': {str(e)}")
            
            if not personalities:
                st.error("No valid personalities to compare")
                return
                
            # Display basic comparison
            st.header("Personality Comparison")
            
            # Compare core traits
            st.subheader("Core Traits")
            
            # Create comparison table
            data = []
            for name, personality in personalities.items():
                data.append({
                    "Personality": name,
                    "Thinking Style": personality.thinking_style.value,
                    "Uncertainty Approach": personality.uncertainty_approach.value,
                    "Reasoning Depth": personality.reasoning_depth.value
                })
            
            st.dataframe(data)
            
            # Compare numeric traits with radar chart
            all_numeric_traits = set()
            personality_traits = {}
            
            for name, personality in personalities.items():
                numeric_traits = {}
                for trait_name, trait in personality.traits.items():
                    if isinstance(trait.value, (int, float)):
                        numeric_traits[trait_name] = trait.value
                        all_numeric_traits.add(trait_name)
                
                personality_traits[name] = numeric_traits
            
            if all_numeric_traits:
                st.subheader("Trait Comparison")
                
                # Filter by requested traits if specified
                if comparison_traits:
                    all_numeric_traits = [t for t in all_numeric_traits if t in comparison_traits]
                
                # Create radar chart comparing all personalities
                PersonalityDetails._create_comparison_radar_chart(
                    personality_traits, list(all_numeric_traits)
                )
            
            # Display diversity analysis
            st.subheader("Diversity Analysis")
            
            # Use the PersonalityDiversityScorer to measure diversity
            diversity_scorer = PersonalityDiversityScorer()
            diversity_metrics = diversity_scorer.calculate_ensemble_diversity(
                {name: personality for name, personality in personalities.items()}
            )
            
            # Display overall diversity score
            st.metric(
                label="Overall Diversity",
                value=f"{diversity_metrics['overall_diversity']:.2f}",
                help="Score between 0-1 measuring how diverse these personalities are"
            )
            
            # Display uniqueness scores
            st.caption("Uniqueness Scores")
            uniqueness_data = [
                {"Personality": name, "Uniqueness Score": score}
                for name, score in diversity_metrics["uniqueness_scores"].items()
            ]
            st.dataframe(uniqueness_data)
            
            # Display coverage gaps if any
            if diversity_metrics["coverage_gaps"]:
                st.caption("Coverage Gaps")
                for gap in diversity_metrics["coverage_gaps"]:
                    st.info(f"Gap in {gap['trait']}: {', '.join(gap['missing_values'])}")
        
        except Exception as e:
            st.error(f"Error comparing personalities: {str(e)}")
    
    @staticmethod
    def _create_trait_radar_chart(traits: Dict[str, float]) -> None:
        """
        Create and display a radar chart for personality traits.
        
        Args:
            traits: Dictionary mapping trait names to values
        """
        # Get trait names and values
        trait_names = list(traits.keys())
        trait_values = list(traits.values())
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(trait_names), endpoint=False).tolist()
        
        # Close the polygon
        trait_values = trait_values + [trait_values[0]]
        angles = angles + [angles[0]]
        trait_names = trait_names + [trait_names[0]]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
        
        # Plot data
        ax.plot(angles, trait_values, 'o-', linewidth=2)
        ax.fill(angles, trait_values, alpha=0.25)
        
        # Set labels
        ax.set_thetagrids(np.degrees(angles), trait_names)
        
        # Add title
        ax.set_title("Personality Trait Profile", y=1.1)
        
        # Display chart
        st.pyplot(fig)
    
    @staticmethod
    def _create_comparison_radar_chart(
        personality_traits: Dict[str, Dict[str, float]],
        trait_names: List[str]
    ) -> None:
        """
        Create and display a radar chart comparing multiple personalities.
        
        Args:
            personality_traits: Dict mapping personality names to trait dictionaries
            trait_names: List of trait names to include in the comparison
        """
        # Create angles for radar chart
        angles = np.linspace(0, 2*np.pi, len(trait_names), endpoint=False).tolist()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
        
        # Create a color map for different personalities
        colors = plt.cm.get_cmap('tab10', len(personality_traits))
        
        # Plot each personality
        for i, (name, traits) in enumerate(personality_traits.items()):
            # Get values for each trait (default to 0 if trait not present)
            values = [traits.get(trait, 0) for trait in trait_names]
            
            # Close the polygon
            values = values + [values[0]]
            angles_plot = angles + [angles[0]]
            
            # Plot
            ax.plot(angles_plot, values, 'o-', linewidth=2, label=name, color=colors(i))
            ax.fill(angles_plot, values, alpha=0.1, color=colors(i))
        
        # Set labels
        trait_names_plot = trait_names + [trait_names[0]]
        ax.set_thetagrids(np.degrees(angles + [angles[0]]), trait_names_plot)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Add title
        ax.set_title("Personality Trait Comparison", y=1.1)
        
        # Display chart
        st.pyplot(fig)
    
    @staticmethod
    def _create_domain_effectiveness_chart(
        domains: List[str],
        effectiveness_scores: List[float]
    ) -> None:
        """
        Create and display a chart showing domain effectiveness.
        
        Args:
            domains: List of domain names
            effectiveness_scores: List of effectiveness scores for each domain
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot horizontal bar chart
        bars = ax.barh(domains, effectiveness_scores, color='skyblue')
        
        # Add values to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height()/2,
                f"{effectiveness_scores[i]:.2f}",
                va='center'
            )
        
        # Set limits
        ax.set_xlim(0, 1.0)
        
        # Add labels and title
        ax.set_xlabel("Effectiveness Score")
        ax.set_title("Domain-Specific Effectiveness")
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Display chart
        st.pyplot(fig)
    
    @staticmethod
    def _generate_mock_domain_effectiveness(
        personality: PersonalityConfig,
        domains: List[str]
    ) -> List[float]:
        """
        Generate mock domain effectiveness scores based on personality traits.
        
        Args:
            personality: Personality configuration
            domains: List of domains
            
        Returns:
            List of effectiveness scores
        """
        # In a real implementation, this would use actual performance data
        # For this demonstration, generate scores based on personality traits
        
        # Define domain affinities for different thinking styles
        thinking_style_affinity = {
            "analytical": {
                "economics": 0.9, "finance": 0.9, "science": 0.8, 
                "technology": 0.7, "health": 0.7, "geopolitics": 0.6,
                "sports": 0.5, "entertainment": 0.4
            },
            "creative": {
                "entertainment": 0.9, "technology": 0.8, "politics": 0.7,
                "sports": 0.7, "science": 0.6, "health": 0.6,
                "economics": 0.5, "finance": 0.4
            },
            "balanced": {
                "politics": 0.8, "geopolitics": 0.8, "health": 0.7,
                "science": 0.7, "technology": 0.7, "economics": 0.7,
                "finance": 0.6, "sports": 0.6, "entertainment": 0.6
            },
            "bayesian": {
                "science": 0.9, "geopolitics": 0.8, "economics": 0.8,
                "finance": 0.8, "politics": 0.7, "health": 0.7,
                "technology": 0.7, "sports": 0.6, "entertainment": 0.5
            }
        }
        
        # Get affinities for this personality's thinking style
        style = personality.thinking_style.value.lower()
        affinities = thinking_style_affinity.get(style, {})
        
        # Generate scores with some randomness
        scores = []
        for domain in domains:
            base_score = affinities.get(domain.lower(), 0.5)
            
            # Add some variance based on uncertainty approach
            if personality.uncertainty_approach.value.lower() == "cautious":
                # Cautious personalities are more consistent
                variance = np.random.normal(0, 0.05)
            else:
                # Bold personalities have more variance
                variance = np.random.normal(0, 0.1)
                
            # Adjust score
            score = base_score + variance
            
            # Ensure score is between 0 and 1
            score = max(0.1, min(0.95, score))
            
            scores.append(score)
            
        return scores 