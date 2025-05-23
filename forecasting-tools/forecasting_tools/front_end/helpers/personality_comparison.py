import logging
from typing import List, Optional, Dict, Any

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.config import PersonalityConfig
from forecasting_tools.personality_management.diversity import PersonalityDiversityScorer
from forecasting_tools.forecast_helpers.competition import CompetitionTracker, CompetitionMetric
from forecasting_tools.front_end.helpers.personality_details import PersonalityDetails

logger = logging.getLogger(__name__)


class PersonalityComparison:
    """
    A component for comparing multiple personalities and their performance.
    
    This component provides:
    - Multi-personality selection interface
    - Performance comparison across domains
    - Trait comparison visualizations
    - Ensemble diversity analysis
    """
    
    @staticmethod
    def display_comparison_interface(key_prefix: str = "comparison") -> None:
        """
        Display a full personality comparison interface.
        
        Args:
            key_prefix: Prefix for session state keys
        """
        st.header("Personality Comparison")
        
        # Initialize personality manager
        personality_manager = PersonalityManager()
        
        # Get available personalities
        available_personalities = personality_manager.list_available_personalities()
        
        if not available_personalities:
            st.warning("No personalities available for comparison")
            return
        
        # Allow multi-selection of personalities
        selected_personalities = st.multiselect(
            "Select personalities to compare:",
            options=available_personalities,
            default=available_personalities[:2] if len(available_personalities) >= 2 else available_personalities[:1],
            key=f"{key_prefix}_selection"
        )
        
        if not selected_personalities:
            st.info("Please select at least one personality to continue")
            return
        
        # Show comparison options
        st.subheader("Comparison Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            compare_traits = st.checkbox("Compare Traits", value=True, key=f"{key_prefix}_traits")
            compare_performance = st.checkbox("Compare Performance", value=True, key=f"{key_prefix}_performance")
        
        with col2:
            analyze_diversity = st.checkbox("Analyze Diversity", value=True, key=f"{key_prefix}_diversity")
            show_domain_specific = st.checkbox("Show Domain Performance", value=True, key=f"{key_prefix}_domains")
        
        # Display comparison
        if compare_traits:
            PersonalityComparison._display_trait_comparison(selected_personalities)
        
        if compare_performance:
            PersonalityComparison._display_performance_comparison(selected_personalities)
        
        if analyze_diversity and len(selected_personalities) > 1:
            PersonalityComparison._display_diversity_analysis(selected_personalities)
        
        if show_domain_specific:
            PersonalityComparison._display_domain_comparison(selected_personalities)
    
    @staticmethod
    def _display_trait_comparison(personality_names: List[str]) -> None:
        """
        Display a comparison of personality traits.
        
        Args:
            personality_names: List of personality names to compare
        """
        st.subheader("Trait Comparison")
        
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
        
        # Create comparison table for core traits
        st.markdown("#### Core Traits")
        
        data = []
        for name, personality in personalities.items():
            data.append({
                "Personality": name,
                "Thinking Style": personality.thinking_style.value,
                "Uncertainty Approach": personality.uncertainty_approach.value,
                "Reasoning Depth": personality.reasoning_depth.value,
                "Temperature": getattr(personality, "temperature", "N/A")
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        # Compare numeric traits if available
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
            st.markdown("#### Trait Radar Chart")
            
            # Get traits sorted for consistency
            all_numeric_traits = sorted(list(all_numeric_traits))
            
            # Let user select traits to include
            selected_traits = st.multiselect(
                "Select traits to include in radar chart:",
                options=all_numeric_traits,
                default=all_numeric_traits[:min(5, len(all_numeric_traits))],
                key="trait_selection"
            )
            
            if selected_traits:
                # Use PersonalityDetails to create radar chart
                PersonalityDetails._create_comparison_radar_chart(
                    personality_traits, selected_traits
                )
            else:
                st.info("Select at least one trait to display the radar chart")
        else:
            st.info("No numeric traits available for comparison")
    
    @staticmethod
    def _display_performance_comparison(personality_names: List[str]) -> None:
        """
        Display a comparison of personality performance.
        
        Args:
            personality_names: List of personality names to compare
        """
        st.subheader("Performance Comparison")
        
        # In a real implementation, this would use the CompetitionTracker to get actual data
        # For now, use mock data for demonstration
        
        # Create mock metrics for each personality
        metrics = ["Accuracy", "Calibration", "Information Score", "Expected Score"]
        
        # Generate random performance data with some patterns
        data = []
        for name in personality_names:
            # Base values depend on personality type
            if "bayesian" in name.lower():
                base = [0.75, 0.85, 0.65, 0.78]
            elif "economist" in name.lower():
                base = [0.72, 0.78, 0.70, 0.75]
            elif "creative" in name.lower():
                base = [0.65, 0.70, 0.80, 0.72]
            elif "cautious" in name.lower():
                base = [0.70, 0.82, 0.60, 0.71]
            else:
                base = [0.70, 0.75, 0.70, 0.73]
            
            # Add some randomness
            values = [min(0.95, max(0.5, b + np.random.normal(0, 0.05))) for b in base]
            
            row = {"Personality": name}
            for i, metric in enumerate(metrics):
                row[metric] = values[i]
            
            data.append(row)
        
        # Create dataframe and display
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        # Create a bar chart for visualization
        if len(personality_names) > 0:
            st.markdown("#### Performance Visualization")
            
            # Let user select which metric to visualize
            selected_metric = st.selectbox(
                "Select metric to visualize:",
                options=metrics,
                index=0,
                key="metric_selection"
            )
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Extract data for the selected metric
            names = df["Personality"].tolist()
            values = df[selected_metric].tolist()
            
            # Plot
            bars = ax.bar(names, values, color='skyblue')
            
            # Add values above bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 0.01,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom"
                )
            
            # Set limits and labels
            ax.set_ylim(0, 1.0)
            ax.set_ylabel(selected_metric)
            ax.set_title(f"{selected_metric} by Personality")
            
            # Rotate x-axis labels if there are many personalities
            if len(personality_names) > 3:
                plt.xticks(rotation=45, ha="right")
            
            # Display chart
            st.pyplot(fig)
            
            st.info("Note: These metrics are simulated for demonstration purposes.")
    
    @staticmethod
    def _display_diversity_analysis(personality_names: List[str]) -> None:
        """
        Display diversity analysis for selected personalities.
        
        Args:
            personality_names: List of personality names to analyze
        """
        if len(personality_names) < 2:
            return
            
        st.subheader("Diversity Analysis")
        
        # Load personality configurations
        personality_manager = PersonalityManager()
        personalities = {}
        
        for name in personality_names:
            try:
                personalities[name] = personality_manager.load_personality(name)
            except Exception as e:
                st.warning(f"Could not load personality '{name}': {str(e)}")
        
        if len(personalities) < 2:
            st.error("At least two valid personalities are needed for diversity analysis")
            return
        
        # Use the PersonalityDiversityScorer to analyze diversity
        diversity_scorer = PersonalityDiversityScorer()
        
        try:
            diversity_metrics = diversity_scorer.calculate_ensemble_diversity(personalities)
            
            # Display overall diversity score
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Overall Diversity Score",
                    value=f"{diversity_metrics['overall_diversity']:.2f}",
                    help="Score between 0-1 measuring overall ensemble diversity"
                )
            
            # Display trait diversity scores
            with col2:
                trait_div = diversity_metrics.get("trait_diversity", {})
                if "thinking_style" in trait_div:
                    st.metric(
                        label="Thinking Style Diversity",
                        value=f"{trait_div['thinking_style']:.2f}"
                    )
            
            # Display uniqueness scores for each personality
            st.markdown("#### Uniqueness Scores")
            uniqueness_data = []
            
            for name, score in diversity_metrics["uniqueness_scores"].items():
                uniqueness_data.append({
                    "Personality": name,
                    "Uniqueness Score": score
                })
            
            if uniqueness_data:
                df = pd.DataFrame(uniqueness_data)
                df = df.sort_values("Uniqueness Score", ascending=False)
                st.dataframe(df, use_container_width=True)
            
            # Display coverage gaps if any
            if diversity_metrics["coverage_gaps"]:
                st.markdown("#### Coverage Gaps")
                for gap in diversity_metrics["coverage_gaps"]:
                    st.info(f"Gap in {gap['trait']}: {', '.join(gap['missing_values'])}")
                
                # Add recommendation for improving diversity
                st.markdown("#### Recommendations")
                st.markdown("To improve ensemble diversity, consider adding personalities with:")
                
                for gap in diversity_metrics["coverage_gaps"]:
                    trait = gap['trait']
                    missing = gap['missing_values']
                    st.markdown(f"- {trait.replace('_', ' ').title()}: {', '.join(missing)}")
            else:
                st.success("No significant coverage gaps found in this ensemble!")
        
        except Exception as e:
            st.error(f"Error analyzing diversity: {str(e)}")
    
    @staticmethod
    def _display_domain_comparison(personality_names: List[str]) -> None:
        """
        Display domain-specific performance comparison.
        
        Args:
            personality_names: List of personality names to compare
        """
        st.subheader("Domain Performance")
        
        # Define domains to compare
        domains = [
            "Economics", "Finance", "Politics", "Technology", 
            "Science", "Health", "Sports", "Entertainment"
        ]
        
        # In a real implementation, this would use actual performance data
        # For now, generate mock data based on personality traits
        personality_manager = PersonalityManager()
        domain_data = []
        
        for name in personality_names:
            try:
                # Load personality
                personality = personality_manager.load_personality(name)
                
                # Generate mock scores
                scores = PersonalityDetails._generate_mock_domain_effectiveness(personality, domains)
                
                # Create row with personality name and domain scores
                row = {"Personality": name}
                for i, domain in enumerate(domains):
                    row[domain] = scores[i]
                
                domain_data.append(row)
                
            except Exception as e:
                st.warning(f"Could not generate domain data for '{name}': {str(e)}")
        
        # Create dataframe and display
        if domain_data:
            df = pd.DataFrame(domain_data)
            st.dataframe(df, use_container_width=True)
            
            # Create heatmap for visualization
            st.markdown("#### Domain Performance Heatmap")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Prepare data for heatmap
            heatmap_data = df.drop("Personality", axis=1).values
            y_labels = df["Personality"].tolist()
            x_labels = domains
            
            # Create heatmap
            im = ax.imshow(heatmap_data, cmap='YlGnBu')
            
            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("Performance Score", rotation=-90, va="bottom")
            
            # Set labels
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_yticks(np.arange(len(y_labels)))
            ax.set_xticklabels(x_labels)
            ax.set_yticklabels(y_labels)
            
            # Rotate x-axis labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add text annotations
            for i in range(len(y_labels)):
                for j in range(len(x_labels)):
                    text = ax.text(j, i, f"{heatmap_data[i, j]:.2f}",
                                  ha="center", va="center", color="black")
            
            # Set title
            ax.set_title("Domain Performance Heatmap")
            
            # Display chart
            st.pyplot(fig)
            
            # Add recommendation on optimal domains
            st.markdown("#### Domain Recommendations")
            
            for name in personality_names:
                # Get best domains for this personality
                if name in df["Personality"].values:
                    row = df[df["Personality"] == name].iloc[0]
                    domain_scores = [(domain, row[domain]) for domain in domains]
                    domain_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # Get top 3 domains
                    top_domains = domain_scores[:3]
                    
                    st.markdown(f"**{name}** performs best in:")
                    for domain, score in top_domains:
                        st.markdown(f"- {domain} ({score:.2f})")
            
            st.info("Note: These domain scores are simulated based on personality traits.")
        else:
            st.warning("No valid domain data available for comparison") 