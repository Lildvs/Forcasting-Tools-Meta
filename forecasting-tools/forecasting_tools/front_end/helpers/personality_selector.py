import logging
from typing import List, Optional, Dict, Any, Callable

import streamlit as st

from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.config import PersonalityConfig

logger = logging.getLogger(__name__)


class PersonalitySelector:
    """
    A component for selecting and managing personalities in the Streamlit UI.
    
    This component provides:
    - Dropdown selection of available personalities
    - Display of personality attributes
    - Filtering by personality traits
    - Recently used personalities tracking
    """
    
    @staticmethod
    def display_selector(
        key_prefix: str = "personality",
        on_change: Optional[Callable] = None,
        show_details: bool = True,
        filter_by_domain: Optional[str] = None
    ) -> Optional[str]:
        """
        Display a personality selector component.
        
        Args:
            key_prefix: Prefix for session state keys
            on_change: Optional callback function when selection changes
            show_details: Whether to show personality details
            filter_by_domain: Optional domain to filter personalities by
            
        Returns:
            Selected personality name or None if none selected
        """
        # Initialize personality manager
        personality_manager = PersonalityManager()
        
        # Get available personalities
        available_personalities = personality_manager.list_available_personalities()
        
        # Apply domain filtering if needed
        if filter_by_domain:
            # In a real implementation, we would filter based on domain performance
            # For now, just mention that filtering is applied
            st.info(f"Showing personalities optimized for: {filter_by_domain}")
        
        # Get user's recent personalities from session state
        recent_key = f"{key_prefix}_recent"
        if recent_key not in st.session_state:
            st.session_state[recent_key] = []
            
        recent_personalities = st.session_state[recent_key]
        
        # Define selection key
        selection_key = f"{key_prefix}_selection"
        
        # Create tabs for different selection methods
        tab1, tab2 = st.tabs(["All Personalities", "Recent Personalities"])
        
        selected_personality = None
        
        with tab1:
            # Display all available personalities
            selected_personality = st.selectbox(
                "Select a personality:",
                options=available_personalities,
                index=0 if available_personalities else None,
                key=selection_key,
                on_change=on_change
            )
            
        with tab2:
            # Display recent personalities or a message if none
            if recent_personalities:
                selected_recent = st.selectbox(
                    "Recent personalities:",
                    options=recent_personalities,
                    key=f"{key_prefix}_recent_selection",
                    on_change=lambda: setattr(st.session_state, selection_key, 
                                             st.session_state[f"{key_prefix}_recent_selection"])
                )
            else:
                st.info("No recently used personalities")
        
        # Update recent personalities list if a selection was made
        if selected_personality and selected_personality not in recent_personalities:
            recent_personalities.insert(0, selected_personality)
            # Keep only the 5 most recent
            st.session_state[recent_key] = recent_personalities[:5]
        
        # Display personality details if requested
        if show_details and selected_personality:
            PersonalitySelector.display_personality_details(selected_personality)
            
        return selected_personality
    
    @staticmethod
    def display_personality_details(personality_name: str) -> None:
        """
        Display details of a personality.
        
        Args:
            personality_name: Name of the personality to display
        """
        try:
            # Load personality configuration
            personality_manager = PersonalityManager()
            personality = personality_manager.load_personality(personality_name)
            
            # Create expandable section for details
            with st.expander("Personality Details", expanded=False):
                # Display basic info
                st.markdown(f"### {personality.name}")
                if personality.description:
                    st.markdown(f"*{personality.description}*")
                
                # Create columns for core traits
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Thinking Style:**")
                    st.markdown(f"*{personality.thinking_style.value}*")
                
                with col2:
                    st.markdown("**Uncertainty Approach:**")
                    st.markdown(f"*{personality.uncertainty_approach.value}*")
                
                with col3:
                    st.markdown("**Reasoning Depth:**")
                    st.markdown(f"*{personality.reasoning_depth.value}*")
                
                # Display temperature if available
                if hasattr(personality, "temperature"):
                    st.markdown(f"**Temperature:** {personality.temperature}")
                
                # Display additional traits
                if personality.traits:
                    st.markdown("#### Traits")
                    for trait_name, trait in personality.traits.items():
                        st.markdown(f"**{trait_name}:** {trait.value}")
        
        except Exception as e:
            st.error(f"Error loading personality details: {str(e)}")
    
    @staticmethod
    def get_optimal_personality_for_domain(domain: str) -> str:
        """
        Get the optimal personality for a specific domain.
        
        Args:
            domain: The domain to get an optimal personality for
            
        Returns:
            Name of the optimal personality
        """
        # This would normally use domain-specific data from performance tracking
        # For now, use a simple mapping
        domain_mapping = {
            "economics": "economist",
            "finance": "economist",
            "politics": "bayesian",
            "technology": "creative",
            "science": "bayesian",
            "health": "cautious",
            "sports": "bayesian",
            "entertainment": "creative",
            "geopolitics": "bayesian",
            "environment": "cautious",
            "energy": "economist",
            "social": "creative",
        }
        
        return domain_mapping.get(domain.lower(), "balanced") 