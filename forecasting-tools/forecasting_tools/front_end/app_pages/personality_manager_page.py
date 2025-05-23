import logging
import os
import json
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd
import numpy as np
from pydantic import BaseModel

from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.config import (
    PersonalityConfig,
    ThinkingStyle,
    UncertaintyApproach,
    ReasoningDepth,
    PersonalityTrait
)
from forecasting_tools.front_end.helpers.tool_page import ToolPage
from forecasting_tools.front_end.helpers.personality_details import PersonalityDetails
from forecasting_tools.front_end.helpers.personality_comparison import PersonalityComparison
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


class PersonalityInput(Jsonable, BaseModel):
    name: str
    description: Optional[str] = None
    thinking_style: str
    uncertainty_approach: str
    reasoning_depth: str
    temperature: float = 0.7
    traits: Dict[str, Any] = {}


class PersonalityOutput(Jsonable, BaseModel):
    name: str
    description: Optional[str] = None
    thinking_style: str
    uncertainty_approach: str
    reasoning_depth: str
    temperature: float = 0.7
    traits: Dict[str, Any] = {}
    message: str


class PersonalityManagerPage(ToolPage):
    PAGE_DISPLAY_NAME: str = "👤 Personality Manager"
    URL_PATH: str = "/personality-manager"
    INPUT_TYPE = PersonalityInput
    OUTPUT_TYPE = PersonalityOutput
    EXAMPLES_FILE_PATH = None

    # Form input keys
    NAME_INPUT = "personality_name"
    DESCRIPTION_INPUT = "personality_description"
    THINKING_STYLE_INPUT = "thinking_style"
    UNCERTAINTY_APPROACH_INPUT = "uncertainty_approach"
    REASONING_DEPTH_INPUT = "reasoning_depth"
    TEMPERATURE_INPUT = "temperature"
    
    @classmethod
    async def _display_intro_text(cls) -> None:
        st.markdown("""
        # Personality Manager
        
        This tool allows you to view, create, and customize forecaster personalities.
        Each personality has distinct traits that influence how it approaches forecasting tasks.
        """)
        
        # Create tabs for different functionality
        tab1, tab2, tab3 = st.tabs(["Browse Personalities", "Create Custom Personality", "Compare Personalities"])
        
        with tab1:
            await cls._display_browse_personalities()
        
        with tab2:
            await cls._display_create_personality_form()
        
        with tab3:
            PersonalityComparison.display_comparison_interface()

    @classmethod
    async def _display_browse_personalities(cls) -> None:
        """Display a browsable list of available personalities."""
        st.subheader("Available Personalities")
        
        # Get personality manager
        personality_manager = PersonalityManager()
        
        # Get all available personalities
        available_personalities = personality_manager.list_available_personalities()
        
        if not available_personalities:
            st.info("No personalities available.")
            return
        
        # Create expandable sections for each personality
        for name in available_personalities:
            with st.expander(f"{name.capitalize()}"):
                try:
                    # Display personality details
                    PersonalityDetails.display_personality_profile(name)
                    
                    # Add edit button
                    if st.button(f"Edit {name}", key=f"edit_{name}"):
                        # Store personality name in session state for editing
                        st.session_state["edit_personality"] = name
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error loading personality {name}: {str(e)}")
    
    @classmethod
    async def _display_create_personality_form(cls) -> None:
        """Display a form for creating a new personality."""
        st.subheader("Create or Edit Personality")
        
        # Initialize personality manager
        personality_manager = PersonalityManager()
        
        # Check if we're editing an existing personality
        editing_personality = st.session_state.get("edit_personality", None)
        
        # Initial values
        initial_values = {
            "name": "",
            "description": "",
            "thinking_style": ThinkingStyle.BALANCED.value,
            "uncertainty_approach": UncertaintyApproach.BALANCED.value,
            "reasoning_depth": ReasoningDepth.MODERATE.value,
            "temperature": 0.7,
            "traits": {}
        }
        
        # If editing, load existing values
        if editing_personality:
            try:
                personality = personality_manager.load_personality(editing_personality)
                
                initial_values["name"] = editing_personality
                initial_values["description"] = personality.description or ""
                initial_values["thinking_style"] = personality.thinking_style.value
                initial_values["uncertainty_approach"] = personality.uncertainty_approach.value
                initial_values["reasoning_depth"] = personality.reasoning_depth.value
                initial_values["temperature"] = getattr(personality, "temperature", 0.7)
                
                # Convert traits to a simple dictionary
                traits_dict = {}
                for trait_name, trait in personality.traits.items():
                    traits_dict[trait_name] = trait.value
                
                initial_values["traits"] = traits_dict
                
                st.info(f"Editing personality: {editing_personality}")
            except Exception as e:
                st.error(f"Error loading personality {editing_personality}: {str(e)}")
                editing_personality = None
        
        # Create form
        with st.form("personality_form"):
            # Basic information
            name = st.text_input(
                "Personality Name", 
                value=initial_values["name"],
                key=cls.NAME_INPUT,
                disabled=editing_personality is not None
            )
            
            description = st.text_area(
                "Description",
                value=initial_values["description"],
                key=cls.DESCRIPTION_INPUT
            )
            
            # Core traits
            col1, col2, col3 = st.columns(3)
            
            with col1:
                thinking_style = st.selectbox(
                    "Thinking Style",
                    options=[ts.value for ts in ThinkingStyle],
                    index=[ts.value for ts in ThinkingStyle].index(initial_values["thinking_style"]),
                    key=cls.THINKING_STYLE_INPUT
                )
            
            with col2:
                uncertainty_approach = st.selectbox(
                    "Uncertainty Approach",
                    options=[ua.value for ua in UncertaintyApproach],
                    index=[ua.value for ua in UncertaintyApproach].index(initial_values["uncertainty_approach"]),
                    key=cls.UNCERTAINTY_APPROACH_INPUT
                )
            
            with col3:
                reasoning_depth = st.selectbox(
                    "Reasoning Depth",
                    options=[rd.value for rd in ReasoningDepth],
                    index=[rd.value for rd in ReasoningDepth].index(initial_values["reasoning_depth"]),
                    key=cls.REASONING_DEPTH_INPUT
                )
            
            # Temperature
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=1.0,
                value=initial_values["temperature"],
                step=0.1,
                key=cls.TEMPERATURE_INPUT,
                help="Higher values make the personality more creative but less predictable"
            )
            
            # Custom traits
            st.subheader("Custom Traits")
            st.info("Add custom traits to further customize the personality.")
            
            # Display existing traits
            trait_rows = []
            for trait_name, trait_value in initial_values["traits"].items():
                trait_rows.append({"name": trait_name, "value": trait_value})
            
            # Add empty rows for new traits
            for _ in range(3):
                trait_rows.append({"name": "", "value": ""})
            
            # Create editable trait table
            trait_edit = st.experimental_data_editor(
                pd.DataFrame(trait_rows),
                use_container_width=True,
                num_rows="fixed"
            )
            
            # Submit button
            submit_label = "Update Personality" if editing_personality else "Create Personality"
            submitted = st.form_submit_button(submit_label)
            
            if submitted:
                # Validate input
                if not name and not editing_personality:
                    st.error("Personality name is required.")
                    return
                
                # Process traits from the data editor
                traits = {}
                for _, row in trait_edit.iterrows():
                    trait_name = row["name"]
                    trait_value = row["value"]
                    
                    if trait_name and trait_value != "":
                        # Convert numeric values
                        if isinstance(trait_value, str) and trait_value.replace(".", "", 1).isdigit():
                            if "." in trait_value:
                                trait_value = float(trait_value)
                            else:
                                trait_value = int(trait_value)
                        
                        traits[trait_name] = trait_value
                
                # Create personality configuration
                personality_config = {
                    "name": name if not editing_personality else editing_personality,
                    "description": description,
                    "thinking_style": thinking_style,
                    "uncertainty_approach": uncertainty_approach,
                    "reasoning_depth": reasoning_depth,
                    "temperature": temperature,
                    "traits": {}
                }
                
                # Add traits
                for trait_name, trait_value in traits.items():
                    personality_config["traits"][trait_name] = {
                        "name": trait_name,
                        "description": f"Custom trait: {trait_name}",
                        "value": trait_value
                    }
                
                try:
                    # Convert to PersonalityConfig
                    config = PersonalityConfig.from_dict(personality_config)
                    
                    # Save personality
                    save_path = os.path.join(
                        "user_personalities", 
                        f"{config.name}.json"
                    )
                    
                    # Ensure directory exists
                    os.makedirs("user_personalities", exist_ok=True)
                    
                    # Save configuration
                    with open(save_path, "w") as f:
                        json.dump(personality_config, f, indent=2)
                    
                    # Show success message
                    action = "updated" if editing_personality else "created"
                    st.success(f"Personality {config.name} {action} successfully!")
                    
                    # If editing, clear edit state
                    if editing_personality:
                        del st.session_state["edit_personality"]
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error saving personality: {str(e)}")
        
        # Cancel button (outside form)
        if editing_personality and st.button("Cancel Editing"):
            del st.session_state["edit_personality"]
            st.experimental_rerun()

    @classmethod
    async def _get_input(cls) -> PersonalityInput | None:
        # Input is handled in the _display_intro_text method
        return None

    @classmethod
    async def _run_tool(cls, input: PersonalityInput) -> PersonalityOutput:
        # Tool functionality is handled in the _display_intro_text method
        return PersonalityOutput(
            name=input.name if input else "",
            description="",
            thinking_style=ThinkingStyle.BALANCED.value,
            uncertainty_approach=UncertaintyApproach.BALANCED.value,
            reasoning_depth=ReasoningDepth.MODERATE.value,
            temperature=0.7,
            traits={},
            message="Operation completed"
        )

    @classmethod
    async def _save_run_to_coda(
        cls,
        input_to_tool: PersonalityInput,
        output: PersonalityOutput,
        is_premade: bool,
    ) -> None:
        # No need to save to database for this tool
        pass

    @classmethod
    async def _display_outputs(cls, outputs: list[PersonalityOutput]) -> None:
        # Output is handled in the _display_intro_text method
        pass


if __name__ == "__main__":
    PersonalityManagerPage.main() 