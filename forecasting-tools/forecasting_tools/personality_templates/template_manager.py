"""
Template Manager

This module provides functionality for loading and applying templates for different
research approaches and forecast types.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type

from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)

logger = logging.getLogger(__name__)

class TemplateManager:
    """
    Advanced template manager that handles loading and combining templates
    based on question type, bot version, and personality traits.
    """
    
    # Base directory for templates
    BASE_DIR = Path(__file__).parent
    
    def __init__(self, bot_version: str = "q2"):
        """
        Initialize the template manager.
        
        Args:
            bot_version: The bot version to use for templates (e.g., "q1", "q2", "q3", "q4")
        """
        self.bot_version = bot_version
        self.templates = self._load_templates()
        self.personality_traits = self._load_personality_traits()
        
    def _load_templates(self) -> Dict[str, Any]:
        """Load all templates from the templates directory."""
        templates = {
            "research": {},
            "forecasts": {
                "binary": {},
                "multiple_choice": {},
                "numeric": {}
            }
        }
        
        # Load research templates
        research_dir = self.BASE_DIR / "research"
        if research_dir.exists():
            for template_file in research_dir.glob("*.txt"):
                try:
                    with open(template_file, "r") as f:
                        templates["research"][template_file.stem] = f.read()
                except Exception as e:
                    logger.error(f"Error loading research template {template_file}: {e}")
        
        # Load forecast templates (binary, multiple_choice, numeric)
        forecasts_dir = self.BASE_DIR / "forecasts"
        if forecasts_dir.exists():
            for forecast_type in ["binary", "multiple_choice", "numeric"]:
                forecast_dir = forecasts_dir / forecast_type
                if forecast_dir.exists():
                    for template_file in forecast_dir.glob("*.txt"):
                        try:
                            with open(template_file, "r") as f:
                                templates["forecasts"][forecast_type][template_file.stem] = f.read()
                        except Exception as e:
                            logger.error(f"Error loading forecast template {template_file}: {e}")
        
        return templates
    
    def _load_personality_traits(self) -> Dict[str, Any]:
        """Load personality traits from JSON files."""
        traits = {}
        traits_dir = self.BASE_DIR / "personality_traits"
        if traits_dir.exists():
            for trait_file in traits_dir.glob("*.json"):
                try:
                    with open(trait_file, "r") as f:
                        traits[trait_file.stem] = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading personality trait {trait_file}: {e}")
        return traits
    
    def get_research_template(self, research_type: str = "default_research") -> str:
        """
        Get a research template of the specified type.
        
        Args:
            research_type: The type of research template to get
            
        Returns:
            The research template
        """
        if research_type not in self.templates["research"]:
            logger.warning(f"Research template {research_type} not found, using default")
            research_type = "default_research"
            
        return self.templates["research"].get(research_type, "")
    
    def get_forecast_template(self, question_type: Type, bot_version: Optional[str] = None) -> str:
        """
        Get a forecast template for the specified question type and bot version.
        
        Args:
            question_type: The type of question
            bot_version: The bot version to use (e.g., "q1", "q2", "q3")
            
        Returns:
            The forecast template
        """
        bot_version = bot_version or self.bot_version
        forecast_type = self._get_forecast_type_for_question(question_type)
        template_key = f"{bot_version}_{forecast_type}"
        
        if template_key not in self.templates["forecasts"][forecast_type]:
            logger.warning(f"Forecast template {template_key} not found, using default")
            template_key = f"q2_{forecast_type}"  # Default to Q2
        
        return self.templates["forecasts"][forecast_type].get(template_key, "")
    
    def apply_personality_traits(self, template: str, traits_config: Dict[str, str]) -> str:
        """
        Apply personality traits to a template.
        
        Args:
            template: The template to apply traits to
            traits_config: The traits configuration
            
        Returns:
            The template with traits applied
        """
        replacements = {}
        
        # Apply reasoning depth
        if "reasoning_depth" in traits_config and traits_config["reasoning_depth"] in self.personality_traits.get("reasoning_depth", {}):
            depth = traits_config["reasoning_depth"]
            replacements.update(self.personality_traits["reasoning_depth"][depth])
        
        # Apply uncertainty approach
        if "uncertainty_approach" in traits_config and traits_config["uncertainty_approach"] in self.personality_traits.get("uncertainty_approach", {}):
            approach = traits_config["uncertainty_approach"]
            replacements.update(self.personality_traits["uncertainty_approach"][approach])
        
        # Apply expert persona
        if "expert_persona" in traits_config and traits_config["expert_persona"] in self.personality_traits.get("expert_persona", {}):
            persona = traits_config["expert_persona"]
            replacements.update(self.personality_traits["expert_persona"][persona])
        
        # Apply thinking style
        if "thinking_style" in traits_config and traits_config["thinking_style"] in self.personality_traits.get("thinking_style", {}):
            style = traits_config["thinking_style"]
            replacements.update(self.personality_traits["thinking_style"][style])
        
        # Replace placeholders in template
        for key, value in replacements.items():
            template = template.replace(f"{{{key}}}", value)
        
        return template
    
    def _get_forecast_type_for_question(self, question_type: Type) -> str:
        """
        Map question type to forecast type.
        
        Args:
            question_type: The type of question
            
        Returns:
            The forecast type
        """
        if issubclass(question_type, BinaryQuestion):
            return "binary"
        elif issubclass(question_type, MultipleChoiceQuestion):
            return "multiple_choice"
        elif issubclass(question_type, NumericQuestion):
            return "numeric"
        else:
            logger.warning(f"Unknown question type: {question_type}, defaulting to binary")
            return "binary" 