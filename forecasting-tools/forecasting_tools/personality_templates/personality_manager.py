"""
Personality Manager

This module provides a high-level interface for managing personalities and templates
for forecasting bots.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Type

from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.personality_templates.personality_config import PersonalityConfig
from forecasting_tools.personality_templates.template_manager import TemplateManager

logger = logging.getLogger(__name__)

class PersonalityManager:
    """
    High-level manager for personality-based templates.
    Provides an interface for forecasting bots to get appropriate prompts.
    """
    
    def __init__(
        self,
        bot_version: str = "q2",
        personality_name: Optional[str] = None,
        research_type: str = "default_research"
    ):
        """
        Initialize the personality manager.
        
        Args:
            bot_version: The bot version to use (e.g., "q1", "q2", "q3", "q4")
            personality_name: The name of the personality to use
            research_type: The type of research template to use
        """
        self.bot_version = bot_version
        self.template_manager = TemplateManager(bot_version=bot_version)
        self.personality_config = PersonalityConfig(personality_name=personality_name)
        self.research_type = research_type
        self.traits_config = self.personality_config.get_traits_config()
        
        # Log what we're using
        logger.info(
            f"Using bot version: {bot_version}, " +
            f"personality: {self.personality_config.get_name()}, " +
            f"research type: {research_type}"
        )
    
    def get_prompt(self, prompt_type: str, **kwargs) -> str:
        """
        Get a prompt of the specified type with personality traits applied.
        
        Args:
            prompt_type: The type of prompt to get (e.g., "research_prompt", "binary_forecast_prompt")
            **kwargs: Additional kwargs to format the prompt with
            
        Returns:
            The formatted prompt
        """
        template = ""
        
        if prompt_type == "research_prompt":
            template = self.template_manager.get_research_template(self.research_type)
        elif prompt_type == "binary_forecast_prompt":
            template = self.template_manager.get_forecast_template(BinaryQuestion, self.bot_version)
        elif prompt_type == "multiple_choice_forecast_prompt":
            template = self.template_manager.get_forecast_template(MultipleChoiceQuestion, self.bot_version)
        elif prompt_type == "numeric_forecast_prompt":
            template = self.template_manager.get_forecast_template(NumericQuestion, self.bot_version)
        else:
            logger.warning(f"Unknown prompt type: {prompt_type}, returning empty template")
            return ""
        
        # Apply personality traits
        template = self.template_manager.apply_personality_traits(template, self.traits_config)
        
        # Format with provided kwargs
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing required key in prompt formatting: {e}")
            return template
    
    def get_thinking_config(self) -> Dict[str, Any]:
        """
        Get thinking configuration parameters for this personality.
        
        Returns:
            Dict with thinking configuration parameters
        """
        thinking_params = self.personality_config.get_thinking_parameters()
        
        if thinking_params.get("type") == "enabled":
            return {
                "thinking": True,
                "thinking_budget_tokens": thinking_params.get("budget_tokens", 32000)
            }
        
        return {} 