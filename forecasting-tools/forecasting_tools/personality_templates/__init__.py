"""
Personality Templates System

This module provides a templating system for forecasting prompts, allowing for
different personalities, research approaches, and question types.
"""

from forecasting_tools.personality_templates.personality_config import PersonalityConfig
from forecasting_tools.personality_templates.personality_manager import PersonalityManager
from forecasting_tools.personality_templates.template_manager import TemplateManager

__all__ = ["TemplateManager", "PersonalityConfig", "PersonalityManager"] 