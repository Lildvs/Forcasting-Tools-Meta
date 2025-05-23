"""
Personality Management Package

This package provides a system for managing personality configurations
that can be applied to forecasting prompts.
"""

from forecasting_tools.personality_management.manager import PersonalityManager
from forecasting_tools.personality_management.template_manager import TemplateManager
from forecasting_tools.personality_management.config import (
    PersonalityConfig,
    PersonalityTrait,
    ReasoningDepth,
    UncertaintyApproach,
    ThinkingStyle,
)

__all__ = [
    "PersonalityManager",
    "TemplateManager",
    "PersonalityConfig",
    "PersonalityTrait",
    "ReasoningDepth",
    "UncertaintyApproach",
    "ThinkingStyle",
] 