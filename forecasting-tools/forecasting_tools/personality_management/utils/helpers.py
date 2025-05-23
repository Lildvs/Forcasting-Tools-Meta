"""
Personality Management Utilities

This module provides utility functions for working with the personality management system.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from forecasting_tools.personality_management.config import PersonalityConfig


def get_default_personality_dir() -> Path:
    """
    Get the default directory for personality configurations.
    
    Returns:
        Path to the default personality configurations directory
    """
    return Path(__file__).parent.parent / "personalities"


def get_default_template_dir() -> Path:
    """
    Get the default directory for templates.
    
    Returns:
        Path to the default templates directory
    """
    return Path(__file__).parent.parent / "templates"


def get_personality_from_env() -> Optional[str]:
    """
    Get the personality name from environment variables.
    
    Checks for the FORECASTING_PERSONALITY environment variable.
    
    Returns:
        The personality name, or None if not set
    """
    return os.environ.get("FORECASTING_PERSONALITY")


def validate_personality_config(config: Dict) -> List[str]:
    """
    Validate a personality configuration dictionary.
    
    Args:
        config: The personality configuration dictionary
        
    Returns:
        List of validation errors, empty if valid
    """
    errors = []
    
    # Check required fields
    required_fields = ["name", "description"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate reasoning_depth
    if "reasoning_depth" in config:
        valid_depths = ["shallow", "moderate", "deep", "exhaustive"]
        if config["reasoning_depth"] not in valid_depths:
            errors.append(f"Invalid reasoning_depth: {config['reasoning_depth']}. "
                         f"Must be one of {valid_depths}")
    
    # Validate uncertainty_approach
    if "uncertainty_approach" in config:
        valid_approaches = ["overconfident", "balanced", "cautious", "explicit"]
        if config["uncertainty_approach"] not in valid_approaches:
            errors.append(f"Invalid uncertainty_approach: {config['uncertainty_approach']}. "
                         f"Must be one of {valid_approaches}")
    
    # Validate thinking_style
    if "thinking_style" in config:
        valid_styles = ["intuitive", "analytical", "creative", "bayesian", "fermi"]
        if config["thinking_style"] not in valid_styles:
            errors.append(f"Invalid thinking_style: {config['thinking_style']}. "
                         f"Must be one of {valid_styles}")
    
    # Validate temperature
    if "temperature" in config:
        try:
            temp = float(config["temperature"])
            if temp < 0.0 or temp > 1.0:
                errors.append(f"Temperature must be between 0.0 and 1.0, got {temp}")
        except (ValueError, TypeError):
            errors.append(f"Temperature must be a number, got {config['temperature']}")
    
    # Validate traits
    if "traits" in config and isinstance(config["traits"], dict):
        for trait_name, trait_data in config["traits"].items():
            if not isinstance(trait_data, dict):
                errors.append(f"Trait '{trait_name}' must be a dictionary")
                continue
                
            if "description" not in trait_data:
                errors.append(f"Trait '{trait_name}' is missing a description")
                
            if "value" not in trait_data:
                errors.append(f"Trait '{trait_name}' is missing a value")
    
    return errors


def merge_personality_configs(base: Dict, overlay: Dict) -> Dict:
    """
    Merge two personality configurations.
    
    The overlay configuration takes precedence over the base configuration.
    
    Args:
        base: The base configuration
        overlay: The overlay configuration
        
    Returns:
        The merged configuration
    """
    result = base.copy()
    
    # Merge top-level fields
    for key, value in overlay.items():
        if key != "traits" and key != "template_variables":
            result[key] = value
    
    # Merge traits
    if "traits" in overlay:
        if "traits" not in result:
            result["traits"] = {}
            
        for trait_name, trait_data in overlay["traits"].items():
            result["traits"][trait_name] = trait_data
    
    # Merge template variables
    if "template_variables" in overlay:
        if "template_variables" not in result:
            result["template_variables"] = {}
            
        for var_name, var_value in overlay["template_variables"].items():
            result["template_variables"][var_name] = var_value
    
    return result 