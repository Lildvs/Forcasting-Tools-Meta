"""
Personality Configuration

This module provides functionality for loading and managing personality configurations
that define how traits are combined and applied to templates.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class PersonalityConfig:
    """
    Manages personality configurations that define how traits are combined.
    """
    
    # Base directory for personality configurations
    BASE_DIR = Path(__file__).parent
    PERSONALITIES_DIR = BASE_DIR / "personalities"
    
    # Default personality to use if none is specified
    DEFAULT_PERSONALITY = "balanced"
    
    def __init__(self, personality_name: Optional[str] = None):
        """
        Initialize the personality configuration.
        
        Args:
            personality_name: The name of the personality to use (e.g., "balanced", "cautious", "creative")
        """
        self.personality_name = personality_name or self.DEFAULT_PERSONALITY
        self.config = self._load_config(self.personality_name)
        
    def _load_config(self, personality_name: str) -> Dict[str, Any]:
        """
        Load a personality configuration from a JSON file.
        
        Args:
            personality_name: The name of the personality to load
            
        Returns:
            The personality configuration
        """
        config_path = self.PERSONALITIES_DIR / f"{personality_name}.json"
        
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                logger.info(f"Loaded personality: {personality_name}")
                return config
        except FileNotFoundError:
            logger.warning(f"Personality {personality_name} not found, using default")
            # If specified personality not found, fall back to default
            if personality_name != self.DEFAULT_PERSONALITY:
                return self._load_config(self.DEFAULT_PERSONALITY)
            # If default also not found, return empty config
            logger.error(f"Default personality {self.DEFAULT_PERSONALITY} not found")
            return {}
        except Exception as e:
            logger.error(f"Error loading personality {personality_name}: {e}")
            if personality_name != self.DEFAULT_PERSONALITY:
                return self._load_config(self.DEFAULT_PERSONALITY)
            return {}
    
    def get_traits_config(self) -> Dict[str, str]:
        """
        Get the traits configuration for this personality.
        
        Returns:
            The traits configuration
        """
        return {
            "reasoning_depth": self.config.get("reasoning_depth", "medium"),
            "uncertainty_approach": self.config.get("uncertainty_approach", "balanced"),
            "expert_persona": self.config.get("expert_persona", "forecaster"),
            "thinking_style": self.config.get("thinking_style", "analytical")
        }
    
    def get_thinking_parameters(self) -> Dict[str, Any]:
        """
        Get the thinking parameters for this personality.
        
        Returns:
            The thinking parameters
        """
        thinking = self.config.get("thinking", {})
        return {
            "type": thinking.get("type", "enabled"),
            "budget_tokens": thinking.get("budget_tokens", 32000)
        }
    
    def get_name(self) -> str:
        """
        Get the display name of this personality.
        
        Returns:
            The display name
        """
        return self.config.get("name", self.personality_name.capitalize())
    
    def get_description(self) -> str:
        """
        Get the description of this personality.
        
        Returns:
            The description
        """
        return self.config.get("description", "")
    
    @classmethod
    def get_available_personalities(cls) -> List[str]:
        """
        Get a list of available personalities.
        
        Returns:
            List of personality names
        """
        personalities = []
        if cls.PERSONALITIES_DIR.exists():
            personalities = [p.stem for p in cls.PERSONALITIES_DIR.glob("*.json")]
        return personalities 