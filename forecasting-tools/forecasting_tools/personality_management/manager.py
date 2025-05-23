"""
Personality Manager Module

This module provides the main interface for the personality management system,
allowing loading, validation, and application of personality configurations.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from forecasting_tools.personality_management.config import PersonalityConfig
from forecasting_tools.personality_management.template_manager import TemplateManager


class PersonalityManager:
    """
    Manager for personality configurations.
    
    This class handles loading personality configurations from files,
    applying them to templates, and providing access to personality traits.
    """
    
    def __init__(
        self, 
        personality_name: Optional[str] = None,
        personalities_dir: Optional[str] = None,
        templates_dir: Optional[str] = None
    ):
        """
        Initialize the PersonalityManager.
        
        Args:
            personality_name: Name of the personality to load. If None, uses "balanced"
            personalities_dir: Directory containing personality configuration files.
                              If None, uses the default directory.
            templates_dir: Directory containing template files. If None, uses the default.
        """
        # Set up directories
        if personalities_dir is None:
            current_dir = Path(__file__).parent
            self.personalities_dir = current_dir / "personalities"
        else:
            self.personalities_dir = Path(personalities_dir)
            
        # Initialize template manager
        self.template_manager = TemplateManager(templates_dir)
        
        # Cache for loaded personalities
        self._personality_cache: Dict[str, PersonalityConfig] = {}
        
        # Load the specified personality or the default
        self._current_personality: Optional[PersonalityConfig] = None
        if personality_name:
            self.load_personality(personality_name)
    
    def load_personality(self, name: str) -> PersonalityConfig:
        """
        Load a personality by name.
        
        Args:
            name: Name of the personality to load
            
        Returns:
            The loaded personality configuration
            
        Raises:
            FileNotFoundError: If the personality file doesn't exist
            ValueError: If the personality configuration is invalid
        """
        # Check if already cached
        if name in self._personality_cache:
            self._current_personality = self._personality_cache[name]
            return self._current_personality
            
        # Find the file
        for ext in (".yaml", ".yml", ".json"):
            file_path = self.personalities_dir / f"{name}{ext}"
            if file_path.exists():
                # Load and parse the file
                config_dict = self._load_config_file(file_path)
                
                # Ensure name is set
                if "name" not in config_dict:
                    config_dict["name"] = name
                    
                # Create the configuration
                personality = PersonalityConfig.from_dict(config_dict)
                
                # Cache and set as current
                self._personality_cache[name] = personality
                self._current_personality = personality
                return personality
                
        raise FileNotFoundError(f"Personality '{name}' not found in {self.personalities_dir}")
    
    def _load_config_file(self, file_path: Path) -> Dict:
        """
        Load a configuration file.
        
        Args:
            file_path: Path to the configuration file
            
        Returns:
            The parsed configuration as a dictionary
            
        Raises:
            ValueError: If the file format is unsupported or the content is invalid
        """
        ext = file_path.suffix.lower()
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if ext in (".yaml", ".yml"):
                    return yaml.safe_load(f)
                elif ext == ".json":
                    return json.load(f)
                else:
                    raise ValueError(f"Unsupported file format: {ext}")
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValueError(f"Error parsing {file_path}: {str(e)}")
    
    def get_prompt(self, template_name: str, **kwargs) -> str:
        """
        Get a prompt using the current personality.
        
        Args:
            template_name: Name of the template to use
            **kwargs: Data to inject into the template
            
        Returns:
            The filled template string
            
        Raises:
            RuntimeError: If no personality is loaded
        """
        if not self._current_personality:
            raise RuntimeError("No personality loaded. Call load_personality() first.")
            
        return self.template_manager.apply_template(
            template_name, 
            self._current_personality,
            **kwargs
        )
    
    def get_thinking_config(self) -> Dict:
        """
        Get the thinking configuration for the current personality.
        
        Returns:
            Dictionary with thinking configuration parameters
            
        Raises:
            RuntimeError: If no personality is loaded
        """
        if not self._current_personality:
            raise RuntimeError("No personality loaded. Call load_personality() first.")
            
        return self._current_personality.get_thinking_config()
    
    def get_trait(self, trait_name: str) -> Any:
        """
        Get a trait value from the current personality.
        
        Args:
            trait_name: Name of the trait
            
        Returns:
            The trait value
            
        Raises:
            RuntimeError: If no personality is loaded
            KeyError: If the trait doesn't exist
        """
        if not self._current_personality:
            raise RuntimeError("No personality loaded. Call load_personality() first.")
            
        # Check standard traits
        if trait_name == "reasoning_depth":
            return self._current_personality.reasoning_depth
        elif trait_name == "uncertainty_approach":
            return self._current_personality.uncertainty_approach
        elif trait_name == "thinking_style":
            return self._current_personality.thinking_style
        elif trait_name == "expert_persona":
            return self._current_personality.expert_persona
        elif trait_name == "temperature":
            return self._current_personality.temperature
            
        # Check custom traits
        elif trait_name in self._current_personality.traits:
            return self._current_personality.traits[trait_name].value
            
        raise KeyError(f"Trait '{trait_name}' not found in the current personality")
    
    def get_all_personalities(self) -> List[str]:
        """
        Get a list of all available personality names.
        
        Returns:
            List of personality names
        """
        personalities = []
        for ext in (".yaml", ".yml", ".json"):
            personalities.extend([
                f.stem for f in self.personalities_dir.glob(f"*{ext}")
            ])
        return sorted(personalities)
    
    def get_current_personality(self) -> Optional[PersonalityConfig]:
        """
        Get the current personality configuration.
        
        Returns:
            The current personality configuration, or None if none is loaded
        """
        return self._current_personality 