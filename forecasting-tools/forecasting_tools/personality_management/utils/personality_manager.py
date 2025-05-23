import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class PersonalityManager:
    """
    Manages personality configurations for forecasting bots.
    This class loads personality configurations from YAML files and provides
    methods to retrieve and apply prompts based on the selected personality.
    """
    
    # Base directory for personality configurations
    BASE_DIR = Path(__file__).parent.parent
    PERSONALITIES_DIR = BASE_DIR / "personalities"
    TEMPLATES_DIR = BASE_DIR / "templates"
    
    # Default personality to use if none is specified
    DEFAULT_PERSONALITY = "balanced"
    
    def __init__(self, personality_name: Optional[str] = None):
        """
        Initialize the PersonalityManager with a specific personality.
        
        Args:
            personality_name: Name of the personality to use. If None, uses the default.
        """
        self.personality_name = personality_name or self.DEFAULT_PERSONALITY
        self.personality_config = self._load_personality(self.personality_name)
        self.templates = self._load_templates()
    
    def _load_personality(self, personality_name: str) -> Dict[str, Any]:
        """
        Load a personality configuration from YAML file.
        
        Args:
            personality_name: Name of the personality to load
            
        Returns:
            Dictionary containing the personality configuration
        """
        file_path = self.PERSONALITIES_DIR / f"{personality_name}.yaml"
        try:
            with open(file_path, "r") as file:
                config = yaml.safe_load(file)
                logger.info(f"Loaded personality: {personality_name}")
                return config
        except FileNotFoundError:
            logger.warning(f"Personality {personality_name} not found, using default")
            # If specified personality not found, fall back to default
            if personality_name != self.DEFAULT_PERSONALITY:
                return self._load_personality(self.DEFAULT_PERSONALITY)
            # If default also not found, return empty config
            return {}
    
    def _load_templates(self) -> Dict[str, str]:
        """
        Load all template files from the templates directory.
        
        Returns:
            Dictionary mapping template names to their content
        """
        templates = {}
        for template_file in self.TEMPLATES_DIR.glob("*.txt"):
            template_name = template_file.stem
            with open(template_file, "r") as file:
                templates[template_name] = file.read()
        return templates
    
    def get_prompt(self, prompt_key: str, **kwargs) -> str:
        """
        Get a prompt template and fill it with personality-specific values.
        
        Args:
            prompt_key: The key for the prompt template
            **kwargs: Additional parameters to format the template with
            
        Returns:
            Formatted prompt string
        """
        # Get the base template
        if prompt_key not in self.templates:
            logger.warning(f"Template {prompt_key} not found")
            return ""
        
        template = self.templates[prompt_key]
        
        # Get personality-specific values for this prompt
        personality_values = self.personality_config.get("prompts", {}).get(prompt_key, {})
        
        # Combine kwargs with personality values (kwargs take precedence)
        format_values = {**personality_values, **kwargs}
        
        # Format the template with the combined values
        try:
            return template.format(**format_values)
        except KeyError as e:
            logger.error(f"Missing key in template formatting: {e}")
            return template
    
    def get_thinking_config(self) -> Dict[str, Any]:
        """
        Get the thinking configuration for the current personality.
        
        Returns:
            Dictionary with thinking configuration parameters
        """
        return self.personality_config.get("thinking", {})
    
    def get_all_personalities(self) -> List[str]:
        """
        Get a list of all available personalities.
        
        Returns:
            List of personality names
        """
        return [f.stem for f in self.PERSONALITIES_DIR.glob("*.yaml")]
    
    def get_personality_trait(self, trait_key: str, default: Any = None) -> Any:
        """
        Get a specific trait value from the personality configuration.
        
        Args:
            trait_key: The key for the trait
            default: Default value if trait is not found
            
        Returns:
            The trait value or default
        """
        return self.personality_config.get("traits", {}).get(trait_key, default) 