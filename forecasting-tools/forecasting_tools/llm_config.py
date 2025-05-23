"""
LLM Configuration

This module provides configuration for LLM models used throughout the application.
"""

import os
import logging
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class LLMConfigManager:
    """
    Manager for LLM configuration settings.
    
    This class provides a centralized way to configure and access LLM settings
    for different purposes (researcher, summarizer, default, etc.)
    """
    
    @staticmethod
    def get_default_config() -> Dict[str, Union[str, Dict[str, Any]]]:
        """
        Get the default LLM configuration.
        
        Returns:
            Dictionary with LLM configurations
        """
        # Default configuration
        config = {
            "default": {
                "model": "gpt-4.1",  # Latest GPT-4 version
                "temperature": 0.3,
                "max_tokens": 4000
            },
            "researcher": {
                "model": "gpt-4o",  # More cost-effective for research
                "temperature": 0.1,
                "max_tokens": 4000
            },
            "summarizer": {
                "model": "gpt-4o-mini",  # Efficient for summarization
                "temperature": 0.1,
                "max_tokens": 2000
            }
        }
        
        # Override with environment variables if specified
        if os.getenv("DEFAULT_LLM_MODEL"):
            config["default"]["model"] = os.getenv("DEFAULT_LLM_MODEL")
            
        if os.getenv("RESEARCHER_LLM_MODEL"):
            config["researcher"]["model"] = os.getenv("RESEARCHER_LLM_MODEL")
            
        if os.getenv("SUMMARIZER_LLM_MODEL"):
            config["summarizer"]["model"] = os.getenv("SUMMARIZER_LLM_MODEL")
        
        return config
    
    @staticmethod
    def get_model_for_purpose(purpose: str) -> str:
        """
        Get the model name for a specific purpose.
        
        Args:
            purpose: The purpose (default, researcher, summarizer)
            
        Returns:
            Model name
        """
        config = LLMConfigManager.get_default_config()
        if purpose in config:
            return config[purpose]["model"]
        else:
            logger.warning(f"User forgot to set an llm for purpose: '{purpose}'. Using default llm: 'openai/gpt-4.1'")
            return "openai/gpt-4.1"
            
    @staticmethod
    def get_settings_for_purpose(purpose: str) -> Dict[str, Any]:
        """
        Get all settings for a specific purpose.
        
        Args:
            purpose: The purpose (default, researcher, summarizer)
            
        Returns:
            Dictionary with all settings
        """
        config = LLMConfigManager.get_default_config()
        if purpose in config:
            return config[purpose]
        else:
            logger.warning(f"No config found for purpose: '{purpose}'. Using default config.")
            return config["default"] 