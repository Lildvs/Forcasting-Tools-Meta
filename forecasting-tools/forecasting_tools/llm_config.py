"""
LLM Configuration

This module provides configuration for LLM models used throughout the application.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List

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
                "model": "gpt-4.1",  # Latest GPT-4 version for primary reasoning
                "temperature": 0.3,
                "max_tokens": 4000,
                "description": "Primary reasoning layer for initial query understanding and final forecast generation"
            },
            "researcher": {
                "model": "perplexity/sonar-medium-online",  # Research-focused model with web access
                "temperature": 0.1,
                "max_tokens": 4000,
                "description": "Specialized research layer for gathering detailed information from multiple sources",
                "fallbacks": ["gpt-4o", "perplexity/sonar-small-online"]
            },
            "summarizer": {
                "model": "gpt-4o-mini",  # Efficient for summarization
                "temperature": 0.1,
                "max_tokens": 2000,
                "description": "Lightweight model for condensing research into key insights"
            },
            "evaluator": {
                "model": "gpt-4o",  # Reliable model for evaluating forecasts
                "temperature": 0.2,
                "max_tokens": 3000,
                "description": "Specialized layer for evaluating forecast quality and calibration"
            }
        }
        
        # Override with environment variables if specified
        if os.getenv("DEFAULT_LLM_MODEL"):
            config["default"]["model"] = os.getenv("DEFAULT_LLM_MODEL")
            
        if os.getenv("RESEARCHER_LLM_MODEL"):
            config["researcher"]["model"] = os.getenv("RESEARCHER_LLM_MODEL")
            
        if os.getenv("SUMMARIZER_LLM_MODEL"):
            config["summarizer"]["model"] = os.getenv("SUMMARIZER_LLM_MODEL")
            
        if os.getenv("EVALUATOR_LLM_MODEL"):
            config["evaluator"]["model"] = os.getenv("EVALUATOR_LLM_MODEL")
        
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
    
    @staticmethod
    def get_fallback_models(purpose: str) -> List[str]:
        """
        Get fallback models for a specific purpose if the primary model fails.
        
        Args:
            purpose: The purpose (default, researcher, summarizer)
            
        Returns:
            List of fallback model names
        """
        config = LLMConfigManager.get_default_config()
        if purpose in config and "fallbacks" in config[purpose]:
            return config[purpose]["fallbacks"]
        return []
    
    @staticmethod 
    def get_workflow_description() -> str:
        """
        Returns a description of the multi-layered LLM workflow.
        
        Returns:
            String describing the workflow
        """
        return """
        The forecasting system uses a multi-layered LLM approach:
        
        1. Base Layer (GPT-4.1): 
           - Handles initial query understanding
           - Formulates research strategy
           - Performs initial reasoning
        
        2. Researcher Layer (Perplexity):
           - Conducts extensive web research
           - Gathers up-to-date information
           - Compiles relevant data from multiple sources
        
        3. Summarizer Layer (GPT-4o-mini):
           - Condenses research into key insights
           - Extracts most relevant information
           - Creates structured summaries
        
        4. Final Base Layer (GPT-4.1 again):
           - Takes all processed information
           - Performs final reasoning and computation
           - Generates the final forecast with confidence levels
        """ 