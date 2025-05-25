"""
Secrets management for the forecasting tools.

This module provides backward compatibility with the old SecretsManager
while delegating to the new ConfigManager class.
"""

import logging
import os
from typing import Any, Dict, Optional, Union

import streamlit as st

from forecasting_tools.util.config_manager import config

# Configure logger
logger = logging.getLogger(__name__)


class SecretsManager:
    """
    Legacy secrets manager class that delegates to ConfigManager.
    
    This class provides backward compatibility with code that uses
    the old SecretsManager interface.
    """
    
    @classmethod
    def get_secret(cls, section: str, key: str, default: Any = None) -> Any:
        """
        Get a secret value from the configuration.
        
        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if not found
            
        Returns:
            The configuration value or default
        """
        if section == "api_keys":
            return config.get_api_key(key) or default
        elif section == "database":
            return config.get_database_config(key) or default
        elif section == "services":
            return config.get_service_config(key) or default
        elif section == "features":
            feature_name = f"enable_{key}" if not key.startswith("enable_") else key
            return config.is_feature_enabled(feature_name, default)
        elif section == "deployment":
            if key == "environment":
                return config.get_deployment_environment()
            return default
        elif section == "error_handling":
            return config.get_error_handling_config(key) or default
        else:
            return default
    
    @classmethod
    def get_deployment_environment(cls) -> str:
        """
        Get the current deployment environment.
        
        Returns:
            Environment name (development, staging, production)
        """
        return config.get_deployment_environment()
    
    @classmethod
    def is_feature_enabled(cls, feature_name: str, default: bool = False) -> bool:
        """
        Check if a feature is enabled.
        
        Args:
            feature_name: Feature name to check
            default: Default value if not found
            
        Returns:
            True if feature is enabled, False otherwise
        """
        # Handle both with and without the enable_ prefix
        if not feature_name.startswith("enable_"):
            feature_name = f"enable_{feature_name}"
        
        return config.is_feature_enabled(feature_name, default)

    @staticmethod
    def get_api_key(service: str, default: str = "") -> str:
        """
        Get an API key for a specific service.
        
        Args:
            service: Service name (e.g., 'openai', 'anthropic')
            default: Default value if the key is not found
            
        Returns:
            The API key or empty string if not found
        """
        return SecretsManager.get_secret("api_keys", f"{service}_api_key", default)
    
    @staticmethod
    def get_database_config() -> Dict[str, str]:
        """
        Get database configuration parameters.
        
        Returns:
            Dictionary with database configuration
        """
        return {
            "connection_string": SecretsManager.get_secret("database", "connection_string", ""),
            "username": SecretsManager.get_secret("database", "username", ""),
            "password": SecretsManager.get_secret("database", "password", "")
        }
    
    @staticmethod
    def initialize() -> None:
        """
        Initialize the secrets manager and validate required secrets.
        
        Should be called early in application startup.
        """
        # Log information about the environment
        environment = SecretsManager.get_deployment_environment()
        logging.info(f"Initializing secrets manager in {environment} environment")
        
        # Check for critical API keys
        openai_key = SecretsManager.get_api_key("openai")
        if not openai_key:
            logging.warning("OpenAI API key not found. Some features may be unavailable.")
        
        # Check for database configuration
        db_config = SecretsManager.get_database_config()
        if not db_config["connection_string"]:
            logging.warning("Database connection string not found. Using default settings.") 