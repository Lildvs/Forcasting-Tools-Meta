"""
Configuration management for the forecasting tools.

This module provides a centralized configuration management system with
validation, default values, and integration with Streamlit's secrets.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast

import streamlit as st
from pydantic import BaseModel, Field, ValidationError

T = TypeVar('T')

# Configure logger
logger = logging.getLogger(__name__)


class APIKeysConfig(BaseModel):
    """API keys configuration model."""
    openai_api_key: str = Field(default="")
    openai_org_id: str = Field(default="")
    anthropic_api_key: str = Field(default="")
    google_api_key: str = Field(default="")
    serper_api_key: str = Field(default="")


class DatabaseConfig(BaseModel):
    """Database configuration model."""
    connection_string: str = Field(default="sqlite:///./data/forecast_costs.db")
    username: str = Field(default="")
    password: str = Field(default="")


class ServicesConfig(BaseModel):
    """External services configuration model."""
    error_reporting: bool = Field(default=False)
    monitoring_endpoint: str = Field(default="")
    sentry_dsn: str = Field(default="")
    cloudwatch_group: str = Field(default="")
    cloudwatch_stream: str = Field(default="")


class FeaturesConfig(BaseModel):
    """Feature flags configuration model."""
    enable_web_search: bool = Field(default=True)
    enable_research_tools: bool = Field(default=True)
    enable_benchmark_tools: bool = Field(default=False)
    enable_health_check: bool = Field(default=True)
    enable_error_reporting: bool = Field(default=False)


class DeploymentConfig(BaseModel):
    """Deployment configuration model."""
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    max_history_size: int = Field(default=100)


class ErrorHandlingConfig(BaseModel):
    """Error handling configuration model."""
    show_error_details: bool = Field(default=True)
    retry_api_calls: bool = Field(default=True)
    max_retries: int = Field(default=3)
    enable_error_boundaries: bool = Field(default=True)


class AppConfig(BaseModel):
    """Complete application configuration model."""
    api_keys: APIKeysConfig = Field(default_factory=APIKeysConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    services: ServicesConfig = Field(default_factory=ServicesConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    error_handling: ErrorHandlingConfig = Field(default_factory=ErrorHandlingConfig)


class ConfigManager:
    """
    Configuration management for the forecasting tools.
    
    Provides a centralized configuration management system with
    validation, default values, and integration with Streamlit's secrets.
    """
    
    _instance = None  # Singleton pattern
    
    @classmethod
    def get_instance(cls) -> 'ConfigManager':
        """Get singleton instance of ConfigManager."""
        if cls._instance is None:
            cls._instance = ConfigManager()
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize configuration manager."""
        self._config = self._load_config()
        self._validate_critical_config()
        logger.info("Configuration manager initialized")
    
    def _load_config(self) -> AppConfig:
        """
        Load configuration from multiple sources in priority order:
        1. Environment variables
        2. Streamlit secrets
        3. Default values
        """
        # Start with default config
        config_dict = {}
        
        # Load from Streamlit secrets if available
        if hasattr(st, "secrets"):
            for section in ["api_keys", "database", "services", "features", "deployment", "error_handling"]:
                if section in st.secrets:
                    if section not in config_dict:
                        config_dict[section] = {}
                    for key, value in st.secrets[section].items():
                        config_dict[section][key] = value
        
        # Override with environment variables
        # Format: FORECASTING_SECTION_KEY=value
        for env_key, env_value in os.environ.items():
            if env_key.startswith("FORECASTING_"):
                parts = env_key.split("_", 2)
                if len(parts) == 3:
                    _, section, key = parts
                    section = section.lower()
                    key = key.lower()
                    
                    if section not in config_dict:
                        config_dict[section] = {}
                    
                    # Convert string values to appropriate types
                    if env_value.lower() == "true":
                        config_dict[section][key] = True
                    elif env_value.lower() == "false":
                        config_dict[section][key] = False
                    elif env_value.isdigit():
                        config_dict[section][key] = int(env_value)
                    else:
                        config_dict[section][key] = env_value
        
        # Create AppConfig from dictionary
        try:
            return AppConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            # Fall back to default config if validation fails
            return AppConfig()
    
    def _validate_critical_config(self) -> None:
        """Validate critical configuration values and log warnings."""
        # Check API keys
        if self.is_feature_enabled("enable_web_search"):
            if not (self.get_api_key("google_api_key") or self.get_api_key("serper_api_key")):
                logger.warning("Web search is enabled but no search API keys are configured")
        
        # Check API keys based on features
        llm_api_keys = [self.get_api_key("openai_api_key"), self.get_api_key("anthropic_api_key")]
        if not any(llm_api_keys):
            logger.warning("No LLM API keys configured, some features may not work")
        
        # Check error reporting configuration
        if self.is_feature_enabled("enable_error_reporting"):
            if not self.get_service_config("sentry_dsn"):
                logger.warning("Error reporting is enabled but Sentry DSN is not configured")
        
        # Validate database connection string
        db_conn = self.get_database_config("connection_string")
        if not db_conn or db_conn == "":
            logger.warning("Database connection string is not configured")
    
    def get_api_key(self, key_name: str) -> str:
        """Get API key by name."""
        return getattr(self._config.api_keys, key_name, "")
    
    def get_database_config(self, key_name: str) -> str:
        """Get database configuration value by name."""
        return getattr(self._config.database, key_name, "")
    
    def get_service_config(self, key_name: str) -> Union[str, bool]:
        """Get service configuration value by name."""
        return getattr(self._config.services, key_name, "")
    
    def get_deployment_environment(self) -> str:
        """Get the current deployment environment."""
        return self._config.deployment.environment
    
    def is_feature_enabled(self, feature_name: str, default: bool = False) -> bool:
        """Check if a feature is enabled."""
        return getattr(self._config.features, feature_name, default)
    
    def get_error_handling_config(self, key_name: str) -> Union[bool, int]:
        """Get error handling configuration value by name."""
        return getattr(self._config.error_handling, key_name, None)
    
    def get_config_section(self, section_name: str) -> Optional[BaseModel]:
        """Get an entire configuration section."""
        return getattr(self._config, section_name, None)
    
    def reload(self) -> None:
        """Reload configuration from sources."""
        self._config = self._load_config()
        self._validate_critical_config()
        logger.info("Configuration reloaded")


# Create a global instance for easy importing
config = ConfigManager.get_instance() 