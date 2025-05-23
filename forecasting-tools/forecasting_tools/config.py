"""
Configuration Settings

This module provides configuration settings for the forecasting tools, including
API credentials, search preferences, system-wide settings, and environment-specific
configuration.
"""

import os
from typing import Dict, List, Optional, Literal, Any
from pathlib import Path
import logging
import json
from enum import Enum

# Environment type
class EnvironmentType(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

# Get current environment
ENVIRONMENT = os.getenv("FORECASTING_ENVIRONMENT", EnvironmentType.DEVELOPMENT)

# Database Configuration
class DatabaseConfig:
    """
    Configuration settings for database connections.
    """
    
    # Database connection strings by environment
    CONNECTION_STRINGS = {
        EnvironmentType.DEVELOPMENT: os.getenv("DEV_DB_CONNECTION", "sqlite:///data/forecasting_dev.db"),
        EnvironmentType.STAGING: os.getenv("STAGING_DB_CONNECTION", "sqlite:///data/forecasting_staging.db"),
        EnvironmentType.PRODUCTION: os.getenv("PROD_DB_CONNECTION", "postgresql://user:password@localhost:5432/forecasting"),
        EnvironmentType.TESTING: "sqlite:///:memory:",
    }
    
    # Get current connection string
    CONNECTION_STRING = CONNECTION_STRINGS.get(ENVIRONMENT, CONNECTION_STRINGS[EnvironmentType.DEVELOPMENT])
    
    # Connection pool settings
    POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
    MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "300"))
    
    # Query settings
    QUERY_TIMEOUT = int(os.getenv("DB_QUERY_TIMEOUT", "60"))
    
    # Debug flags
    ECHO_SQL = os.getenv("DB_ECHO_SQL", "False").lower() == "true"


# Search Provider Configuration
class SearchConfig:
    """
    Configuration settings for search providers.
    """
    
    # API Keys
    PERPLEXITY_API_KEY: str = os.getenv("PERPLEXITY_API_KEY", "")
    EXA_API_KEY: str = os.getenv("EXA_API_KEY", "")
    CRAWL4AI_API_KEY: str = os.getenv("CRAWL4AI_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    
    # Default Provider (auto, perplexity, smart, crawl4ai)
    DEFAULT_SEARCH_PROVIDER: str = os.getenv("DEFAULT_SEARCH_PROVIDER", "auto")
    
    # Search Settings
    DEFAULT_SEARCH_TYPE: Literal["basic", "deep"] = os.getenv("DEFAULT_SEARCH_TYPE", "basic")
    DEFAULT_SEARCH_DEPTH: Literal["low", "medium", "high"] = os.getenv("DEFAULT_SEARCH_DEPTH", "medium")
    
    # Rate Limiting
    SEARCH_RATE_LIMIT: int = int(os.getenv("SEARCH_RATE_LIMIT", "10"))  # requests per minute
    SEARCH_TIMEOUT: int = int(os.getenv("SEARCH_TIMEOUT", "60"))  # seconds
    
    # Caching
    ENABLE_SEARCH_CACHE: bool = os.getenv("ENABLE_SEARCH_CACHE", "True").lower() == "true"
    SEARCH_CACHE_TTL: int = int(os.getenv("SEARCH_CACHE_TTL", "86400"))  # 24 hours in seconds
    SEARCH_CACHE_DIR: Optional[Path] = None  # Default is ~/.forecasting_tools/cache
    
    # Fallback Settings
    SEARCH_FALLBACK_ORDER: List[str] = ["perplexity", "smart", "crawl4ai"]
    
    # Provider-specific settings
    PERPLEXITY_SETTINGS: Dict[str, Any] = {
        "timeout": 120,
    }
    
    SMART_SEARCHER_SETTINGS: Dict[str, Any] = {
        "include_works_cited_list": True,
        "use_brackets_around_citations": True,
        "num_searches_to_run": 2,
        "num_sites_per_search": 10,
        "model": "gpt-4o",
        "use_advanced_filters": True,
    }
    
    CRAWL4AI_SETTINGS: Dict[str, Any] = {
        "timeout": 120,
        "max_pages": 15,
        "crawl_depth": 2,
        "follow_links": True,
        "synthesis_model": "gpt-4o",
    }


# LLM Configuration
class LLMConfig:
    """
    Configuration settings for large language models.
    """
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_ORG_ID: str = os.getenv("OPENAI_ORG_ID", "")
    
    # Anthropic
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Metaculus
    METACULUS_TOKEN: str = os.getenv("METACULUS_TOKEN", "")
    
    # Default Models
    DEFAULT_LLM: str = os.getenv("DEFAULT_LLM", "gpt-4o")
    RESEARCH_SUMMARIZER_LLM: str = os.getenv("RESEARCH_SUMMARIZER_LLM", "gpt-4o-mini")
    REASONING_LLM: str = os.getenv("REASONING_LLM", "gpt-4o")
    
    # Model Settings
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.3"))
    RESEARCH_TEMPERATURE: float = float(os.getenv("RESEARCH_TEMPERATURE", "0.1"))
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE: int = int(os.getenv("MAX_LLM_REQUESTS_PER_MINUTE", "50"))
    RETRY_ATTEMPTS: int = int(os.getenv("LLM_RETRY_ATTEMPTS", "3"))
    REQUEST_TIMEOUT: int = int(os.getenv("LLM_REQUEST_TIMEOUT", "120"))


# Cache Configuration
class CacheConfig:
    """
    Configuration settings for caching.
    """
    
    # Cache storage options: "memory", "disk", "redis"
    CACHE_STORAGE: str = os.getenv("CACHE_STORAGE", "disk")
    
    # Cache directory (for disk cache)
    CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", "~/.forecasting_tools/cache")).expanduser()
    
    # Redis cache config (for redis cache)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # TTL Settings
    DEFAULT_TTL: int = int(os.getenv("DEFAULT_TTL", "86400"))  # 24 hours in seconds
    RESEARCH_CACHE_TTL: int = int(os.getenv("RESEARCH_CACHE_TTL", "86400"))  # 24 hours in seconds
    FORECAST_CACHE_TTL: int = int(os.getenv("FORECAST_CACHE_TTL", "3600"))  # 1 hour in seconds
    
    # Cache Control
    ENABLE_PERSISTENCE: bool = os.getenv("ENABLE_CACHE_PERSISTENCE", "True").lower() == "true"
    CLEANUP_INTERVAL: int = int(os.getenv("CACHE_CLEANUP_INTERVAL", "3600"))  # 1 hour in seconds
    
    # Cache strategy
    # - "simple" - Simple key-value caching
    # - "tiered" - Tiered caching (memory -> disk/redis)
    # - "distributed" - Distributed caching (requires redis)
    CACHE_STRATEGY: str = os.getenv("CACHE_STRATEGY", "simple")
    
    # Cache size limits
    MEMORY_CACHE_MAX_ITEMS: int = int(os.getenv("MEMORY_CACHE_MAX_ITEMS", "1000"))
    DISK_CACHE_MAX_SIZE_MB: int = int(os.getenv("DISK_CACHE_MAX_SIZE_MB", "500"))
    
    # Initialize Cache Directory
    if ENABLE_PERSISTENCE and CACHE_STORAGE == "disk":
        CACHE_DIR.mkdir(parents=True, exist_ok=True)


# System Configuration
class SystemConfig:
    """
    System-wide configuration settings.
    """
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    
    # Concurrency
    MAX_CONCURRENT_QUESTIONS: int = int(os.getenv("MAX_CONCURRENT_QUESTIONS", "3"))
    MAX_CONCURRENT_SEARCHES: int = int(os.getenv("MAX_CONCURRENT_SEARCHES", "5"))
    JOB_QUEUE_WORKERS: int = int(os.getenv("JOB_QUEUE_WORKERS", "4"))
    
    # Error Notification
    ENABLE_ERROR_NOTIFICATIONS: bool = os.getenv("ENABLE_ERROR_NOTIFICATIONS", "False").lower() == "true"
    ERROR_NOTIFICATION_EMAIL: Optional[str] = os.getenv("ERROR_NOTIFICATION_EMAIL")
    
    # Default Paths
    FORECAST_REPORTS_DIR: Path = Path(os.getenv("FORECAST_REPORTS_DIR", "~/forecasts")).expanduser()
    
    # Application metrics
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "False").lower() == "true"
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", "9090"))
    
    # Security settings
    API_KEY_REQUIRED: bool = os.getenv("API_KEY_REQUIRED", "False").lower() == "true"
    JWT_SECRET: str = os.getenv("JWT_SECRET", "")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRATION_MINUTES: int = int(os.getenv("JWT_EXPIRATION_MINUTES", "60"))
    
    # Initialize Forecast Reports Directory
    FORECAST_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# Personality Configuration
class PersonalityConfig:
    """
    Configuration settings for personality system.
    """
    
    # Default Personality
    DEFAULT_PERSONALITY: str = os.getenv("DEFAULT_PERSONALITY", "balanced")
    
    # Personality Templates Directory
    PERSONALITY_TEMPLATES_DIR: Path = Path(__file__).parent / "personality_templates"


# Async execution configuration
class AsyncConfig:
    """
    Configuration for asynchronous processing.
    """
    
    # Thread pool
    THREAD_POOL_SIZE: int = int(os.getenv("THREAD_POOL_SIZE", "10"))
    
    # Process pool
    PROCESS_POOL_SIZE: int = int(os.getenv("PROCESS_POOL_SIZE", "4"))
    
    # Job queue settings
    JOB_QUEUE_MAX_SIZE: int = int(os.getenv("JOB_QUEUE_MAX_SIZE", "1000"))
    JOB_DEFAULT_TIMEOUT: int = int(os.getenv("JOB_DEFAULT_TIMEOUT", "300"))
    JOB_MAX_RETRIES: int = int(os.getenv("JOB_MAX_RETRIES", "3"))
    JOB_RETRY_DELAY: int = int(os.getenv("JOB_RETRY_DELAY", "5"))


# Load environment-specific config overrides
def load_env_config():
    """Load environment-specific configuration from JSON file."""
    config_dir = Path(__file__).parent / "config"
    config_file = config_dir / f"{ENVIRONMENT}.json"
    
    if not config_dir.exists():
        config_dir.mkdir(parents=True, exist_ok=True)
    
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                env_config = json.load(f)
            
            # Update environment variables with config file values
            for key, value in env_config.items():
                if isinstance(value, dict):
                    # Handle nested config sections
                    for sub_key, sub_value in value.items():
                        env_var = f"{key}_{sub_key}".upper()
                        if os.getenv(env_var) is None:  # Don't override existing env vars
                            os.environ[env_var] = str(sub_value)
                else:
                    # Handle top-level config values
                    if os.getenv(key) is None:  # Don't override existing env vars
                        os.environ[key] = str(value)
                
            logging.info(f"Loaded configuration for environment: {ENVIRONMENT}")
            
        except Exception as e:
            logging.error(f"Error loading environment config: {str(e)}")
    else:
        # Create default config file for the environment if it doesn't exist
        if ENVIRONMENT != EnvironmentType.TESTING:
            default_config = {
                "database": {
                    "connection_string": DatabaseConfig.CONNECTION_STRING,
                    "pool_size": DatabaseConfig.POOL_SIZE,
                    "max_overflow": DatabaseConfig.MAX_OVERFLOW,
                },
                "cache": {
                    "storage": CacheConfig.CACHE_STORAGE,
                    "strategy": CacheConfig.CACHE_STRATEGY,
                    "enable_persistence": CacheConfig.ENABLE_PERSISTENCE,
                },
                "system": {
                    "log_level": SystemConfig.LOG_LEVEL,
                    "job_queue_workers": SystemConfig.JOB_QUEUE_WORKERS,
                    "enable_metrics": SystemConfig.ENABLE_METRICS,
                },
                "llm": {
                    "default_llm": LLMConfig.DEFAULT_LLM,
                    "default_temperature": LLMConfig.DEFAULT_TEMPERATURE,
                }
            }
            
            try:
                with open(config_file, "w") as f:
                    json.dump(default_config, f, indent=2)
                
                logging.info(f"Created default configuration for environment: {ENVIRONMENT}")
                
            except Exception as e:
                logging.error(f"Error creating default environment config: {str(e)}")


# Initialize logging configuration
def configure_logging():
    """Configure application logging based on settings."""
    log_level = getattr(logging, SystemConfig.LOG_LEVEL.upper(), logging.INFO)
    
    handlers = []
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(SystemConfig.LOG_FORMAT))
    handlers.append(console_handler)
    
    # Create file handler if log file specified
    if SystemConfig.LOG_FILE:
        log_dir = os.path.dirname(SystemConfig.LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(SystemConfig.LOG_FILE)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(SystemConfig.LOG_FORMAT))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=SystemConfig.LOG_FORMAT,
        handlers=handlers
    )
    
    # Set specific loggers to different levels if needed
    if ENVIRONMENT == EnvironmentType.DEVELOPMENT:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO if DatabaseConfig.ECHO_SQL else logging.WARNING)
    else:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


# Load environment-specific config and configure logging on module import
load_env_config()
configure_logging() 