"""
Feature Flags for Personality Management System

This module provides feature flags to control the gradual rollout of the 
personality management system and enable A/B testing between legacy and 
new implementations.
"""

import os
import json
import logging
import random
from typing import Dict, Any, List, Optional, Set, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_FLAGS = {
    # Main system feature flags
    "personality_system_enabled": True,
    "use_caching": True,
    "use_telemetry": True,
    "debug_mode": False,
    
    # Gradual rollout percentages (0-100)
    "rollout_percentage": 100,
    "enable_for_users": "*",  # "*" means all users, or a list of user IDs
    "enable_for_questions": "*",  # "*" means all questions, or a list of question IDs/types
    
    # A/B testing
    "ab_testing_enabled": False,
    "ab_test_group_a_percentage": 50,  # Percentage of traffic in group A (legacy)
    "ab_test_metrics_collection": True,  # Collect metrics for A/B comparison
    
    # Feature-specific flags
    "advanced_templates_enabled": True,
    "custom_traits_enabled": True,
    "performance_monitoring_enabled": True,
    "auto_personality_detection_enabled": True,
    
    # Development flags
    "development_mode": False,
    "verbose_logging": False
}


class PersonalityFeatureFlags:
    """
    Feature flag manager for the personality system.
    
    This class loads feature flag configurations from environment variables
    or a configuration file and provides methods to check if features are enabled.
    """
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern for feature flags."""
        if cls._instance is None:
            cls._instance = super(PersonalityFeatureFlags, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Initialize feature flags."""
        self._flags = DEFAULT_FLAGS.copy()
        self._load_from_env()
        self._load_from_file()
        
        if self._flags.get("verbose_logging", False):
            logger.setLevel(logging.DEBUG)
            logger.debug("Verbose logging enabled for personality feature flags")
        
        # Setup random seed for consistent A/B testing per session
        random.seed(os.getpid())
        
        # Generate A/B test bucket once per instance
        self._ab_test_bucket = random.random() * 100
        
        logger.debug(f"Personality feature flags initialized: {self._flags}")
    
    def _load_from_env(self) -> None:
        """Load feature flags from environment variables."""
        prefix = "FORECASTING_PERSONALITY_"
        
        for key in DEFAULT_FLAGS.keys():
            env_key = f"{prefix}{key.upper()}"
            if env_key in os.environ:
                value = os.environ[env_key]
                
                # Convert string values to appropriate types
                if value.lower() in ("true", "yes", "1"):
                    value = True
                elif value.lower() in ("false", "no", "0"):
                    value = False
                elif value.isdigit():
                    value = int(value)
                
                self._flags[key] = value
                logger.debug(f"Loaded feature flag from environment: {key}={value}")
    
    def _load_from_file(self) -> None:
        """Load feature flags from configuration file."""
        # Check for config file path in environment
        config_path = os.environ.get(
            "FORECASTING_PERSONALITY_CONFIG", 
            "~/.forecasting-tools/personality_flags.json"
        )
        config_path = os.path.expanduser(config_path)
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                
                if isinstance(config, dict) and "feature_flags" in config:
                    for key, value in config["feature_flags"].items():
                        if key in DEFAULT_FLAGS:
                            self._flags[key] = value
                            logger.debug(f"Loaded feature flag from file: {key}={value}")
            except Exception as e:
                logger.warning(f"Failed to load feature flags from {config_path}: {str(e)}")
    
    def is_enabled(self, flag_name: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if a feature flag is enabled.
        
        Args:
            flag_name: Name of the feature flag
            context: Optional context dictionary with user_id, question_id, etc.
            
        Returns:
            True if the feature is enabled, False otherwise
        """
        # Check if flag exists
        if flag_name not in self._flags:
            logger.warning(f"Unknown feature flag: {flag_name}")
            return False
        
        # Get base flag value
        enabled = self._flags.get(flag_name, False)
        
        # If flag is disabled, no need to check further
        if not enabled:
            return False
        
        # Special handling for the main system toggle
        if flag_name != "personality_system_enabled" and not self.is_enabled("personality_system_enabled"):
            return False
        
        # Check rollout percentage (for main flag)
        if flag_name == "personality_system_enabled":
            # Check rollout percentage
            rollout_pct = self._flags.get("rollout_percentage", 100)
            if rollout_pct < 100:
                # Get a consistent random value for this user/question
                random_val = self._get_consistent_random(context)
                if random_val > rollout_pct / 100.0:
                    return False
            
            # Check user allowlist
            enable_for_users = self._flags.get("enable_for_users", "*")
            if enable_for_users != "*" and context and "user_id" in context:
                user_id = context["user_id"]
                if isinstance(enable_for_users, list) and user_id not in enable_for_users:
                    return False
            
            # Check question allowlist
            enable_for_questions = self._flags.get("enable_for_questions", "*")
            if enable_for_questions != "*" and context and "question_id" in context:
                question_id = context["question_id"]
                if isinstance(enable_for_questions, list) and question_id not in enable_for_questions:
                    return False
        
        # A/B testing for main system
        if flag_name == "personality_system_enabled" and self._flags.get("ab_testing_enabled", False):
            group_a_pct = self._flags.get("ab_test_group_a_percentage", 50)
            
            # If in group A (legacy system), disable new personality system
            if self._ab_test_bucket < group_a_pct:
                return False
        
        # Development mode overrides
        if self._flags.get("development_mode", False):
            # In development mode, enable all features
            return True
        
        return enabled
    
    def _get_consistent_random(self, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Get a consistent random value based on context.
        
        Args:
            context: Context dictionary with user_id, question_id, etc.
            
        Returns:
            Random value between 0.0 and 1.0
        """
        if not context:
            return self._ab_test_bucket / 100.0
        
        # Create a consistent seed based on user and question
        seed_parts = []
        if "user_id" in context:
            seed_parts.append(f"user:{context['user_id']}")
        if "question_id" in context:
            seed_parts.append(f"question:{context['question_id']}")
        
        if not seed_parts:
            return self._ab_test_bucket / 100.0
        
        # Generate a consistent random value
        seed = ":".join(seed_parts)
        random.seed(seed)
        result = random.random()
        # Reset seed to not affect other randomness
        random.seed()
        
        return result
    
    def get_all_flags(self) -> Dict[str, Any]:
        """
        Get all feature flags.
        
        Returns:
            Dictionary of all feature flags
        """
        return self._flags.copy()
    
    def update_flag(self, flag_name: str, value: Any) -> bool:
        """
        Update a feature flag value (runtime only, not persisted).
        
        Args:
            flag_name: Name of the feature flag
            value: New value for the flag
            
        Returns:
            True if updated successfully, False otherwise
        """
        if flag_name not in DEFAULT_FLAGS:
            logger.warning(f"Unknown feature flag: {flag_name}")
            return False
        
        self._flags[flag_name] = value
        logger.info(f"Feature flag updated: {flag_name}={value}")
        return True
    
    def save_configuration(self, file_path: Optional[str] = None) -> bool:
        """
        Save current feature flag configuration to a file.
        
        Args:
            file_path: Path to save the configuration (optional)
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not file_path:
            file_path = os.path.expanduser("~/.forecasting-tools/personality_flags.json")
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save configuration
            with open(file_path, "w") as f:
                json.dump({"feature_flags": self._flags}, f, indent=2)
            
            logger.info(f"Feature flags saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save feature flags to {file_path}: {str(e)}")
            return False


# Singleton instance for convenience
_flags = None

def get_feature_flags() -> PersonalityFeatureFlags:
    """Get the singleton feature flags instance."""
    global _flags
    if _flags is None:
        _flags = PersonalityFeatureFlags()
    return _flags 