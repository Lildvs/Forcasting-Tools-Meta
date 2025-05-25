"""
Error handling utilities for robust cloud deployment.

This module provides a comprehensive error handling system designed for
cloud deployments, including structured logging, error boundaries, and
integration with monitoring systems.
"""

import functools
import inspect
import json
import logging
import os
import sys
import traceback
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

import streamlit as st

from forecasting_tools.util.config_manager import config

# Type variables for function typing
F = TypeVar('F', bound=Callable[..., Any])
R = TypeVar('R')

# Configure logger
logger = logging.getLogger(__name__)


class ErrorReporter:
    """
    Error reporting service integration.
    
    Provides integration with external error tracking services while
    maintaining compatibility with local development.
    """
    
    _instance = None  # Singleton pattern
    
    # Error tracking integrations
    SENTRY_ENABLED = False
    CLOUDWATCH_ENABLED = False
    
    @classmethod
    def get_instance(cls) -> 'ErrorReporter':
        """Get singleton instance of ErrorReporter."""
        if cls._instance is None:
            cls._instance = ErrorReporter()
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize error reporting services based on configuration."""
        # Initialize error reporting services if enabled
        self.environment = config.get_deployment_environment()
        self.service_enabled = config.is_feature_enabled("enable_error_reporting", False)
        
        # Initialize Sentry if available and enabled
        if self.service_enabled:
            sentry_dsn = config.get_service_config("sentry_dsn")
            if sentry_dsn:
                try:
                    import sentry_sdk
                    from sentry_sdk.integrations.logging import LoggingIntegration
                    
                    # Configure Sentry with proper environment and release info
                    sentry_sdk.init(
                        dsn=sentry_dsn,
                        environment=self.environment,
                        traces_sample_rate=0.1,
                        release=self._get_release_version(),
                        integrations=[
                            LoggingIntegration(
                                level=logging.INFO,
                                event_level=logging.ERROR
                            )
                        ]
                    )
                    self.SENTRY_ENABLED = True
                    logger.info("Sentry error reporting initialized")
                except ImportError:
                    logger.warning("Sentry SDK not installed, error reporting disabled")
                except Exception as e:
                    logger.error(f"Failed to initialize Sentry: {e}")
    
    def _get_release_version(self) -> str:
        """Get the current application release version."""
        try:
            import importlib.metadata
            version = importlib.metadata.version("forecasting-tools")
            return version
        except (ImportError, importlib.metadata.PackageNotFoundError):
            # Try to get version from version.txt if it exists
            version_file = os.path.join(os.getcwd(), "version.txt")
            if os.path.exists(version_file):
                with open(version_file, "r") as f:
                    version_data = f.read().strip()
                    for line in version_data.split("\n"):
                        if line.startswith("version:"):
                            return line.split(":", 1)[1].strip()
            
            # Return a default version
            return "0.0.0-unknown"
    
    def report(self, error_info: Dict[str, Any]) -> str:
        """
        Report an error to configured error tracking services.
        
        Args:
            error_info: Dictionary with error information
            
        Returns:
            Error ID for reference
        """
        # Generate unique error ID
        error_id = str(uuid.uuid4())
        error_info["error_id"] = error_id
        
        # Add timestamp and environment info
        error_info["timestamp"] = datetime.now().isoformat()
        error_info["environment"] = self.environment
        
        # Log the error locally
        logger.error(f"Error {error_id}: {json.dumps(error_info)}")
        
        # Report to Sentry if enabled
        if self.SENTRY_ENABLED:
            try:
                import sentry_sdk
                
                # Extract the original exception if available
                exception = error_info.get("exception")
                
                with sentry_sdk.push_scope() as scope:
                    # Add context information
                    for key, value in error_info.items():
                        if key not in ["exception", "traceback"]:
                            scope.set_extra(key, value)
                    
                    # Set user context if available
                    user_info = error_info.get("user_info", {})
                    if user_info:
                        scope.set_user(user_info)
                    
                    # Capture the exception or message
                    if exception:
                        sentry_sdk.capture_exception(exception)
                    else:
                        sentry_sdk.capture_message(
                            error_info.get("error_message", "Unknown error"),
                            level="error"
                        )
            except Exception as e:
                logger.error(f"Failed to report error to Sentry: {e}")
        
        return error_id


def global_exception_handler(func: F) -> F:
    """
    Decorator to add global exception handling to a function.
    
    Args:
        func: Function to wrap with exception handling
        
    Returns:
        Wrapped function with exception handling
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get error details
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "exception": e,
                "function": func.__name__,
                "module": func.__module__,
                "args": str(args),
                "kwargs": str(kwargs),
            }
            
            # Add session info if using Streamlit
            if "st" in sys.modules and hasattr(st, "session_state"):
                # Don't include potential sensitive data
                safe_session_keys = [
                    k for k in st.session_state.keys() 
                    if not any(sensitive in k for sensitive in ["key", "token", "password", "secret"])
                ]
                error_details["session_info"] = {
                    k: str(st.session_state[k]) for k in safe_session_keys
                }
            
            # Log detailed error
            logger.error(
                f"Exception in {func.__name__}: {e}\n{traceback.format_exc()}"
            )
            
            # Report to error tracking service
            try:
                error_reporter = ErrorReporter.get_instance()
                error_id = error_reporter.report(error_details)
                error_details["error_id"] = error_id
            except Exception as reporting_error:
                logger.error(f"Failed to report error: {reporting_error}")
            
            # Show user-friendly error in Streamlit if possible
            if "st" in sys.modules:
                environment = config.get_deployment_environment()
                
                # Show friendly error message
                st.error(
                    f"An unexpected error occurred. "
                    f"Reference ID: {error_details.get('error_id', 'unknown')}"
                )
                
                # Show detailed error info in development
                if environment == "development":
                    st.exception(e)
                    st.write("### Error Details")
                    st.json({
                        k: v for k, v in error_details.items()
                        if k not in ["exception", "traceback"]
                    })
            
            # Re-raise in development, swallow in production
            if config.get_deployment_environment() == "development":
                raise
            else:
                # Return a default value or error indicator
                return None
    
    return cast(F, wrapper)


def create_error_boundary(fallback_ui: Optional[Callable] = None) -> Callable:
    """
    Create an error boundary for a Streamlit component.
    
    Args:
        fallback_ui: Function to render fallback UI on error
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Create a unique key for this error boundary
            component_name = func.__name__
            error_key = f"error_boundary_{component_name}_{id(func)}"
            
            # Check if we've already had an error
            had_error = error_key in st.session_state and st.session_state[error_key]
            
            if had_error:
                # We've already had an error, show fallback UI
                if fallback_ui:
                    return fallback_ui(*args, **kwargs)
                else:
                    st.warning(
                        f"Component '{component_name}' has been disabled due to an error. "
                        f"Refresh the page to try again."
                    )
                    return None
            
            # Try to render the component
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Mark this component as having an error
                st.session_state[error_key] = True
                
                # Log and report the error
                error_details = {
                    "component": component_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc()
                }
                
                # Report error
                error_reporter = ErrorReporter.get_instance()
                error_id = error_reporter.report(error_details)
                
                # Show error UI
                st.error(
                    f"Error rendering component. Reference ID: {error_id}"
                )
                
                # Show fallback UI if provided
                if fallback_ui:
                    return fallback_ui(*args, **kwargs)
                
                # Show detailed error in development
                if config.get_deployment_environment() == "development":
                    st.exception(e)
                
                return None
        
        return wrapper
    
    return decorator


def handle_api_errors(
    retries: int = 3,
    retry_delay: float = 1.0,
    fallback_value: Any = None
) -> Callable:
    """
    Decorator to handle API errors with retry logic.
    
    Args:
        retries: Number of retry attempts
        retry_delay: Delay between retries in seconds
        fallback_value: Value to return if all retries fail
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import time
            
            # Get retry configuration from config if available
            max_retries = config.get_error_handling_config("max_retries")
            if max_retries is not None:
                actual_retries = max_retries
            else:
                actual_retries = retries
                
            # Track attempts
            last_exception = None
            
            # Try multiple times
            for attempt in range(actual_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Log the error
                    logger.warning(
                        f"API call failed (attempt {attempt+1}/{actual_retries}): {e}"
                    )
                    
                    # Store the last exception
                    last_exception = e
                    
                    # Wait before retrying (with exponential backoff)
                    if attempt < actual_retries - 1:
                        time.sleep(retry_delay * (2 ** attempt))
            
            # All retries failed
            error_details = {
                "function": func.__name__,
                "error_message": str(last_exception),
                "retries": actual_retries,
                "args": str(args),
                "kwargs": str(kwargs)
            }
            
            # Report the error
            error_reporter = ErrorReporter.get_instance()
            error_id = error_reporter.report(error_details)
            
            # Log the final failure
            logger.error(
                f"API call failed after {actual_retries} attempts. "
                f"Error ID: {error_id}, Error: {last_exception}"
            )
            
            # Return fallback value
            return fallback_value
        
        return wrapper
    
    return decorator


def initialize_error_handling() -> None:
    """Initialize error handling for the application."""
    # Initialize error reporter
    ErrorReporter.get_instance()
    
    # Set uncaught exception handler
    def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
        # Don't break on KeyboardInterrupt
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # Get error details
        error_details = {
            "error_type": exc_type.__name__,
            "error_message": str(exc_value),
            "traceback": "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
            "exception": exc_value
        }
        
        # Report the error
        try:
            error_reporter = ErrorReporter.get_instance()
            error_reporter.report(error_details)
        except Exception:
            # Last resort logging if reporting fails
            logger.critical(
                f"Uncaught exception: {exc_value}",
                exc_info=(exc_type, exc_value, exc_traceback)
            )
    
    # Set the exception hook
    sys.excepthook = handle_uncaught_exception
    
    logger.info("Error handling initialized") 