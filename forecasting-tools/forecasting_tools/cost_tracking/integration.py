"""
Integration with ForecastingBot for cost tracking.

This module provides integration with the ForecastingBot class to track
token usage and costs for forecasts.
"""

from typing import Dict, Any, Optional, Union, List, Type
from uuid import uuid4
import logging
import functools
import inspect

from forecasting_tools.cost_tracking.cost_tracker import CostTracker, ForecastCost

logger = logging.getLogger(__name__)

class CostTrackingMixin:
    """
    Mixin to add cost tracking to ForecastingBot.
    
    This mixin class provides methods to track token usage and costs for
    forecasts made with different models and personalities.
    """
    
    def __init__(self, *args, cost_tracker: Optional[CostTracker] = None, **kwargs):
        """
        Initialize the mixin with an optional cost tracker.
        
        Args:
            cost_tracker: Custom CostTracker instance (optional)
            *args, **kwargs: Passed to the parent class
        """
        # Initialize cost tracker
        self._cost_tracker = cost_tracker or CostTracker()
        
        # Store the original __init__ args for potential use
        self._init_args = args
        self._init_kwargs = kwargs.copy()
        
        # Call parent's __init__
        super().__init__(*args, **kwargs)
    
    def _track_forecast_cost(self, 
                            question: Any, 
                            result: Any,
                            model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Track the cost of a forecast.
        
        Args:
            question: The question that was forecast
            result: The forecast result
            model_name: Override for the model name
            
        Returns:
            Dictionary with cost information
        """
        # Extract metadata from the result if available
        metadata = getattr(result, 'metadata', {}) or {}
        
        # Get token usage from metadata
        token_usage = metadata.get('token_usage', {})
        tokens_used = token_usage.get('total_tokens', 0)
        
        # If no token usage info, try to estimate
        if tokens_used == 0:
            # Estimate based on the length of the reasoning
            reasoning = getattr(result, 'reasoning', '')
            if reasoning:
                # Rough estimate: ~4 chars per token
                tokens_used = len(reasoning) // 4
            else:
                # Default to a moderate value if we can't estimate
                tokens_used = 1000
                logger.warning("No token usage info available, using estimate: %d tokens", tokens_used)
        
        # Get model name from metadata or instance variable
        if model_name is None:
            model_name = metadata.get('model', None)
        if model_name is None:
            model_name = getattr(self, 'model_name', None)
        if model_name is None:
            model_name = getattr(self, 'model', None)
        if model_name is None:
            model_name = "default"
        
        # Get personality name from instance variable
        personality_name = getattr(self, 'personality_name', None)
        
        # Get question id and text
        question_id = getattr(question, 'id', None)
        if question_id is None:
            question_id = str(uuid4())
            
        question_text = getattr(question, 'question_text', str(question))
        
        # Track the forecast cost
        cost_record = self._cost_tracker.track_forecast(
            question_id=question_id,
            question_text=question_text,
            tokens_used=tokens_used,
            model_name=model_name,
            personality_name=personality_name
        )
        
        # Return cost information
        return {
            "cost_usd": cost_record.cost_usd,
            "tokens_used": cost_record.tokens_used,
            "timestamp": cost_record.timestamp,
            "model_name": cost_record.model_name,
            "personality_name": cost_record.personality_name
        }
    
    def get_cost_history(self, limit: int = 100) -> List[ForecastCost]:
        """
        Get the cost history for forecasts.
        
        Args:
            limit: Maximum number of records to retrieve
            
        Returns:
            List of ForecastCost objects
        """
        return self._cost_tracker.get_cost_history(limit=limit)
    
    def get_total_cost(self) -> float:
        """
        Get the total cost of all forecasts.
        
        Returns:
            Total cost in USD
        """
        return self._cost_tracker.get_total_cost()
    
    def get_cost_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cost statistics.
        
        Returns:
            Dictionary with cost statistics
        """
        return self._cost_tracker.get_cost_statistics()


def with_cost_tracking(cls):
    """
    Class decorator to add cost tracking to a class.
    
    This decorator can be used to add cost tracking to any ForecastingBot class
    without modifying the original class.
    
    Example:
        @with_cost_tracking
        class MyForecastingBot(ForecastingBot):
            pass
    """
    # Skip if already has cost tracking
    if hasattr(cls, '_track_forecast_cost'):
        return cls
    
    # Create a new class that inherits from CostTrackingMixin and the original class
    class_name = f"CostTracking{cls.__name__}"
    tracked_cls = type(class_name, (CostTrackingMixin, cls), {})
    
    # Copy over docstring and other attributes
    tracked_cls.__doc__ = cls.__doc__
    tracked_cls.__module__ = cls.__module__
    
    return tracked_cls


class CostTrackingBot:
    """
    ForecastingBot with cost tracking functionality.
    
    This class implements the composition pattern to add cost tracking to
    any ForecastingBot without inheritance. It wraps all methods of the
    original bot and adds cost tracking.
    """
    
    def __init__(self, bot_cls=None, cost_tracker=None, **kwargs):
        """
        Initialize with a bot class and optional cost tracker.
        
        Args:
            bot_cls: ForecastingBot class or instance
            cost_tracker: Custom CostTracker instance (optional)
            **kwargs: Passed to the bot constructor
        """
        from forecasting_tools import ForecastingBot
        
        # Import locally to avoid circular imports
        if bot_cls is None:
            # Use the default ForecastingBot
            self.bot_cls = ForecastingBot
            self.bot = ForecastingBot(**kwargs)
        elif isinstance(bot_cls, type):
            # bot_cls is a class, instantiate it
            self.bot_cls = bot_cls
            self.bot = bot_cls(**kwargs)
        else:
            # bot_cls is an instance, use it directly
            self.bot = bot_cls
            self.bot_cls = bot_cls.__class__
        
        # Initialize cost tracker
        self.cost_tracker = cost_tracker or CostTracker()
        
        # Store kwargs for potential use
        self.kwargs = kwargs
    
    def _track_forecast_cost(self, question, result, model_name=None):
        """Track the cost of a forecast."""
        # Extract metadata from the result if available
        metadata = getattr(result, 'metadata', {}) or {}
        
        # Get token usage from metadata
        token_usage = metadata.get('token_usage', {})
        tokens_used = token_usage.get('total_tokens', 0)
        
        # If no token usage info, try to estimate
        if tokens_used == 0:
            # Estimate based on the length of the reasoning
            reasoning = getattr(result, 'reasoning', '')
            if reasoning:
                # Rough estimate: ~4 chars per token
                tokens_used = len(reasoning) // 4
            else:
                # Default to a moderate value if we can't estimate
                tokens_used = 1000
                logger.warning("No token usage info available, using estimate: %d tokens", tokens_used)
        
        # Get model name from metadata or instance variable
        if model_name is None:
            model_name = metadata.get('model', None)
        if model_name is None:
            model_name = getattr(self.bot, 'model_name', None)
        if model_name is None:
            model_name = getattr(self.bot, 'model', None)
        if model_name is None:
            model_name = "default"
        
        # Get personality name from instance variable
        personality_name = getattr(self.bot, 'personality_name', None)
        
        # Get question id and text
        question_id = getattr(question, 'id', None)
        if question_id is None:
            question_id = str(uuid4())
            
        question_text = getattr(question, 'question_text', str(question))
        
        # Track the forecast cost
        cost_record = self.cost_tracker.track_forecast(
            question_id=question_id,
            question_text=question_text,
            tokens_used=tokens_used,
            model_name=model_name,
            personality_name=personality_name
        )
        
        # Return cost information
        return {
            "cost_usd": cost_record.cost_usd,
            "tokens_used": cost_record.tokens_used,
            "timestamp": cost_record.timestamp,
            "model_name": cost_record.model_name,
            "personality_name": cost_record.personality_name
        }
    
    def __getattr__(self, name):
        """
        Forward attribute access to the wrapped bot.
        
        This allows transparent access to all methods and attributes of the
        wrapped bot instance.
        
        Special handling for forecast methods to add cost tracking.
        """
        attr = getattr(self.bot, name)
        
        # If it's a forecast method, wrap it to add cost tracking
        if callable(attr) and name.startswith('forecast_'):
            @functools.wraps(attr)
            def wrapper(*args, **kwargs):
                # Call the original method
                result = attr(*args, **kwargs)
                
                # Track the cost
                if args and hasattr(args[0], 'question_text'):
                    question = args[0]
                    cost_info = self._track_forecast_cost(question, result)
                    
                    # Add cost info to result metadata
                    if hasattr(result, 'metadata'):
                        if result.metadata is None:
                            result.metadata = {}
                        result.metadata['cost_info'] = cost_info
                
                return result
            
            return wrapper
        
        # Return the original attribute
        return attr
    
    def get_cost_history(self, limit=100):
        """Get the cost history for forecasts."""
        return self.cost_tracker.get_cost_history(limit=limit)
    
    def get_total_cost(self):
        """Get the total cost of all forecasts."""
        return self.cost_tracker.get_total_cost()
    
    def get_cost_statistics(self):
        """Get comprehensive cost statistics."""
        return self.cost_tracker.get_cost_statistics()


# Create a function to easily get a cost tracking bot
def get_cost_tracking_bot(**kwargs):
    """
    Get a ForecastingBot with cost tracking.
    
    This function returns a ForecastingBot instance with cost tracking
    functionality added.
    
    Args:
        **kwargs: Passed to the ForecastingBot constructor
        
    Returns:
        CostTrackingBot instance
    """
    return CostTrackingBot(**kwargs) 