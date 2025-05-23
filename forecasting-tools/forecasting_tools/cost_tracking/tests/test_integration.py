"""
Tests for the integration module.
"""

import unittest
import tempfile
import os
from unittest.mock import MagicMock, patch

from forecasting_tools.cost_tracking.integration import (
    CostTrackingMixin, 
    with_cost_tracking, 
    CostTrackingBot
)
from forecasting_tools.cost_tracking.cost_tracker import CostTracker


class MockForecastingBot:
    """Mock ForecastingBot for testing."""
    
    def __init__(self, personality_name=None, model_name=None):
        self.personality_name = personality_name
        self.model_name = model_name
    
    def forecast_binary(self, question):
        """Mock forecast_binary method."""
        # Create a mock result
        result = MagicMock()
        result.binary_prob = 0.7
        result.reasoning = "This is the reasoning"
        result.metadata = {
            "token_usage": {
                "total_tokens": 1000
            },
            "model": self.model_name
        }
        return result


class TestCostTrackingMixin(unittest.TestCase):
    """Tests for the CostTrackingMixin class."""
    
    def setUp(self):
        """Set up a temporary database for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_integration.db")
        
        # Create a bot class with the mixin
        class TestBot(CostTrackingMixin, MockForecastingBot):
            pass
        
        # Initialize the bot
        self.tracker = CostTracker(db_path=self.db_path)
        self.bot = TestBot(
            personality_name="analytical",
            model_name="gpt-4",
            cost_tracker=self.tracker
        )
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_track_forecast_cost(self):
        """Test tracking forecast cost."""
        # Create mock question and result
        question = MagicMock()
        question.question_text = "Test question"
        question.id = "test-q1"
        
        result = MagicMock()
        result.reasoning = "This is the reasoning"
        result.metadata = {
            "token_usage": {
                "total_tokens": 1000
            },
            "model": "gpt-4"
        }
        
        # Track forecast cost
        cost_info = self.bot._track_forecast_cost(question, result)
        
        # Check cost info
        self.assertEqual(cost_info["tokens_used"], 1000)
        self.assertEqual(cost_info["model_name"], "gpt-4")
        self.assertEqual(cost_info["personality_name"], "analytical")
        self.assertAlmostEqual(cost_info["cost_usd"], 0.03, places=6)  # 1000 tokens at $0.03 per 1K
    
    def test_get_cost_history(self):
        """Test getting cost history."""
        # Create mock question and result
        question = MagicMock()
        question.question_text = "Test question"
        question.id = "test-q1"
        
        result = MagicMock()
        result.reasoning = "This is the reasoning"
        result.metadata = {
            "token_usage": {
                "total_tokens": 1000
            },
            "model": "gpt-4"
        }
        
        # Track forecast cost
        self.bot._track_forecast_cost(question, result)
        
        # Get cost history
        history = self.bot.get_cost_history()
        
        # Check history
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].question_id, "test-q1")
        self.assertEqual(history[0].tokens_used, 1000)
        self.assertEqual(history[0].model_name, "gpt-4")
        self.assertEqual(history[0].personality_name, "analytical")
    
    def test_get_total_cost(self):
        """Test getting total cost."""
        # Create mock question and result
        question = MagicMock()
        question.question_text = "Test question"
        question.id = "test-q1"
        
        result = MagicMock()
        result.reasoning = "This is the reasoning"
        result.metadata = {
            "token_usage": {
                "total_tokens": 1000
            },
            "model": "gpt-4"
        }
        
        # Track forecast cost
        self.bot._track_forecast_cost(question, result)
        
        # Get total cost
        total = self.bot.get_total_cost()
        
        # Check total
        self.assertAlmostEqual(total, 0.03, places=6)  # 1000 tokens at $0.03 per 1K


class TestWithCostTracking(unittest.TestCase):
    """Tests for the with_cost_tracking decorator."""
    
    def test_decorator(self):
        """Test the decorator."""
        # Apply decorator to the mock bot
        TrackedBot = with_cost_tracking(MockForecastingBot)
        
        # Check that the class has the mixin's methods
        self.assertTrue(hasattr(TrackedBot, '_track_forecast_cost'))
        self.assertTrue(hasattr(TrackedBot, 'get_cost_history'))
        self.assertTrue(hasattr(TrackedBot, 'get_total_cost'))
        
        # Create an instance
        bot = TrackedBot(personality_name="analytical", model_name="gpt-4")
        
        # Check instance attributes
        self.assertEqual(bot.personality_name, "analytical")
        self.assertEqual(bot.model_name, "gpt-4")
        
        # Check that it's instance of both classes
        self.assertIsInstance(bot, MockForecastingBot)
        self.assertIsInstance(bot, CostTrackingMixin)


class TestCostTrackingBot(unittest.TestCase):
    """Tests for the CostTrackingBot class."""
    
    def setUp(self):
        """Set up a temporary database for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_tracking_bot.db")
        self.tracker = CostTracker(db_path=self.db_path)
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_initialization_with_class(self):
        """Test initialization with a bot class."""
        # Create a tracking bot with a class
        bot = CostTrackingBot(
            bot_cls=MockForecastingBot,
            personality_name="analytical",
            model_name="gpt-4",
            cost_tracker=self.tracker
        )
        
        # Check that the bot was initialized properly
        self.assertEqual(bot.bot.personality_name, "analytical")
        self.assertEqual(bot.bot.model_name, "gpt-4")
        self.assertEqual(bot.bot_cls, MockForecastingBot)
    
    def test_initialization_with_instance(self):
        """Test initialization with a bot instance."""
        # Create a bot instance
        mock_bot = MockForecastingBot(
            personality_name="analytical",
            model_name="gpt-4"
        )
        
        # Create a tracking bot with the instance
        bot = CostTrackingBot(
            bot_cls=mock_bot,
            cost_tracker=self.tracker
        )
        
        # Check that the bot was initialized properly
        self.assertEqual(bot.bot, mock_bot)
        self.assertEqual(bot.bot_cls, MockForecastingBot)
    
    @patch('forecasting_tools.cost_tracking.integration.CostTracker')
    def test_method_forwarding(self, mock_cost_tracker):
        """Test that methods are forwarded to the wrapped bot."""
        # Create a mock question
        question = MagicMock()
        question.question_text = "Test question"
        question.id = "test-q1"
        
        # Create a tracking bot
        mock_bot = MockForecastingBot(
            personality_name="analytical",
            model_name="gpt-4"
        )
        bot = CostTrackingBot(
            bot_cls=mock_bot,
            cost_tracker=mock_cost_tracker
        )
        
        # Call the forecast method
        result = bot.forecast_binary(question)
        
        # Check that the method was called and returned a result
        self.assertEqual(result.binary_prob, 0.7)
        
        # Check that cost tracking was performed
        self.assertTrue(hasattr(result.metadata, 'cost_info'))


if __name__ == '__main__':
    unittest.main() 