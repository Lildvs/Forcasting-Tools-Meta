"""
Tests for the CostTracker class.
"""

import os
import unittest
import tempfile
from datetime import datetime, timedelta
import sqlite3

from forecasting_tools.cost_tracking.cost_tracker import CostTracker, ForecastCost


class TestCostTracker(unittest.TestCase):
    """Tests for the CostTracker class."""
    
    def setUp(self):
        """Set up a temporary database for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_costs.db")
        self.tracker = CostTracker(db_path=self.db_path)
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_initialize_db(self):
        """Test database initialization."""
        # Check that the database file was created
        self.assertTrue(os.path.exists(self.db_path))
        
        # Check that the table exists
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='forecast_costs'")
        table_exists = cursor.fetchone() is not None
        conn.close()
        
        self.assertTrue(table_exists)
    
    def test_calculate_cost(self):
        """Test cost calculation."""
        # Test with default model
        cost = self.tracker.calculate_cost(1000, "default")
        self.assertEqual(cost, 0.01)  # 1000 tokens at $0.01 per 1K tokens
        
        # Test with specific model
        cost = self.tracker.calculate_cost(2000, "gpt-4")
        self.assertEqual(cost, 0.06)  # 2000 tokens at $0.03 per 1K tokens
        
        # Test with unknown model (should use default rate)
        cost = self.tracker.calculate_cost(1000, "unknown-model")
        self.assertEqual(cost, 0.01)  # 1000 tokens at $0.01 per 1K tokens
    
    def test_track_forecast(self):
        """Test tracking a forecast."""
        # Track a forecast
        cost_record = self.tracker.track_forecast(
            question_id="test-q1",
            question_text="Test question",
            tokens_used=1000,
            model_name="gpt-4",
            personality_name="analytical"
        )
        
        # Check the cost record
        self.assertEqual(cost_record.question_id, "test-q1")
        self.assertEqual(cost_record.question_text, "Test question")
        self.assertEqual(cost_record.tokens_used, 1000)
        self.assertEqual(cost_record.model_name, "gpt-4")
        self.assertEqual(cost_record.personality_name, "analytical")
        self.assertEqual(cost_record.cost_usd, 0.03)  # 1000 tokens at $0.03 per 1K tokens
        
        # Check that a timestamp was set
        self.assertIsInstance(cost_record.timestamp, datetime)
        
        # Track another forecast with generated ID
        cost_record2 = self.tracker.track_forecast(
            question_id=None,
            question_text="Another question",
            tokens_used=2000,
            model_name="gpt-3.5-turbo"
        )
        
        # Check the generated ID
        self.assertIsNotNone(cost_record2.question_id)
        self.assertNotEqual(cost_record2.question_id, "test-q1")
    
    def test_get_cost_history(self):
        """Test retrieving cost history."""
        # Track some forecasts
        self.tracker.track_forecast(
            question_id="test-q1",
            question_text="Question 1",
            tokens_used=1000,
            model_name="gpt-4"
        )
        self.tracker.track_forecast(
            question_id="test-q2",
            question_text="Question 2",
            tokens_used=2000,
            model_name="gpt-3.5-turbo"
        )
        
        # Get history
        history = self.tracker.get_cost_history()
        
        # Check history
        self.assertEqual(len(history), 2)
        
        # Check most recent first ordering
        self.assertEqual(history[0].question_id, "test-q2")
        self.assertEqual(history[1].question_id, "test-q1")
        
        # Test limit parameter
        limited_history = self.tracker.get_cost_history(limit=1)
        self.assertEqual(len(limited_history), 1)
        self.assertEqual(limited_history[0].question_id, "test-q2")
    
    def test_get_total_cost(self):
        """Test getting the total cost."""
        # Track some forecasts
        self.tracker.track_forecast(
            question_id="test-q1",
            question_text="Question 1",
            tokens_used=1000,
            model_name="gpt-4"  # $0.03
        )
        self.tracker.track_forecast(
            question_id="test-q2",
            question_text="Question 2",
            tokens_used=2000,
            model_name="gpt-3.5-turbo"  # $0.004
        )
        
        # Get total cost
        total = self.tracker.get_total_cost()
        
        # Expected: $0.03 + $0.004 = $0.034
        self.assertAlmostEqual(total, 0.034, places=6)
    
    def test_get_cost_by_date_range(self):
        """Test getting costs by date range."""
        # Track forecasts with specific timestamps
        yesterday = datetime.now() - timedelta(days=1)
        today = datetime.now()
        
        # Manually create and save cost records with specific timestamps
        yesterday_cost = ForecastCost(
            question_id="test-q1",
            question_text="Question 1",
            timestamp=yesterday,
            tokens_used=1000,
            cost_usd=0.03,
            model_name="gpt-4"
        )
        self.tracker._save_cost(yesterday_cost)
        
        today_cost = ForecastCost(
            question_id="test-q2",
            question_text="Question 2",
            timestamp=today,
            tokens_used=2000,
            cost_usd=0.004,
            model_name="gpt-3.5-turbo"
        )
        self.tracker._save_cost(today_cost)
        
        # Get cost for today only
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = datetime.now()
        today_total = self.tracker.get_cost_by_date_range(start_date, end_date)
        
        # Expected: $0.004
        self.assertAlmostEqual(today_total, 0.004, places=6)
        
        # Get cost for yesterday and today
        start_date = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = datetime.now()
        total = self.tracker.get_cost_by_date_range(start_date, end_date)
        
        # Expected: $0.03 + $0.004 = $0.034
        self.assertAlmostEqual(total, 0.034, places=6)
    
    def test_get_cost_by_model(self):
        """Test getting costs grouped by model."""
        # Track forecasts with different models
        self.tracker.track_forecast(
            question_id="test-q1",
            question_text="Question 1",
            tokens_used=1000,
            model_name="gpt-4"  # $0.03
        )
        self.tracker.track_forecast(
            question_id="test-q2",
            question_text="Question 2",
            tokens_used=2000,
            model_name="gpt-3.5-turbo"  # $0.004
        )
        self.tracker.track_forecast(
            question_id="test-q3",
            question_text="Question 3",
            tokens_used=1000,
            model_name="gpt-4"  # $0.03
        )
        
        # Get costs by model
        model_costs = self.tracker.get_cost_by_model()
        
        # Check costs
        self.assertAlmostEqual(model_costs["gpt-4"], 0.06, places=6)  # 2 * $0.03
        self.assertAlmostEqual(model_costs["gpt-3.5-turbo"], 0.004, places=6)
    
    def test_get_cost_by_personality(self):
        """Test getting costs grouped by personality."""
        # Track forecasts with different personalities
        self.tracker.track_forecast(
            question_id="test-q1",
            question_text="Question 1",
            tokens_used=1000,
            model_name="gpt-4",
            personality_name="analytical"
        )
        self.tracker.track_forecast(
            question_id="test-q2",
            question_text="Question 2",
            tokens_used=2000,
            model_name="gpt-3.5-turbo",
            personality_name="creative"
        )
        self.tracker.track_forecast(
            question_id="test-q3",
            question_text="Question 3",
            tokens_used=1000,
            model_name="gpt-4",
            personality_name="analytical"
        )
        
        # Get costs by personality
        personality_costs = self.tracker.get_cost_by_personality()
        
        # Check costs
        self.assertAlmostEqual(personality_costs["analytical"], 0.06, places=6)  # 2 * $0.03
        self.assertAlmostEqual(personality_costs["creative"], 0.004, places=6)
    
    def test_get_daily_costs(self):
        """Test getting daily costs."""
        # Track forecasts with specific timestamps
        yesterday = datetime.now() - timedelta(days=1)
        today = datetime.now()
        
        # Manually create and save cost records with specific timestamps
        yesterday_cost = ForecastCost(
            question_id="test-q1",
            question_text="Question 1",
            timestamp=yesterday,
            tokens_used=1000,
            cost_usd=0.03,
            model_name="gpt-4"
        )
        self.tracker._save_cost(yesterday_cost)
        
        today_cost = ForecastCost(
            question_id="test-q2",
            question_text="Question 2",
            timestamp=today,
            tokens_used=2000,
            cost_usd=0.004,
            model_name="gpt-3.5-turbo"
        )
        self.tracker._save_cost(today_cost)
        
        # Get daily costs
        daily_costs = self.tracker.get_daily_costs(days=2)
        
        # Check daily costs
        yesterday_str = yesterday.strftime('%Y-%m-%d')
        today_str = today.strftime('%Y-%m-%d')
        
        self.assertAlmostEqual(daily_costs[yesterday_str], 0.03, places=6)
        self.assertAlmostEqual(daily_costs[today_str], 0.004, places=6)
    
    def test_clear_history(self):
        """Test clearing history."""
        # Track a forecast
        self.tracker.track_forecast(
            question_id="test-q1",
            question_text="Test question",
            tokens_used=1000,
            model_name="gpt-4"
        )
        
        # Check that history exists
        self.assertEqual(len(self.tracker.get_cost_history()), 1)
        
        # Clear history
        result = self.tracker.clear_history()
        
        # Check result
        self.assertTrue(result)
        
        # Check that history is empty
        self.assertEqual(len(self.tracker.get_cost_history()), 0)
    
    def test_export_to_csv(self):
        """Test exporting to CSV."""
        # Track some forecasts
        self.tracker.track_forecast(
            question_id="test-q1",
            question_text="Question 1",
            tokens_used=1000,
            model_name="gpt-4"
        )
        self.tracker.track_forecast(
            question_id="test-q2",
            question_text="Question 2",
            tokens_used=2000,
            model_name="gpt-3.5-turbo"
        )
        
        # Export to CSV
        export_path = os.path.join(self.temp_dir, "export.csv")
        result = self.tracker.export_to_csv(export_path)
        
        # Check result
        self.assertTrue(result)
        
        # Check that file exists
        self.assertTrue(os.path.exists(export_path))
        
        # Check file content
        with open(export_path, 'r') as f:
            content = f.read()
            
        # Check header
        self.assertTrue(content.startswith("question_id,question_text,timestamp,tokens_used,cost_usd,personality_name,model_name"))
        
        # Check records
        self.assertTrue("test-q1" in content)
        self.assertTrue("test-q2" in content)
        self.assertTrue("Question 1" in content)
        self.assertTrue("Question 2" in content)


if __name__ == '__main__':
    unittest.main() 