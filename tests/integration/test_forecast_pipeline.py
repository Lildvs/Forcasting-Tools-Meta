import unittest
import tempfile
import os
import json
import time
from pathlib import Path
import logging
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

# Import the core components for integration testing
from forecasting_tools.data.storage import MemoryStorage, StorageManager
from forecasting_tools.data.models import DatabaseManager, User, Forecast, ResearchData
from forecasting_tools.util.queue_manager import QueueManager, JobType, queue_manager_context
from forecasting_tools.utils.monitoring import get_monitoring_manager


class TestForecastingPipeline(unittest.TestCase):
    """
    Integration tests for the complete forecasting pipeline.
    
    Tests the end-to-end process of generating forecasts, including:
    - Data persistence
    - Job queuing
    - Forecast generation
    - Result storage
    """
    
    def setUp(self):
        """Set up the test environment."""
        # Configure logging for tests
        logging.basicConfig(level=logging.INFO)
        
        # Use in-memory storage for tests
        self.storage = MemoryStorage()
        self.storage_manager = StorageManager(self.storage)
        
        # Set up in-memory SQLite database
        self.db_manager = DatabaseManager("sqlite:///:memory:")
        self.db_manager.create_tables()
        
        # Create test user
        with self.db_manager.session_scope() as session:
            test_user = User(
                id="test-user-id",
                username="testuser",
                email="test@example.com",
                password_hash="test-hash"
            )
            session.add(test_user)
        
        # Initialize monitoring
        self.monitoring = get_monitoring_manager()
        
        # Mock LLM client for testing
        self.mock_llm_patcher = patch("forecasting_tools.llm.client.LLMClient")
        self.mock_llm = self.mock_llm_patcher.start()
        
        # Configure mock LLM response
        self.mock_llm.return_value.__enter__.return_value.generate.return_value = {
            "forecast_type": "binary",
            "probability": 0.75,
            "reasoning": "This is a mock forecast with reasoning."
        }
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop the mock
        self.mock_llm_patcher.stop()
        
        # Clean up database
        self.db_manager.drop_tables()
        self.db_manager.close()
        
        # Clean up storage
        self.storage_manager.close()
    
    @patch("forecasting_tools.search.searcher.Searcher")
    def test_end_to_end_forecast_generation(self, mock_searcher):
        """Test the end-to-end forecast generation process."""
        # Mock search results
        mock_searcher.return_value.__enter__.return_value.search.return_value = [
            {"title": "Test Article 1", "content": "Test content 1", "url": "http://example.com/1"},
            {"title": "Test Article 2", "content": "Test content 2", "url": "http://example.com/2"},
        ]
        
        # Set up a queue manager for the forecast job
        with queue_manager_context(max_workers=2) as queue_manager:
            # Import the forecast handler here to avoid circular imports
            from forecasting_tools.forecast.handlers import handle_forecast_job
            
            # Register the handler
            queue_manager.register_handler(JobType.FORECAST, handle_forecast_job)
            
            # Create a forecast request
            forecast_request = {
                "question": "Will AI replace programmers by 2030?",
                "user_id": "test-user-id",
                "search_queries": ["AI impact on programming jobs", "software engineering future"],
                "forecast_type": "binary"
            }
            
            # Enqueue the forecast job
            job_id = queue_manager.enqueue(
                JobType.FORECAST,
                forecast_request,
                max_retries=1
            )
            
            # Wait for job to complete
            max_wait = 10.0  # seconds
            start_time = time.time()
            status = None
            
            while time.time() - start_time < max_wait:
                status = queue_manager.get_job_status(job_id)
                if status and status["status"] in ("completed", "failed"):
                    break
                time.sleep(0.1)
            
            # Check that job completed successfully
            self.assertIsNotNone(status)
            self.assertEqual(status["status"], "completed")
            
            # Check that forecast was stored in database
            with self.db_manager.session_scope() as session:
                forecasts = session.query(Forecast).all()
                self.assertEqual(len(forecasts), 1)
                
                # Check forecast details
                forecast = forecasts[0]
                self.assertEqual(forecast.question_text, forecast_request["question"])
                self.assertEqual(forecast.user_id, forecast_request["user_id"])
                self.assertEqual(forecast.forecast_type, "binary")
                self.assertIsNotNone(forecast.probability)
                self.assertIsNotNone(forecast.reasoning)
                
                # Check that research data was stored
                research_items = session.query(ResearchData).filter(ResearchData.forecast_id == forecast.id).all()
                self.assertTrue(len(research_items) > 0)
    
    @patch("forecasting_tools.search.searcher.Searcher")
    def test_parallel_forecast_processing(self, mock_searcher):
        """Test processing multiple forecasts in parallel."""
        # Mock search results
        mock_searcher.return_value.__enter__.return_value.search.return_value = [
            {"title": "Test Article", "content": "Test content", "url": "http://example.com/1"},
        ]
        
        # Set up a queue manager
        with queue_manager_context(max_workers=4) as queue_manager:
            # Import the forecast handler
            from forecasting_tools.forecast.handlers import handle_forecast_job
            
            # Register the handler
            queue_manager.register_handler(JobType.FORECAST, handle_forecast_job)
            
            # Create multiple forecast requests
            forecast_questions = [
                "Will global temperatures rise by more than 2 degrees by 2050?",
                "Will quantum computing be commercially viable by 2030?",
                "Will human-level AGI be developed before 2040?"
            ]
            
            job_ids = []
            for question in forecast_questions:
                forecast_request = {
                    "question": question,
                    "user_id": "test-user-id",
                    "search_queries": ["related query 1", "related query 2"],
                    "forecast_type": "binary"
                }
                
                # Enqueue the forecast job
                job_id = queue_manager.enqueue(
                    JobType.FORECAST,
                    forecast_request,
                    max_retries=1
                )
                job_ids.append(job_id)
            
            # Wait for all jobs to complete
            max_wait = 15.0  # seconds
            start_time = time.time()
            completed_count = 0
            
            while time.time() - start_time < max_wait and completed_count < len(job_ids):
                completed_count = 0
                for job_id in job_ids:
                    status = queue_manager.get_job_status(job_id)
                    if status and status["status"] in ("completed", "failed"):
                        completed_count += 1
                
                time.sleep(0.1)
            
            # Check that all jobs completed
            self.assertEqual(completed_count, len(job_ids))
            
            # Check that all forecasts were stored in database
            with self.db_manager.session_scope() as session:
                forecasts = session.query(Forecast).all()
                self.assertEqual(len(forecasts), len(forecast_questions))
    
    @patch("forecasting_tools.search.searcher.Searcher")
    def test_forecast_with_error_handling(self, mock_searcher):
        """Test forecast generation with error handling and retries."""
        # Mock search results to fail on first call, succeed on second
        mock_search = mock_searcher.return_value.__enter__.return_value.search
        mock_search.side_effect = [
            Exception("Simulated search failure"),  # First call fails
            [{"title": "Test Article", "content": "Test content", "url": "http://example.com/1"}]  # Second call succeeds
        ]
        
        # Set up a queue manager
        with queue_manager_context(max_workers=2) as queue_manager:
            # Import the forecast handler
            from forecasting_tools.forecast.handlers import handle_forecast_job
            
            # Register the handler
            queue_manager.register_handler(JobType.FORECAST, handle_forecast_job)
            
            # Create a forecast request
            forecast_request = {
                "question": "Will nuclear fusion be a viable energy source by 2040?",
                "user_id": "test-user-id",
                "search_queries": ["nuclear fusion progress", "fusion energy timeline"],
                "forecast_type": "binary"
            }
            
            # Enqueue the forecast job with retry
            job_id = queue_manager.enqueue(
                JobType.FORECAST,
                forecast_request,
                max_retries=1,
                retry_delay=1
            )
            
            # Wait for job to complete
            max_wait = 15.0  # seconds
            start_time = time.time()
            status = None
            
            while time.time() - start_time < max_wait:
                status = queue_manager.get_job_status(job_id)
                if status and status["status"] in ("completed", "failed"):
                    break
                time.sleep(0.1)
            
            # Check that job completed successfully after retry
            self.assertIsNotNone(status)
            self.assertEqual(status["status"], "completed")
            
            # Verify that search was called twice (initial failure + retry)
            self.assertEqual(mock_search.call_count, 2)
    
    def test_forecast_data_persistence(self):
        """Test that forecast data is properly persisted and retrievable."""
        # Create a test forecast
        forecast_data = {
            "question": "Will SpaceX land humans on Mars before 2035?",
            "forecast_type": "binary",
            "probability": 0.65,
            "reasoning": "Based on current development pace...",
            "user_id": "test-user-id",
            "metadata": {
                "sources": [
                    {"title": "NASA Mars Plans", "url": "http://example.com/nasa"},
                    {"title": "SpaceX Roadmap", "url": "http://example.com/spacex"}
                ]
            }
        }
        
        # Save forecast using the storage manager
        forecast_id = self.storage_manager.save_forecast(forecast_data)
        
        # Retrieve the forecast
        retrieved = self.storage_manager.get_forecast(forecast_id)
        
        # Check that data was correctly persisted
        self.assertEqual(retrieved["question"], forecast_data["question"])
        self.assertEqual(retrieved["probability"], forecast_data["probability"])
        self.assertEqual(retrieved["reasoning"], forecast_data["reasoning"])
        
        # Check metadata was preserved
        self.assertEqual(len(retrieved["metadata"]["sources"]), 2)
        self.assertEqual(retrieved["metadata"]["sources"][0]["title"], "NASA Mars Plans")
        
        # Now save to the ORM database as well
        with self.db_manager.session_scope() as session:
            db_forecast = Forecast(
                question_text=forecast_data["question"],
                forecast_type=forecast_data["forecast_type"],
                probability=forecast_data["probability"],
                reasoning=forecast_data["reasoning"],
                user_id=forecast_data["user_id"],
                metadata=forecast_data["metadata"]
            )
            session.add(db_forecast)
            session.flush()
            db_forecast_id = db_forecast.id
        
        # Retrieve from database and verify
        with self.db_manager.session_scope() as session:
            retrieved_db_forecast = session.query(Forecast).filter(Forecast.id == db_forecast_id).one()
            self.assertEqual(retrieved_db_forecast.question_text, forecast_data["question"])
            self.assertEqual(retrieved_db_forecast.probability, forecast_data["probability"])
            self.assertEqual(retrieved_db_forecast.reasoning, forecast_data["reasoning"])
            self.assertEqual(len(retrieved_db_forecast.metadata["sources"]), 2)


if __name__ == "__main__":
    unittest.main() 