"""
Configuration file for pytest with common fixtures.

This file defines fixtures that can be reused across multiple test files.
"""

import os
import tempfile
import pytest
import asyncio
import logging
from unittest.mock import MagicMock, AsyncMock, patch

from forecasting_tools.data.storage import MemoryStorage, SqliteStorage, StorageManager
from forecasting_tools.data.models import DatabaseManager
from forecasting_tools.util.queue_manager import QueueManager, JobType
from forecasting_tools.utils.monitoring import get_monitoring_manager, reset_monitoring_manager
from forecasting_tools.config import Config


# Configure logging for tests
@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Configure logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


# Reset state between tests
@pytest.fixture(autouse=True)
def reset_state():
    """Reset global state between tests."""
    # Reset monitoring manager
    reset_monitoring_manager()
    
    # Reset config singleton
    Config._instance = None
    
    yield


# Memory storage fixture
@pytest.fixture
def memory_storage():
    """Create and initialize a memory storage instance."""
    storage = MemoryStorage()
    storage.initialize()
    return storage


# SQLite storage fixture with temporary file
@pytest.fixture
def sqlite_storage():
    """Create and initialize a SQLite storage instance with a temporary file."""
    # Create a temporary file for the SQLite database
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_file.close()
    
    # Create storage instance
    storage = SqliteStorage(temp_file.name)
    storage.initialize()
    
    yield storage
    
    # Clean up
    storage.close()
    try:
        os.unlink(temp_file.name)
    except (OSError, PermissionError):
        pass  # Ignore errors on cleanup


# Storage manager fixture
@pytest.fixture
def storage_manager(memory_storage):
    """Create a storage manager with memory storage."""
    manager = StorageManager(memory_storage)
    yield manager
    manager.close()


# In-memory database manager fixture
@pytest.fixture
def db_manager():
    """Create a database manager with an in-memory SQLite database."""
    manager = DatabaseManager("sqlite:///:memory:")
    manager.create_tables()
    yield manager
    manager.close()


# Queue manager fixture
@pytest.fixture
def queue_manager():
    """Create a queue manager for testing."""
    manager = QueueManager(max_workers=2)
    
    # Mock job handler
    def mock_handler(job_data):
        return {"status": "success", "result": job_data}
    
    # Register handler for test job type
    manager.register_handler(JobType.FORECAST, mock_handler)
    
    yield manager
    
    # Shutdown the queue manager
    manager.shutdown(wait=True)


# Monitoring manager fixture
@pytest.fixture
def monitoring_manager():
    """Get the monitoring manager instance."""
    return get_monitoring_manager()


# Mock API client fixture
@pytest.fixture
def mock_api_client():
    """Create a mock API client."""
    with patch("forecasting_tools.util.async_helpers.APIClient") as mock_client:
        # Mock response for get method
        mock_get_response = AsyncMock()
        mock_get_response.return_value = {"data": "test"}
        mock_client.return_value.get = mock_get_response
        
        # Mock response for post method
        mock_post_response = AsyncMock()
        mock_post_response.return_value = {"id": "test-id", "status": "success"}
        mock_client.return_value.post = mock_post_response
        
        yield mock_client


# Event loop fixture for asyncio tests
@pytest.fixture
def event_loop():
    """Create an asyncio event loop for tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Sample forecast data fixture
@pytest.fixture
def sample_forecast_data():
    """Create sample forecast data for testing."""
    return {
        "question": "Will AI replace programmers by 2030?",
        "forecast_type": "binary",
        "probability": 0.65,
        "reasoning": "Based on current AI capabilities and the rate of progress...",
        "user_id": "test-user-id",
        "metadata": {
            "sources": [
                {"title": "AI Progress Report", "url": "http://example.com/ai-report"},
                {"title": "Future of Programming", "url": "http://example.com/future-programming"}
            ]
        }
    }


# Sample research data fixture
@pytest.fixture
def sample_research_data():
    """Create sample research data for testing."""
    return {
        "search_queries": ["AI impact on programming jobs", "software engineering future"],
        "results": [
            {
                "title": "The Future of Programming in the Age of AI",
                "content": "As AI capabilities continue to advance...",
                "url": "http://example.com/future-programming",
                "relevance_score": 0.85
            },
            {
                "title": "AI Progress Report 2023",
                "content": "Recent breakthroughs in large language models...",
                "url": "http://example.com/ai-report",
                "relevance_score": 0.92
            }
        ]
    }


# Sample historical predictions fixture
@pytest.fixture
def sample_historical_predictions():
    """Create sample historical predictions with outcomes for testing."""
    return {
        "metadata": {
            "description": "Sample historical predictions for testing",
            "version": "1.0.0"
        },
        "predictions": [
            {
                "id": "pred-1",
                "question": "Will Bitcoin exceed $50,000 by the end of 2021?",
                "category": "economics",
                "forecast_type": "binary",
                "prediction": {
                    "probability": 0.75,
                    "timestamp": "2021-01-01T00:00:00Z",
                    "reasoning": "Based on institutional adoption trends..."
                },
                "outcome": {
                    "result": True,
                    "resolution_timestamp": "2021-12-31T00:00:00Z",
                    "evidence": "Bitcoin reached a high of $69,000 in November 2021."
                }
            },
            {
                "id": "pred-2",
                "question": "Will the S&P 500 close above 4,000 by the end of 2021?",
                "category": "economics",
                "forecast_type": "binary",
                "prediction": {
                    "probability": 0.8,
                    "timestamp": "2021-01-01T00:00:00Z",
                    "reasoning": "Given the economic recovery expected post-vaccine..."
                },
                "outcome": {
                    "result": True,
                    "resolution_timestamp": "2021-12-31T00:00:00Z",
                    "evidence": "The S&P 500 closed at 4,766.18 on December 31, 2021."
                }
            },
            {
                "id": "pred-3",
                "question": "Will at least one COVID-19 vaccine receive FDA approval in 2021?",
                "category": "science",
                "forecast_type": "binary",
                "prediction": {
                    "probability": 0.95,
                    "timestamp": "2021-01-01T00:00:00Z",
                    "reasoning": "Multiple vaccines are already in late-stage trials..."
                },
                "outcome": {
                    "result": True,
                    "resolution_timestamp": "2021-08-23T00:00:00Z",
                    "evidence": "The FDA granted full approval to the Pfizer-BioNTech COVID-19 vaccine on August 23, 2021."
                }
            }
        ]
    } 