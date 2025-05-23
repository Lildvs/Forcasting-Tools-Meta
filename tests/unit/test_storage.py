import unittest
import tempfile
import os
from pathlib import Path
import json
import time
from datetime import datetime

from forecasting_tools.data.storage import StorageAdapter, SqliteStorage, MemoryStorage, StorageManager, PersistenceError


class TestStorageAdapter(unittest.TestCase):
    """Test the abstract StorageAdapter class."""
    
    def test_abstract_methods(self):
        """Test that StorageAdapter is properly abstract."""
        with self.assertRaises(TypeError):
            StorageAdapter()  # Abstract class should not be instantiable


class TestMemoryStorage(unittest.TestCase):
    """Test the MemoryStorage implementation."""
    
    def setUp(self):
        """Set up a memory storage for each test."""
        self.storage = MemoryStorage()
        self.storage.initialize()
    
    def test_initialization(self):
        """Test that initialization creates an empty structure."""
        self.assertTrue(self.storage.initialized)
        self.assertEqual(self.storage.collections, {})
    
    def test_save_and_get(self):
        """Test saving and retrieving data."""
        data = {"name": "test", "value": 42}
        key = self.storage.save("test_collection", data)
        
        # Check that collection was created
        self.assertIn("test_collection", self.storage.collections)
        
        # Check that data was saved with the key
        self.assertIn(key, self.storage.collections["test_collection"])
        
        # Get and check data
        retrieved = self.storage.get("test_collection", key)
        self.assertEqual(retrieved["name"], "test")
        self.assertEqual(retrieved["value"], 42)
        
        # Check that timestamps were added
        self.assertIn("created_at", retrieved)
        self.assertIn("updated_at", retrieved)
    
    def test_save_with_key(self):
        """Test saving data with a specific key."""
        data = {"name": "test", "value": 42}
        key = "custom_key"
        saved_key = self.storage.save("test_collection", data, key)
        
        # Check that the provided key was used
        self.assertEqual(saved_key, key)
        
        # Get and check data
        retrieved = self.storage.get("test_collection", key)
        self.assertEqual(retrieved["name"], "test")
    
    def test_update(self):
        """Test updating existing data."""
        # Save initial data
        data = {"name": "test", "value": 42}
        key = self.storage.save("test_collection", data)
        
        # Update the data
        updated_data = {"name": "test_updated", "value": 43}
        result = self.storage.update("test_collection", key, updated_data)
        
        # Check that update was successful
        self.assertTrue(result)
        
        # Get and check updated data
        retrieved = self.storage.get("test_collection", key)
        self.assertEqual(retrieved["name"], "test_updated")
        self.assertEqual(retrieved["value"], 43)
    
    def test_update_nonexistent(self):
        """Test updating nonexistent data."""
        # Try to update nonexistent data
        updated_data = {"name": "test_updated", "value": 43}
        result = self.storage.update("test_collection", "nonexistent_key", updated_data)
        
        # Check that update fails without upsert
        self.assertFalse(result)
    
    def test_upsert(self):
        """Test upserting data."""
        # Try to update nonexistent data with upsert
        updated_data = {"name": "test_updated", "value": 43}
        result = self.storage.update("test_collection", "upsert_key", updated_data, upsert=True)
        
        # Check that update was successful
        self.assertTrue(result)
        
        # Get and check upserted data
        retrieved = self.storage.get("test_collection", "upsert_key")
        self.assertEqual(retrieved["name"], "test_updated")
        self.assertEqual(retrieved["value"], 43)
    
    def test_delete(self):
        """Test deleting data."""
        # Save initial data
        data = {"name": "test", "value": 42}
        key = self.storage.save("test_collection", data)
        
        # Delete the data
        result = self.storage.delete("test_collection", key)
        
        # Check that delete was successful
        self.assertTrue(result)
        
        # Check that data is no longer retrievable
        retrieved = self.storage.get("test_collection", key)
        self.assertIsNone(retrieved)
    
    def test_delete_nonexistent(self):
        """Test deleting nonexistent data."""
        # Try to delete nonexistent data
        result = self.storage.delete("test_collection", "nonexistent_key")
        
        # Check that delete fails
        self.assertFalse(result)
    
    def test_query(self):
        """Test querying data."""
        # Save multiple items
        self.storage.save("test_collection", {"type": "a", "value": 1})
        self.storage.save("test_collection", {"type": "b", "value": 2})
        self.storage.save("test_collection", {"type": "a", "value": 3})
        
        # Query for type "a"
        results = self.storage.query("test_collection", {"type": "a"})
        
        # Check that query returned the correct items
        self.assertEqual(len(results), 2)
        
        # Check that all results have type "a"
        for result in results:
            self.assertEqual(result["type"], "a")
    
    def test_query_with_limit(self):
        """Test querying data with a limit."""
        # Save multiple items
        for i in range(5):
            self.storage.save("test_collection", {"index": i})
        
        # Query with limit
        results = self.storage.query("test_collection", {}, limit=2)
        
        # Check that query returned the correct number of items
        self.assertEqual(len(results), 2)
    
    def test_close(self):
        """Test closing the storage."""
        # Close the storage
        self.storage.close()
        
        # Check that storage is no longer initialized
        self.assertFalse(self.storage.initialized)
        self.assertEqual(self.storage.collections, {})


class TestSqliteStorage(unittest.TestCase):
    """Test the SqliteStorage implementation."""
    
    def setUp(self):
        """Set up a SQLite storage for each test."""
        # Create a temporary directory for the database
        self.temp_dir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.temp_dir.name, "test.db")
        
        # Create storage
        self.storage = SqliteStorage(db_path)
        self.storage.initialize()
    
    def tearDown(self):
        """Clean up after each test."""
        self.storage.close()
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test that initialization creates the required tables."""
        self.assertTrue(self.storage.initialized)
        
        # Check that tables were created
        with self.storage._get_connection() as conn:
            cursor = conn.cursor()
            
            # Check storage_collections table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='storage_collections'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check storage_data table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='storage_data'")
            self.assertIsNotNone(cursor.fetchone())
    
    def test_save_and_get(self):
        """Test saving and retrieving data."""
        data = {"name": "test", "value": 42}
        key = self.storage.save("test_collection", data)
        
        # Get and check data
        retrieved = self.storage.get("test_collection", key)
        self.assertEqual(retrieved["name"], "test")
        self.assertEqual(retrieved["value"], 42)
    
    def test_update(self):
        """Test updating existing data."""
        # Save initial data
        data = {"name": "test", "value": 42}
        key = self.storage.save("test_collection", data)
        
        # Update the data
        updated_data = {"name": "test_updated", "value": 43}
        result = self.storage.update("test_collection", key, updated_data)
        
        # Check that update was successful
        self.assertTrue(result)
        
        # Get and check updated data
        retrieved = self.storage.get("test_collection", key)
        self.assertEqual(retrieved["name"], "test_updated")
        self.assertEqual(retrieved["value"], 43)
    
    def test_delete(self):
        """Test deleting data."""
        # Save initial data
        data = {"name": "test", "value": 42}
        key = self.storage.save("test_collection", data)
        
        # Delete the data
        result = self.storage.delete("test_collection", key)
        
        # Check that delete was successful
        self.assertTrue(result)
        
        # Check that data is no longer retrievable
        retrieved = self.storage.get("test_collection", key)
        self.assertIsNone(retrieved)
    
    def test_query(self):
        """Test querying data."""
        # Save multiple items
        self.storage.save("test_collection", {"type": "a", "value": 1})
        self.storage.save("test_collection", {"type": "b", "value": 2})
        self.storage.save("test_collection", {"type": "a", "value": 3})
        
        # Query for type "a"
        results = self.storage.query("test_collection", {"type": "a"})
        
        # Check that query returned the correct items
        self.assertEqual(len(results), 2)
        
        # Check that all results have type "a"
        for result in results:
            self.assertEqual(result["type"], "a")
    
    def test_connection_pool(self):
        """Test connection pooling."""
        # Check initial pool size
        self.assertEqual(len(self.storage.connection_pool), self.storage.pool_size)
        
        # Use a connection
        with self.storage._get_connection():
            # Check that pool size decreased
            self.assertEqual(len(self.storage.connection_pool), self.storage.pool_size - 1)
        
        # Check that connection was returned to pool
        self.assertEqual(len(self.storage.connection_pool), self.storage.pool_size)


class TestStorageManager(unittest.TestCase):
    """Test the StorageManager class."""
    
    def setUp(self):
        """Set up a storage manager with memory storage."""
        self.adapter = MemoryStorage()
        self.manager = StorageManager(self.adapter)
    
    def test_save_and_get_forecast(self):
        """Test saving and retrieving forecasts."""
        forecast_data = {
            "question": "Will AI impact jobs?",
            "probability": 0.75,
            "reasoning": "Based on current trends..."
        }
        
        # Save forecast
        forecast_id = self.manager.save_forecast(forecast_data)
        
        # Get forecast
        retrieved = self.manager.get_forecast(forecast_id)
        
        # Check data
        self.assertEqual(retrieved["question"], "Will AI impact jobs?")
        self.assertEqual(retrieved["probability"], 0.75)
        self.assertEqual(retrieved["reasoning"], "Based on current trends...")
    
    def test_save_and_get_research(self):
        """Test saving and retrieving research."""
        research_data = {
            "topic": "AI impact on jobs",
            "sources": ["paper1", "paper2"],
            "findings": "The impact is significant..."
        }
        
        # Save research
        research_id = self.manager.save_research(research_data)
        
        # Get research
        retrieved = self.manager.get_research(research_id)
        
        # Check data
        self.assertEqual(retrieved["topic"], "AI impact on jobs")
        self.assertEqual(retrieved["sources"], ["paper1", "paper2"])
        self.assertEqual(retrieved["findings"], "The impact is significant...")
    
    def test_save_user_interaction(self):
        """Test saving user interactions."""
        interaction_data = {
            "user_id": "user123",
            "action": "view_forecast",
            "forecast_id": "forecast123",
            "timestamp": datetime.now().isoformat()
        }
        
        # Save interaction
        interaction_id = self.manager.save_user_interaction(interaction_data)
        
        # Check that interaction was saved
        self.assertIsNotNone(interaction_id)
    
    def test_query_forecasts(self):
        """Test querying forecasts."""
        # Save multiple forecasts
        self.manager.save_forecast({"domain": "AI", "probability": 0.75})
        self.manager.save_forecast({"domain": "Economics", "probability": 0.6})
        self.manager.save_forecast({"domain": "AI", "probability": 0.8})
        
        # Query for AI forecasts
        results = self.manager.query_forecasts({"domain": "AI"})
        
        # Check results
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertEqual(result["domain"], "AI")
    
    def test_close(self):
        """Test closing the storage manager."""
        # Spy on adapter's close method
        original_close = self.adapter.close
        close_called = False
        
        def spy_close():
            nonlocal close_called
            close_called = True
            original_close()
        
        self.adapter.close = spy_close
        
        # Close the manager
        self.manager.close()
        
        # Check that adapter's close was called
        self.assertTrue(close_called)


if __name__ == "__main__":
    unittest.main() 