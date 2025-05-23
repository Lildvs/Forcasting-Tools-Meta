import unittest
import asyncio
import time
import aiohttp
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from forecasting_tools.util.async_helpers import (
    RateLimiter, APIClient, BatchProcessor, 
    timeout, run_async, gather_with_concurrency
)


class TestRateLimiter(unittest.TestCase):
    """Tests for the RateLimiter class."""
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test that rate limiter enforces limits."""
        # Create a rate limiter that allows 5 calls per second
        rate_limiter = RateLimiter(max_calls=5, time_period=1.0)
        
        # Track call times
        call_times = []
        
        # Make 10 calls (should rate limit after 5)
        for _ in range(10):
            async with rate_limiter:
                call_times.append(time.time())
        
        # Check that the first 5 calls were made quickly
        if len(call_times) >= 6:
            # Time between 5th and 6th call should be ~1 second
            time_diff = call_times[5] - call_times[4]
            self.assertGreaterEqual(time_diff, 0.9, 
                                   "Rate limiter should enforce delay")
    
    @pytest.mark.asyncio
    async def test_concurrent_rate_limiting(self):
        """Test rate limiter with concurrent requests."""
        # Create a rate limiter that allows 3 calls per second
        rate_limiter = RateLimiter(max_calls=3, time_period=1.0)
        
        # Create a task that uses the rate limiter
        async def limited_task(task_id):
            async with rate_limiter:
                return task_id
        
        # Launch 6 tasks concurrently
        tasks = [limited_task(i) for i in range(6)]
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Should take at least 1 second due to rate limiting
        self.assertGreaterEqual(end_time - start_time, 1.0)
        self.assertEqual(results, list(range(6)))


class TestAPIClient(unittest.TestCase):
    """Tests for the APIClient class."""
    
    @patch('aiohttp.ClientSession')
    @pytest.mark.asyncio
    async def test_get_request(self, mock_session):
        """Test GET request with the API client."""
        # Set up mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"data": "test"}
        mock_response.__aenter__.return_value = mock_response
        
        # Set up mock session
        mock_session_instance = MagicMock()
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value.__aenter__.return_value = mock_session_instance
        
        # Create client and make request
        client = APIClient(base_url="https://api.example.com")
        result = await client.get("/test")
        
        # Check request was made correctly
        mock_session_instance.get.assert_called_once_with(
            "https://api.example.com/test",
            headers={},
            params=None,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        # Check result
        self.assertEqual(result, {"data": "test"})
    
    @patch('aiohttp.ClientSession')
    @pytest.mark.asyncio
    async def test_post_request(self, mock_session):
        """Test POST request with the API client."""
        # Set up mock response
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.json.return_value = {"id": 123}
        mock_response.__aenter__.return_value = mock_response
        
        # Set up mock session
        mock_session_instance = MagicMock()
        mock_session_instance.post.return_value = mock_response
        mock_session.return_value.__aenter__.return_value = mock_session_instance
        
        # Create client and make request
        client = APIClient(base_url="https://api.example.com")
        data = {"name": "test"}
        result = await client.post("/resources", json=data)
        
        # Check request was made correctly
        mock_session_instance.post.assert_called_once_with(
            "https://api.example.com/resources",
            headers={},
            json=data,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        # Check result
        self.assertEqual(result, {"id": 123})
    
    @patch('aiohttp.ClientSession')
    @pytest.mark.asyncio
    async def test_retry_on_error(self, mock_session):
        """Test that the client retries on error."""
        # Set up mock responses - first fails, second succeeds
        mock_error_response = AsyncMock()
        mock_error_response.status = 500
        mock_error_response.__aenter__.return_value = mock_error_response
        
        mock_success_response = AsyncMock()
        mock_success_response.status = 200
        mock_success_response.json.return_value = {"data": "test"}
        mock_success_response.__aenter__.return_value = mock_success_response
        
        # Set up mock session
        mock_session_instance = MagicMock()
        mock_session_instance.get.side_effect = [mock_error_response, mock_success_response]
        mock_session.return_value.__aenter__.return_value = mock_session_instance
        
        # Create client with retry and make request
        client = APIClient(base_url="https://api.example.com", retry_attempts=2)
        result = await client.get("/test")
        
        # Check get was called twice
        self.assertEqual(mock_session_instance.get.call_count, 2)
        
        # Check result is from second call
        self.assertEqual(result, {"data": "test"})
    
    @patch('aiohttp.ClientSession')
    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_session):
        """Test that the client handles timeouts."""
        # Mock a timeout error
        mock_session_instance = MagicMock()
        mock_session_instance.get.side_effect = asyncio.TimeoutError()
        mock_session.return_value.__aenter__.return_value = mock_session_instance
        
        # Create client and make request
        client = APIClient(base_url="https://api.example.com", timeout=1)
        
        # Should raise the timeout error
        with self.assertRaises(asyncio.TimeoutError):
            await client.get("/test")


class TestBatchProcessor(unittest.TestCase):
    """Tests for the BatchProcessor class."""
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test processing items in batches."""
        # Define a simple processing function
        async def process_batch(batch):
            return [item * 2 for item in batch]
        
        # Create a batch processor
        processor = BatchProcessor(batch_size=3, process_fn=process_batch)
        
        # Process items
        items = [1, 2, 3, 4, 5, 6, 7, 8]
        result = await processor.process_items(items)
        
        # Check result
        expected = [2, 4, 6, 8, 10, 12, 14, 16]
        self.assertEqual(result, expected)
    
    @pytest.mark.asyncio
    async def test_partial_batch(self):
        """Test processing with a partial final batch."""
        # Define a simple processing function
        async def process_batch(batch):
            return [item * 2 for item in batch]
        
        # Create a batch processor
        processor = BatchProcessor(batch_size=3, process_fn=process_batch)
        
        # Process items with a partial batch
        items = [1, 2, 3, 4, 5, 6, 7]  # Last batch has only 1 item
        result = await processor.process_items(items)
        
        # Check result
        expected = [2, 4, 6, 8, 10, 12, 14]
        self.assertEqual(result, expected)
    
    @pytest.mark.asyncio
    async def test_empty_input(self):
        """Test processing with empty input."""
        # Define a simple processing function
        async def process_batch(batch):
            return [item * 2 for item in batch]
        
        # Create a batch processor
        processor = BatchProcessor(batch_size=3, process_fn=process_batch)
        
        # Process empty list
        result = await processor.process_items([])
        
        # Check result
        self.assertEqual(result, [])
    
    @pytest.mark.asyncio
    async def test_batch_with_concurrency(self):
        """Test batch processing with concurrency limit."""
        # Define a processing function with delay
        async def process_batch(batch):
            await asyncio.sleep(0.1)
            return [item * 2 for item in batch]
        
        # Create a batch processor with concurrency limit
        processor = BatchProcessor(
            batch_size=2, 
            process_fn=process_batch,
            max_concurrency=2
        )
        
        # Process items
        items = [1, 2, 3, 4, 5, 6, 7, 8]
        start_time = time.time()
        result = await processor.process_items(items)
        end_time = time.time()
        
        # Should take at least 0.2 seconds (4 batches, 2 at a time)
        self.assertGreaterEqual(end_time - start_time, 0.2)
        
        # Check result
        expected = [2, 4, 6, 8, 10, 12, 14, 16]
        self.assertEqual(result, expected)


class TestTimeoutDecorator(unittest.TestCase):
    """Tests for the timeout decorator."""
    
    @pytest.mark.asyncio
    async def test_timeout_not_exceeded(self):
        """Test that the function completes when timeout not exceeded."""
        # Define a function that completes quickly
        @timeout(1.0)
        async def quick_function():
            await asyncio.sleep(0.1)
            return "success"
        
        # Call the function
        result = await quick_function()
        
        # Check result
        self.assertEqual(result, "success")
    
    @pytest.mark.asyncio
    async def test_timeout_exceeded(self):
        """Test that timeout is raised when function takes too long."""
        # Define a function that takes too long
        @timeout(0.1)
        async def slow_function():
            await asyncio.sleep(1.0)
            return "success"
        
        # Call the function - should timeout
        with self.assertRaises(asyncio.TimeoutError):
            await slow_function()


class TestRunAsync(unittest.TestCase):
    """Tests for the run_async function."""
    
    def test_run_async_function(self):
        """Test running an async function from sync code."""
        # Define an async function
        async def async_function():
            await asyncio.sleep(0.1)
            return "success"
        
        # Run it with run_async
        result = run_async(async_function())
        
        # Check result
        self.assertEqual(result, "success")


class TestGatherWithConcurrency(unittest.TestCase):
    """Tests for the gather_with_concurrency function."""
    
    @pytest.mark.asyncio
    async def test_gather_with_concurrency(self):
        """Test gathering tasks with concurrency limit."""
        # Create a list to track when tasks are running
        running_tasks = set()
        max_running = 0
        
        # Define a task that tracks concurrency
        async def tracked_task(task_id):
            running_tasks.add(task_id)
            nonlocal max_running
            max_running = max(max_running, len(running_tasks))
            await asyncio.sleep(0.1)
            running_tasks.remove(task_id)
            return task_id
        
        # Create 10 tasks and gather with concurrency 3
        tasks = [tracked_task(i) for i in range(10)]
        results = await gather_with_concurrency(3, *tasks)
        
        # Check results are correct
        self.assertEqual(results, list(range(10)))
        
        # Check concurrency was respected
        self.assertLessEqual(max_running, 3,
                           "Should not exceed concurrency limit")


if __name__ == "__main__":
    unittest.main() 