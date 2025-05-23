import unittest
import time
import threading
import asyncio
from datetime import datetime, timedelta

from forecasting_tools.util.queue_manager import (
    Job, JobStatus, JobType, JobPriority, 
    PriorityJobQueue, QueueManager
)


class TestJob(unittest.TestCase):
    """Tests for the Job class."""
    
    def test_job_initialization(self):
        """Test job initialization with default values."""
        job_data = {"param1": "value1", "param2": 42}
        job = Job(JobType.FORECAST, job_data)
        
        # Check that job ID was generated
        self.assertIsNotNone(job.job_id)
        
        # Check that job properties were set correctly
        self.assertEqual(job.job_type, JobType.FORECAST.value)
        self.assertEqual(job.priority, JobPriority.NORMAL.value)
        self.assertEqual(job.data, job_data)
        self.assertEqual(job.status, JobStatus.PENDING)
        self.assertEqual(job.retries, 0)
        self.assertEqual(job.max_retries, 3)
        self.assertIsNone(job.result)
        
        # Check that timestamps were set
        self.assertIsNotNone(job.created_at)
        self.assertIsNotNone(job.updated_at)
        self.assertIsNone(job.started_at)
        self.assertIsNone(job.completed_at)
    
    def test_job_initialization_with_custom_values(self):
        """Test job initialization with custom values."""
        job_data = {"param1": "value1", "param2": 42}
        job_id = "custom_id"
        job = Job(
            JobType.RESEARCH, 
            job_data, 
            priority=JobPriority.HIGH,
            job_id=job_id,
            max_retries=5,
            retry_delay=10,
            timeout=60
        )
        
        # Check that job properties were set correctly
        self.assertEqual(job.job_id, job_id)
        self.assertEqual(job.job_type, JobType.RESEARCH.value)
        self.assertEqual(job.priority, JobPriority.HIGH.value)
        self.assertEqual(job.max_retries, 5)
        self.assertEqual(job.retry_delay, 10)
        self.assertEqual(job.timeout, 60)
    
    def test_mark_running(self):
        """Test marking a job as running."""
        job = Job(JobType.FORECAST, {})
        job.mark_running()
        
        # Check that job status was updated
        self.assertEqual(job.status, JobStatus.RUNNING)
        self.assertIsNotNone(job.started_at)
        self.assertEqual(job.updated_at, job.started_at)
    
    def test_mark_completed(self):
        """Test marking a job as completed."""
        job = Job(JobType.FORECAST, {})
        result = {"output": "forecast result"}
        job.mark_completed(result)
        
        # Check that job status was updated
        self.assertEqual(job.status, JobStatus.COMPLETED)
        self.assertIsNotNone(job.completed_at)
        self.assertEqual(job.updated_at, job.completed_at)
        self.assertEqual(job.result, result)
    
    def test_mark_failed(self):
        """Test marking a job as failed."""
        job = Job(JobType.FORECAST, {})
        error = "Test error message"
        job.mark_failed(error)
        
        # Check that job status was updated
        self.assertEqual(job.status, JobStatus.FAILED)
        self.assertIsNotNone(job.completed_at)
        self.assertEqual(job.updated_at, job.completed_at)
        self.assertEqual(job.error, error)
    
    def test_mark_failed_with_exception(self):
        """Test marking a job as failed with an exception."""
        job = Job(JobType.FORECAST, {})
        error = ValueError("Test error")
        job.mark_failed(error)
        
        # Check that job status was updated
        self.assertEqual(job.status, JobStatus.FAILED)
        self.assertIn("ValueError: Test error", job.error)
    
    def test_schedule_retry(self):
        """Test scheduling a job for retry."""
        job = Job(JobType.FORECAST, {}, max_retries=2)
        
        # First retry
        result = job.schedule_retry()
        self.assertTrue(result)
        self.assertEqual(job.retries, 1)
        self.assertEqual(job.status, JobStatus.RETRYING)
        self.assertIsNotNone(job.next_retry_at)
        
        # Second retry
        result = job.schedule_retry()
        self.assertTrue(result)
        self.assertEqual(job.retries, 2)
        
        # Third retry (should fail as max_retries is 2)
        result = job.schedule_retry()
        self.assertFalse(result)
        self.assertEqual(job.retries, 2)
    
    def test_cancel(self):
        """Test cancelling a job."""
        job = Job(JobType.FORECAST, {})
        job.cancel()
        
        # Check that job was cancelled
        self.assertTrue(job.is_cancelled())
        self.assertEqual(job.status, JobStatus.CANCELLED)
    
    def test_to_dict(self):
        """Test converting a job to a dictionary."""
        job = Job(JobType.FORECAST, {"param": "value"})
        job_dict = job.to_dict()
        
        # Check that dictionary contains expected fields
        self.assertEqual(job_dict["job_id"], job.job_id)
        self.assertEqual(job_dict["job_type"], job.job_type)
        self.assertEqual(job_dict["priority"], job.priority)
        self.assertEqual(job_dict["data"], job.data)
        self.assertEqual(job_dict["status"], job.status.value)


class TestPriorityJobQueue(unittest.TestCase):
    """Tests for the PriorityJobQueue class."""
    
    def setUp(self):
        """Set up a queue for each test."""
        self.queue = PriorityJobQueue()
    
    def test_put_and_get(self):
        """Test putting and getting jobs from the queue."""
        # Create and add jobs with different priorities
        job1 = Job(JobType.FORECAST, {}, priority=JobPriority.NORMAL)
        job2 = Job(JobType.RESEARCH, {}, priority=JobPriority.HIGH)
        job3 = Job(JobType.DATA_EXPORT, {}, priority=JobPriority.LOW)
        
        self.queue.put(job1)
        self.queue.put(job2)
        self.queue.put(job3)
        
        # Get jobs from queue and check order (high to low priority)
        self.assertEqual(self.queue.get().job_id, job2.job_id)  # HIGH
        self.assertEqual(self.queue.get().job_id, job1.job_id)  # NORMAL
        self.assertEqual(self.queue.get().job_id, job3.job_id)  # LOW
    
    def test_get_nonblocking(self):
        """Test getting jobs from an empty queue in non-blocking mode."""
        # Queue is empty
        result = self.queue.get(block=False)
        self.assertIsNone(result)
    
    def test_cancel(self):
        """Test cancelling a job in the queue."""
        # Add a job
        job = Job(JobType.FORECAST, {})
        self.queue.put(job)
        
        # Cancel the job
        result = self.queue.cancel(job.job_id)
        self.assertTrue(result)
        self.assertTrue(job.is_cancelled())
        
        # Try to cancel a non-existent job
        result = self.queue.cancel("nonexistent_id")
        self.assertFalse(result)
    
    def test_get_job(self):
        """Test getting a job by ID without removing it from queue."""
        # Add a job
        job = Job(JobType.FORECAST, {})
        self.queue.put(job)
        
        # Get the job by ID
        retrieved = self.queue.get_job(job.job_id)
        self.assertEqual(retrieved.job_id, job.job_id)
        
        # Try to get a non-existent job
        retrieved = self.queue.get_job("nonexistent_id")
        self.assertIsNone(retrieved)
    
    def test_remove_completed(self):
        """Test removing completed jobs."""
        # Add jobs with different statuses
        job1 = Job(JobType.FORECAST, {})
        job2 = Job(JobType.RESEARCH, {})
        job3 = Job(JobType.DATA_EXPORT, {})
        
        self.queue.put(job1)
        self.queue.put(job2)
        self.queue.put(job3)
        
        # Mark jobs with different statuses
        job1.mark_completed()
        job2.mark_failed("Test error")
        # job3 remains PENDING
        
        # Remove completed jobs
        self.queue.remove_completed()
        
        # Check that completed and failed jobs were removed
        self.assertIsNone(self.queue.get_job(job1.job_id))
        self.assertIsNone(self.queue.get_job(job2.job_id))
        self.assertIsNotNone(self.queue.get_job(job3.job_id))


class TestQueueManager(unittest.TestCase):
    """Tests for the QueueManager class."""
    
    def setUp(self):
        """Set up a queue manager for each test."""
        self.manager = QueueManager(max_workers=2)
    
    def tearDown(self):
        """Clean up after each test."""
        self.manager.stop(wait=True)
    
    def test_enqueue_and_get_status(self):
        """Test enqueueing a job and getting its status."""
        # Enqueue a job without starting the manager
        job_data = {"param": "value"}
        job_id = self.manager.enqueue(JobType.FORECAST, job_data, priority=JobPriority.HIGH)
        
        # Get job status
        status = self.manager.get_job_status(job_id)
        
        # Check status
        self.assertIsNotNone(status)
        self.assertEqual(status["job_type"], JobType.FORECAST.value)
        self.assertEqual(status["priority"], JobPriority.HIGH.value)
        self.assertEqual(status["data"], job_data)
        self.assertEqual(status["status"], JobStatus.PENDING.value)
    
    def test_cancel_job(self):
        """Test cancelling a job."""
        # Enqueue a job without starting the manager
        job_id = self.manager.enqueue(JobType.FORECAST, {})
        
        # Cancel the job
        result = self.manager.cancel_job(job_id)
        self.assertTrue(result)
        
        # Get job status
        status = self.manager.get_job_status(job_id)
        self.assertEqual(status["status"], JobStatus.CANCELLED.value)
        
        # Try to cancel a non-existent job
        result = self.manager.cancel_job("nonexistent_id")
        self.assertFalse(result)
    
    def test_job_execution(self):
        """Test job execution."""
        # Define a test handler
        def test_handler(data, job):
            return {"input": data, "processed": True}
        
        # Register the handler
        self.manager.register_handler(JobType.FORECAST, test_handler)
        
        # Start the manager
        self.manager.start()
        
        # Enqueue a job
        job_data = {"param": "value"}
        job_id = self.manager.enqueue(JobType.FORECAST, job_data)
        
        # Wait for job to complete
        max_wait = 5.0  # seconds
        start_time = time.time()
        status = None
        
        while time.time() - start_time < max_wait:
            status = self.manager.get_job_status(job_id)
            if status and status["status"] in (JobStatus.COMPLETED.value, JobStatus.FAILED.value):
                break
            time.sleep(0.1)
        
        # Check that job completed successfully
        self.assertIsNotNone(status)
        self.assertEqual(status["status"], JobStatus.COMPLETED.value)
        self.assertEqual(status["result"]["input"], job_data)
        self.assertTrue(status["result"]["processed"])
    
    def test_async_job_execution(self):
        """Test async job execution."""
        # Define an async test handler
        async def async_test_handler(data, job):
            await asyncio.sleep(0.1)  # Simulate async work
            return {"input": data, "processed": True}
        
        # Register the handler
        self.manager.register_handler(JobType.RESEARCH, async_test_handler, is_async=True)
        
        # Start the manager
        self.manager.start()
        
        # Enqueue a job
        job_data = {"param": "async_value"}
        job_id = self.manager.enqueue(JobType.RESEARCH, job_data)
        
        # Wait for job to complete
        max_wait = 5.0  # seconds
        start_time = time.time()
        status = None
        
        while time.time() - start_time < max_wait:
            status = self.manager.get_job_status(job_id)
            if status and status["status"] in (JobStatus.COMPLETED.value, JobStatus.FAILED.value):
                break
            time.sleep(0.1)
        
        # Check that job completed successfully
        self.assertIsNotNone(status)
        self.assertEqual(status["status"], JobStatus.COMPLETED.value)
        self.assertEqual(status["result"]["input"], job_data)
        self.assertTrue(status["result"]["processed"])
    
    def test_job_failure_and_retry(self):
        """Test job failure and retry."""
        # Define a failing handler that succeeds on second attempt
        attempt_count = {}
        
        def failing_handler(data, job):
            job_id = job.job_id
            if job_id not in attempt_count:
                attempt_count[job_id] = 0
            
            attempt_count[job_id] += 1
            
            # Fail on first attempt
            if attempt_count[job_id] == 1:
                raise ValueError("Test failure")
            
            # Succeed on second attempt
            return {"attempt": attempt_count[job_id], "success": True}
        
        # Register the handler
        self.manager.register_handler(JobType.FORECAST, failing_handler)
        
        # Start the manager
        self.manager.start()
        
        # Enqueue a job
        job_id = self.manager.enqueue(
            JobType.FORECAST, 
            {}, 
            max_retries=1, 
            retry_delay=1
        )
        
        # Wait for job to complete
        max_wait = 10.0  # seconds
        start_time = time.time()
        status = None
        
        while time.time() - start_time < max_wait:
            status = self.manager.get_job_status(job_id)
            if status and status["status"] in (JobStatus.COMPLETED.value, JobStatus.FAILED.value):
                break
            time.sleep(0.1)
        
        # Check that job completed successfully after retry
        self.assertIsNotNone(status)
        self.assertEqual(status["status"], JobStatus.COMPLETED.value)
        self.assertEqual(status["result"]["attempt"], 2)
        self.assertTrue(status["result"]["success"])
    
    def test_job_priority(self):
        """Test job priority processing order."""
        # Define a slow handler to allow testing priority
        processed_jobs = []
        
        def slow_handler(data, job):
            time.sleep(0.1)  # Small delay to ensure priority is respected
            processed_jobs.append(job.job_id)
            return {"processed": True}
        
        # Register the handler
        self.manager.register_handler(JobType.FORECAST, slow_handler)
        
        # Start the manager with just 1 worker to ensure sequential processing
        manager = QueueManager(max_workers=1)
        manager.register_handler(JobType.FORECAST, slow_handler)
        manager.start()
        
        try:
            # Enqueue jobs with different priorities
            job1_id = manager.enqueue(JobType.FORECAST, {}, priority=JobPriority.NORMAL)
            job2_id = manager.enqueue(JobType.FORECAST, {}, priority=JobPriority.HIGH)
            job3_id = manager.enqueue(JobType.FORECAST, {}, priority=JobPriority.LOW)
            
            # Wait for jobs to complete
            max_wait = 10.0  # seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                if len(processed_jobs) == 3:
                    break
                time.sleep(0.1)
            
            # Check processing order
            self.assertEqual(processed_jobs[0], job2_id)  # HIGH should be first
            self.assertEqual(processed_jobs[1], job1_id)  # NORMAL should be second
            self.assertEqual(processed_jobs[2], job3_id)  # LOW should be last
        finally:
            manager.stop(wait=True)


if __name__ == "__main__":
    unittest.main() 