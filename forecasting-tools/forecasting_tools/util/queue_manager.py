"""
Job Queue System

This module provides a job queue system for handling forecast generation
requests and background research tasks. It supports task prioritization,
concurrency limits, and error handling with retries.
"""

import threading
import time
import queue
import logging
import traceback
import uuid
import asyncio
import json
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, TypeVar, Generic
from enum import Enum, auto
from datetime import datetime, timedelta
import concurrent.futures
from contextlib import contextmanager
import signal

# Define job status enum
class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

# Define job types
class JobType(Enum):
    FORECAST = "forecast"
    RESEARCH = "research"
    DATA_EXPORT = "data_export"
    CACHE_CLEANUP = "cache_cleanup"
    NOTIFICATION = "notification"

# Define job priority levels
class JobPriority(Enum):
    LOW = 0
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20

# Type definition for job handlers
T = TypeVar('T')
JobResult = TypeVar('JobResult')
JobHandler = Callable[[Dict[str, Any], "Job"], Any]
AsyncJobHandler = Callable[[Dict[str, Any], "Job"], asyncio.coroutine]

class Job:
    """
    Represents a job in the queue system.
    
    Tracks job status, data, retries, and execution details.
    """
    
    def __init__(
        self,
        job_type: Union[str, JobType],
        data: Dict[str, Any],
        priority: Union[int, JobPriority] = JobPriority.NORMAL,
        job_id: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 5,
        timeout: Optional[int] = None
    ):
        self.job_id = job_id or str(uuid.uuid4())
        self.job_type = job_type.value if isinstance(job_type, JobType) else job_type
        self.priority = priority.value if isinstance(priority, JobPriority) else priority
        self.data = data.copy() if data else {}
        self.status = JobStatus.PENDING
        
        # Tracking fields
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.started_at = None
        self.completed_at = None
        self.retries = 0
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.next_retry_at = None
        self.error = None
        self.result = None
        self.timeout = timeout
        
        # Job cancellation flag
        self._cancelled = False
        self._cancel_event = threading.Event()
    
    def mark_running(self) -> None:
        """Mark job as running."""
        self.status = JobStatus.RUNNING
        self.started_at = datetime.now()
        self.updated_at = self.started_at
    
    def mark_completed(self, result: Any = None) -> None:
        """Mark job as completed with optional result."""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.now()
        self.updated_at = self.completed_at
        self.result = result
    
    def mark_failed(self, error: Union[str, Exception]) -> None:
        """Mark job as failed with error details."""
        self.status = JobStatus.FAILED
        self.completed_at = datetime.now()
        self.updated_at = self.completed_at
        if isinstance(error, Exception):
            self.error = f"{type(error).__name__}: {str(error)}\n{traceback.format_exc()}"
        else:
            self.error = str(error)
    
    def schedule_retry(self) -> bool:
        """
        Schedule job for retry if retries are available.
        
        Returns:
            True if retry scheduled, False if max retries reached
        """
        if self.retries >= self.max_retries:
            return False
        
        self.retries += 1
        self.status = JobStatus.RETRYING
        self.next_retry_at = datetime.now() + timedelta(seconds=self.retry_delay * self.retries)
        self.updated_at = datetime.now()
        return True
    
    def cancel(self) -> None:
        """Cancel the job."""
        self._cancelled = True
        self._cancel_event.set()
        if self.status in (JobStatus.PENDING, JobStatus.RETRYING):
            self.status = JobStatus.CANCELLED
            self.updated_at = datetime.now()
    
    def is_cancelled(self) -> bool:
        """Check if job is cancelled."""
        return self._cancelled
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for storage/serialization."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "priority": self.priority,
            "data": self.data,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "next_retry_at": self.next_retry_at.isoformat() if self.next_retry_at else None,
            "error": self.error,
            "result": self.result,
            "timeout": self.timeout,
        }


class PriorityJobQueue:
    """
    Priority-based job queue implementation.
    """
    
    def __init__(self):
        self._queue = queue.PriorityQueue()
        self._jobs = {}  # job_id -> Job mapping
        self._lock = threading.RLock()
    
    def put(self, job: Job) -> None:
        """Add job to queue with priority handling."""
        with self._lock:
            # Store job in jobs dict
            self._jobs[job.job_id] = job
            
            # Add to priority queue (lower number = higher priority)
            # Use created_at as tiebreaker for same priority
            priority_tuple = (-job.priority, job.created_at.timestamp(), job.job_id)
            self._queue.put((priority_tuple, job.job_id))
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Job]:
        """Get the next job from queue based on priority."""
        try:
            priority_tuple, job_id = self._queue.get(block=block, timeout=timeout)
            with self._lock:
                job = self._jobs.get(job_id)
                return job
        except queue.Empty:
            return None
    
    def task_done(self, job_id: str) -> None:
        """Mark a job as done."""
        self._queue.task_done()
    
    def cancel(self, job_id: str) -> bool:
        """
        Cancel a pending job.
        
        Returns:
            True if job was found and cancelled, False otherwise
        """
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                job.cancel()
                return True
            return False
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID without removing from queue."""
        with self._lock:
            return self._jobs.get(job_id)
    
    def remove_completed(self) -> None:
        """Remove completed jobs from the jobs dictionary."""
        with self._lock:
            completed_ids = [
                job_id for job_id, job in self._jobs.items() 
                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
            ]
            for job_id in completed_ids:
                self._jobs.pop(job_id, None)


class QueueManager:
    """
    Manager for handling job queues and workers.
    
    Manages multiple job queues for different job types, handles prioritization,
    concurrency, retries, and timeout monitoring.
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        max_queue_size: int = 1000,
        default_timeout: int = 300
    ):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.default_timeout = default_timeout
        
        # Initialize queues for different job types
        self.queues = {
            job_type.value: PriorityJobQueue()
            for job_type in JobType
        }
        
        # Job handlers
        self.handlers = {}
        self.async_handlers = {}
        
        # Worker threads
        self.workers = []
        self.worker_shutdown = threading.Event()
        
        # Timeout monitoring
        self.timeout_monitor = None
        self.running_jobs = {}  # job_id -> (worker_thread, start_time, timeout)
        self.running_jobs_lock = threading.RLock()
        
        # Retry queue
        self.retry_queue = []
        self.retry_lock = threading.RLock()
        self.retry_event = threading.Event()
        
        # Logging
        self.logger = logging.getLogger("queue_manager")
    
    def register_handler(
        self,
        job_type: Union[str, JobType],
        handler: JobHandler,
        is_async: bool = False
    ) -> None:
        """
        Register a handler for a specific job type.
        
        Args:
            job_type: Type of job the handler processes
            handler: Function to call for processing the job
            is_async: Whether the handler is an async function
        """
        job_type_value = job_type.value if isinstance(job_type, JobType) else job_type
        
        if is_async:
            self.async_handlers[job_type_value] = handler
        else:
            self.handlers[job_type_value] = handler
    
    def enqueue(
        self,
        job_type: Union[str, JobType],
        data: Dict[str, Any],
        priority: Union[int, JobPriority] = JobPriority.NORMAL,
        job_id: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 5,
        timeout: Optional[int] = None
    ) -> str:
        """
        Add a job to the queue.
        
        Args:
            job_type: Type of job
            data: Job data
            priority: Job priority
            job_id: Optional job ID (generated if not provided)
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            timeout: Job timeout in seconds
            
        Returns:
            Job ID
        """
        job_type_value = job_type.value if isinstance(job_type, JobType) else job_type
        
        # Ensure job type has a queue
        if job_type_value not in self.queues:
            self.queues[job_type_value] = PriorityJobQueue()
        
        # Create job
        job = Job(
            job_type=job_type_value,
            data=data,
            priority=priority,
            job_id=job_id,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout or self.default_timeout
        )
        
        # Add to queue
        self.queues[job_type_value].put(job)
        
        self.logger.info(f"Enqueued job {job.job_id} of type {job_type_value} with priority {job.priority}")
        
        return job.job_id
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get job status by ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status dictionary or None if not found
        """
        for queue in self.queues.values():
            job = queue.get_job(job_id)
            if job:
                return job.to_dict()
        
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job by ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            True if job was found and cancelled, False otherwise
        """
        for queue in self.queues.values():
            if queue.cancel(job_id):
                self.logger.info(f"Cancelled job {job_id}")
                return True
        
        # Check if job is running
        with self.running_jobs_lock:
            if job_id in self.running_jobs:
                job = self.running_jobs[job_id][0]
                job.cancel()
                self.logger.info(f"Cancelled running job {job_id}")
                return True
        
        return False
    
    def start(self) -> None:
        """Start the queue manager workers and monitoring threads."""
        if self.workers:
            self.logger.warning("Queue manager already started")
            return
        
        self.worker_shutdown.clear()
        
        # Create and start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"QueueWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        # Start timeout monitor
        self.timeout_monitor = threading.Thread(
            target=self._timeout_monitor_loop,
            name="TimeoutMonitor",
            daemon=True
        )
        self.timeout_monitor.start()
        
        # Start retry monitor
        self.retry_handler = threading.Thread(
            target=self._retry_handler_loop,
            name="RetryHandler",
            daemon=True
        )
        self.retry_handler.start()
        
        self.logger.info(f"Queue manager started with {self.max_workers} workers")
    
    def stop(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """
        Stop the queue manager.
        
        Args:
            wait: Whether to wait for workers to finish
            timeout: Maximum time to wait for workers
        """
        self.logger.info("Stopping queue manager")
        self.worker_shutdown.set()
        self.retry_event.set()
        
        if wait:
            for worker in self.workers:
                worker.join(timeout=timeout)
            
            if self.timeout_monitor:
                self.timeout_monitor.join(timeout=timeout)
            
            if self.retry_handler:
                self.retry_handler.join(timeout=timeout)
        
        self.workers = []
        self.timeout_monitor = None
        self.retry_handler = None
        self.logger.info("Queue manager stopped")
    
    def _worker_loop(self) -> None:
        """Worker loop for processing jobs from queues."""
        while not self.worker_shutdown.is_set():
            try:
                # Check all queues based on priority order
                job = None
                for job_type in sorted(
                    self.queues.keys(),
                    key=lambda jt: -1 * JobPriority[jt.upper()].value if jt.upper() in JobPriority.__members__ else 0
                ):
                    queue = self.queues[job_type]
                    job = queue.get(block=False)
                    if job:
                        break
                
                if not job:
                    # No jobs found, sleep and try again
                    time.sleep(0.1)
                    continue
                
                # Process the job
                self._process_job(job)
                queue.task_done(job.job_id)
                
            except queue.Empty:
                # No jobs available, sleep briefly
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error in worker loop: {str(e)}")
                time.sleep(1)  # Sleep on error to prevent tight error loops
    
    def _process_job(self, job: Job) -> None:
        """
        Process a single job with error handling and timeout tracking.
        
        Args:
            job: Job to process
        """
        if job.is_cancelled():
            self.logger.info(f"Skipping cancelled job {job.job_id}")
            return
        
        try:
            # Mark job as running
            job.mark_running()
            
            # Register for timeout monitoring
            with self.running_jobs_lock:
                self.running_jobs[job.job_id] = (job, time.time(), job.timeout)
            
            handler = None
            is_async = False
            
            # Find the appropriate handler
            if job.job_type in self.handlers:
                handler = self.handlers[job.job_type]
            elif job.job_type in self.async_handlers:
                handler = self.async_handlers[job.job_type]
                is_async = True
            
            if handler is None:
                job.mark_failed(f"No handler registered for job type: {job.job_type}")
                self.logger.error(f"No handler for job type: {job.job_type}, job ID: {job.job_id}")
                return
            
            # Execute the handler
            self.logger.info(f"Processing job {job.job_id} of type {job.job_type}")
            
            if is_async:
                # Run async handler in event loop
                result = asyncio.run(handler(job.data, job))
            else:
                # Run sync handler directly
                result = handler(job.data, job)
            
            # Mark job as completed with result
            if not job.is_cancelled():
                job.mark_completed(result)
                self.logger.info(f"Completed job {job.job_id}")
        
        except Exception as e:
            # Handle job failure
            self.logger.error(f"Error processing job {job.job_id}: {str(e)}")
            job.mark_failed(e)
            
            # Schedule retry if applicable
            if job.schedule_retry():
                self.logger.info(f"Scheduled retry for job {job.job_id}, attempt {job.retries}/{job.max_retries}")
                with self.retry_lock:
                    self.retry_queue.append(job)
                    self.retry_event.set()
        
        finally:
            # Remove from running jobs
            with self.running_jobs_lock:
                self.running_jobs.pop(job.job_id, None)
    
    def _timeout_monitor_loop(self) -> None:
        """Monitor for job timeouts."""
        while not self.worker_shutdown.is_set():
            try:
                current_time = time.time()
                timed_out_jobs = []
                
                # Check for timed out jobs
                with self.running_jobs_lock:
                    for job_id, (job, start_time, timeout) in self.running_jobs.items():
                        if timeout and current_time - start_time > timeout:
                            timed_out_jobs.append(job_id)
                
                # Handle timed out jobs
                for job_id in timed_out_jobs:
                    with self.running_jobs_lock:
                        if job_id in self.running_jobs:
                            job, _, _ = self.running_jobs.pop(job_id)
                            
                            # Mark job as failed with timeout error
                            job.mark_failed(f"Job timed out after {job.timeout} seconds")
                            self.logger.warning(f"Job {job_id} timed out")
                            
                            # Schedule retry if applicable
                            if job.schedule_retry():
                                self.logger.info(f"Scheduled retry for timed out job {job.job_id}")
                                with self.retry_lock:
                                    self.retry_queue.append(job)
                                    self.retry_event.set()
                
                # Sleep before next check
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in timeout monitor: {str(e)}")
                time.sleep(5)
    
    def _retry_handler_loop(self) -> None:
        """Handle job retries."""
        while not self.worker_shutdown.is_set():
            try:
                # Wait for retry event or check periodically
                self.retry_event.wait(timeout=10)
                self.retry_event.clear()
                
                current_time = datetime.now()
                retry_jobs = []
                
                # Get jobs ready for retry
                with self.retry_lock:
                    ready_jobs = []
                    
                    for job in self.retry_queue:
                        if job.next_retry_at and job.next_retry_at <= current_time:
                            ready_jobs.append(job)
                    
                    # Remove ready jobs from retry queue
                    self.retry_queue = [j for j in self.retry_queue if j not in ready_jobs]
                    retry_jobs = ready_jobs.copy()
                
                # Requeue jobs ready for retry
                for job in retry_jobs:
                    if not job.is_cancelled():
                        self.logger.info(f"Retrying job {job.job_id}, attempt {job.retries}/{job.max_retries}")
                        
                        # Get the appropriate queue for this job type
                        queue = self.queues.get(job.job_type)
                        if queue:
                            queue.put(job)
                        else:
                            # Create queue if it doesn't exist
                            self.queues[job.job_type] = PriorityJobQueue()
                            self.queues[job.job_type].put(job)
                
            except Exception as e:
                self.logger.error(f"Error in retry handler: {str(e)}")
                time.sleep(5)


# Context manager for handling graceful shutdown
@contextmanager
def queue_manager_context(
    max_workers: int = 4,
    max_queue_size: int = 1000,
    default_timeout: int = 300
) -> QueueManager:
    """
    Context manager for QueueManager to ensure proper cleanup.
    
    Usage:
        with queue_manager_context() as manager:
            manager.register_handler(JobType.FORECAST, forecast_handler)
            job_id = manager.enqueue(JobType.FORECAST, data)
    """
    manager = QueueManager(
        max_workers=max_workers,
        max_queue_size=max_queue_size,
        default_timeout=default_timeout
    )
    
    # Setup signal handlers for graceful shutdown
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    
    def signal_handler(signum, frame):
        logging.info(f"Received signal {signum}, shutting down queue manager")
        manager.stop(wait=True, timeout=10)
        
        # Restore original handlers and re-raise
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        if signum == signal.SIGINT:
            raise KeyboardInterrupt()
        else:
            sys.exit(0)
    
    try:
        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start manager
        manager.start()
        yield manager
    finally:
        # Stop manager on exit
        manager.stop(wait=True, timeout=10)
        
        # Restore original signal handlers
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm) 