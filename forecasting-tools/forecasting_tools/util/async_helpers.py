"""
Asynchronous Processing Utilities

This module provides asynchronous processing utilities for handling multiple 
API calls simultaneously, implementing rate limiting, retries, and 
concurrent batch processing.
"""

import asyncio
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar, Union, Generic
from functools import wraps
import logging
import aiohttp
import backoff
from concurrent.futures import ThreadPoolExecutor

T = TypeVar('T')
R = TypeVar('R')

class RateLimiter:
    """Rate limiter for API calls to prevent exceeding API rate limits."""
    
    def __init__(self, calls_per_second: float = 1.0, max_concurrent: int = 10):
        self.calls_per_second = calls_per_second
        self.interval = 1.0 / calls_per_second
        self.max_concurrent = max_concurrent
        self.last_call_time = 0.0
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make an API call."""
        async with self._lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            if time_since_last_call < self.interval:
                await asyncio.sleep(self.interval - time_since_last_call)
            self.last_call_time = time.time()
        
        await self.semaphore.acquire()
    
    def release(self):
        """Release the semaphore after the API call is complete."""
        self.semaphore.release()

class APIClient:
    """Asynchronous API client with rate limiting and retries."""
    
    def __init__(
        self, 
        base_url: str, 
        headers: Optional[Dict[str, str]] = None,
        rate_limit: float = 1.0,
        max_concurrent: int = 10,
        timeout: int = 60,
        max_retries: int = 3
    ):
        self.base_url = base_url
        self.headers = headers or {}
        self.rate_limiter = RateLimiter(rate_limit, max_concurrent)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.session = None
    
    async def __aenter__(self):
        """Create session when entering context manager."""
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=self.timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close session when exiting context manager."""
        if self.session:
            await self.session.close()
            self.session = None
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        giveup=lambda e: isinstance(e, aiohttp.ClientResponseError) and e.status not in (429, 500, 502, 503, 504)
    )
    async def request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make an API request with rate limiting and retries."""
        if self.session is None:
            raise RuntimeError("APIClient must be used as a context manager")
        
        try:
            await self.rate_limiter.acquire()
            
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            async with self.session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()
        finally:
            self.rate_limiter.release()

class BatchProcessor(Generic[T, R]):
    """Process items in batches with concurrent execution."""
    
    def __init__(
        self, 
        batch_size: int = 10,
        max_concurrency: int = 5,
        timeout: Optional[float] = None
    ):
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.timeout = timeout
    
    async def process(
        self,
        items: List[T],
        processor: Callable[[T], Coroutine[Any, Any, R]]
    ) -> List[R]:
        """
        Process a list of items in batches.
        
        Args:
            items: List of items to process
            processor: Async function to process each item
            
        Returns:
            List of results
        """
        results = []
        batches = [items[i:i+self.batch_size] for i in range(0, len(items), self.batch_size)]
        
        for batch in batches:
            batch_tasks = [processor(item) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logging.error(f"Error processing batch item: {result}")
                else:
                    results.append(result)
        
        return results

async def with_timeout(coro: Coroutine, timeout: float, default: Any = None) -> Any:
    """
    Execute coroutine with a timeout.
    
    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        default: Default value to return if timeout occurs
        
    Returns:
        Result of coroutine or default value if timeout occurs
    """
    try:
        return await asyncio.wait_for(coro, timeout)
    except asyncio.TimeoutError:
        logging.warning(f"Operation timed out after {timeout} seconds")
        return default

def run_async_in_thread(coro: Coroutine) -> Any:
    """
    Run an async coroutine in a separate thread from synchronous code.
    
    Args:
        coro: Coroutine to run
        
    Returns:
        Result of the coroutine
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()

def async_to_sync(func):
    """
    Decorator to convert an async function to a sync function.
    
    Args:
        func: Async function to convert
        
    Returns:
        Synchronous wrapper function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return run_async_in_thread(func(*args, **kwargs))
    return wrapper 