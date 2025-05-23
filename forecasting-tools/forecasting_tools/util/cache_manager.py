"""
Cache Manager

This module provides a simple caching mechanism for storing and retrieving data 
with time-to-live (TTL) functionality to prevent stale data.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union, Generic, TypeVar, cast

from forecasting_tools.config import CacheConfig

logger = logging.getLogger(__name__)

T = TypeVar('T')

class CacheManager(Generic[T]):
    """
    A generic cache manager that provides in-memory and persistent caching
    with time-to-live (TTL) functionality.
    
    Features:
    - Generic type for cached values
    - In-memory caching with optional persistence to disk
    - TTL (time-to-live) for cache entries
    - Thread-safe operations with asyncio locks
    - Background cache cleanup
    - Singleton pattern to ensure only one instance exists
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """Implement the Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(
        self, 
        cache_dir: Optional[Union[str, Path]] = None,
        default_ttl: Optional[int] = None,
        enable_persistence: Optional[bool] = None,
        cleanup_interval: Optional[int] = None
    ):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store persistent cache files
            default_ttl: Default time-to-live for cache entries in seconds
            enable_persistence: Whether to persist cache to disk
            cleanup_interval: How often to clean up expired cache entries
        """
        # Skip initialization if already initialized (singleton pattern)
        if CacheManager._initialized:
            return
        
        CacheManager._initialized = True
        
        # Use provided values or fall back to config
        self.default_ttl = default_ttl if default_ttl is not None else CacheConfig.DEFAULT_TTL
        self.enable_persistence = enable_persistence if enable_persistence is not None else CacheConfig.ENABLE_PERSISTENCE
        self.cleanup_interval = cleanup_interval if cleanup_interval is not None else CacheConfig.CLEANUP_INTERVAL
        
        # Set up cache directory
        if cache_dir is None:
            self.cache_dir = CacheConfig.CACHE_DIR
        else:
            self.cache_dir = Path(cache_dir)
        
        if self.enable_persistence:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize in-memory cache and locks
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_lock = asyncio.Lock()
        
        # Start background cleanup task
        if self.cleanup_interval > 0:
            self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start a background task to clean up expired cache entries."""
        async def cleanup_task():
            while True:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_expired()
        
        # Create task to run in the background
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(cleanup_task())
                logger.debug("Started cache cleanup background task")
        except Exception as e:
            logger.warning(f"Failed to start cache cleanup task: {e}")
    
    async def get(self, key: str, namespace: str = "default") -> Optional[T]:
        """
        Get a value from the cache by key and namespace.
        
        Args:
            key: The cache key
            namespace: The cache namespace
            
        Returns:
            The cached value, or None if not found or expired
        """
        async with self.cache_lock:
            # Check in-memory cache first
            namespace_cache = self.cache.get(namespace, {})
            cache_entry = namespace_cache.get(key)
            
            if cache_entry is not None:
                # Check if entry has expired
                expiration = cache_entry.get("expiration")
                if expiration is None or expiration > time.time():
                    return cast(T, cache_entry.get("value"))
                else:
                    # Remove expired entry
                    del namespace_cache[key]
                    return None
            
            # If not in memory and persistence is enabled, check disk
            if self.enable_persistence:
                disk_value = await self._load_from_disk(key, namespace)
                if disk_value is not None:
                    # Update in-memory cache
                    if namespace not in self.cache:
                        self.cache[namespace] = {}
                    self.cache[namespace][key] = disk_value
                    
                    # Check if disk entry has expired
                    expiration = disk_value.get("expiration")
                    if expiration is None or expiration > time.time():
                        return cast(T, disk_value.get("value"))
            
            return None
    
    async def set(
        self, 
        key: str, 
        value: T, 
        namespace: str = "default", 
        ttl: Optional[int] = None
    ) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            namespace: The cache namespace
            ttl: Time-to-live in seconds, or None for default
        """
        ttl = ttl if ttl is not None else self.default_ttl
        expiration = None if ttl is None else time.time() + ttl
        
        cache_entry = {
            "value": value,
            "created": time.time(),
            "expiration": expiration,
            "ttl": ttl
        }
        
        async with self.cache_lock:
            # Update in-memory cache
            if namespace not in self.cache:
                self.cache[namespace] = {}
            self.cache[namespace][key] = cache_entry
            
            # Update disk cache if persistence is enabled
            if self.enable_persistence:
                await self._save_to_disk(key, cache_entry, namespace)
    
    async def delete(self, key: str, namespace: str = "default") -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: The cache key
            namespace: The cache namespace
            
        Returns:
            True if the key was deleted, False if it didn't exist
        """
        deleted = False
        
        async with self.cache_lock:
            # Delete from in-memory cache
            if namespace in self.cache and key in self.cache[namespace]:
                del self.cache[namespace][key]
                deleted = True
            
            # Delete from disk if persistence is enabled
            if self.enable_persistence:
                file_path = self._get_cache_file_path(key, namespace)
                if file_path.exists():
                    try:
                        os.remove(file_path)
                        deleted = True
                    except Exception as e:
                        logger.warning(f"Failed to delete cache file {file_path}: {e}")
        
        return deleted
    
    async def clear(self, namespace: str = None) -> None:
        """
        Clear the cache, optionally for a specific namespace.
        
        Args:
            namespace: The namespace to clear, or None to clear all
        """
        async with self.cache_lock:
            # Clear in-memory cache
            if namespace is None:
                self.cache = {}
            elif namespace in self.cache:
                del self.cache[namespace]
            
            # Clear disk cache if persistence is enabled
            if self.enable_persistence:
                if namespace is None:
                    # Clear all namespaces
                    try:
                        for item in self.cache_dir.iterdir():
                            if item.is_dir():
                                for file in item.iterdir():
                                    os.remove(file)
                    except Exception as e:
                        logger.warning(f"Failed to clear all cache directories: {e}")
                else:
                    # Clear specific namespace
                    namespace_dir = self.cache_dir / namespace
                    if namespace_dir.exists():
                        try:
                            for file in namespace_dir.iterdir():
                                os.remove(file)
                        except Exception as e:
                            logger.warning(f"Failed to clear cache directory {namespace_dir}: {e}")
    
    async def cleanup_expired(self) -> int:
        """
        Remove all expired cache entries.
        
        Returns:
            Number of expired entries removed
        """
        removed_count = 0
        current_time = time.time()
        
        async with self.cache_lock:
            # Clean up in-memory cache
            for namespace, namespace_cache in list(self.cache.items()):
                for key, cache_entry in list(namespace_cache.items()):
                    expiration = cache_entry.get("expiration")
                    if expiration is not None and expiration <= current_time:
                        del namespace_cache[key]
                        removed_count += 1
            
            # Clean up disk cache if persistence is enabled
            if self.enable_persistence:
                try:
                    for namespace_dir in self.cache_dir.iterdir():
                        if namespace_dir.is_dir():
                            for file_path in namespace_dir.iterdir():
                                try:
                                    with open(file_path, 'r') as f:
                                        cache_entry = json.load(f)
                                    
                                    expiration = cache_entry.get("expiration")
                                    if expiration is not None and expiration <= current_time:
                                        os.remove(file_path)
                                        removed_count += 1
                                except Exception as e:
                                    logger.warning(f"Failed to process cache file {file_path}: {e}")
                except Exception as e:
                    logger.warning(f"Failed to clean up expired cache entries: {e}")
        
        if removed_count > 0:
            logger.debug(f"Removed {removed_count} expired cache entries")
        
        return removed_count
    
    def _get_cache_file_path(self, key: str, namespace: str) -> Path:
        """Get the file path for a cache key."""
        # Create namespace directory if it doesn't exist
        namespace_dir = self.cache_dir / namespace
        namespace_dir.mkdir(parents=True, exist_ok=True)
        
        # Use a hash of the key to avoid invalid filename characters
        key_hash = str(hash(key))
        return namespace_dir / f"cache_{key_hash}.json"
    
    async def _save_to_disk(self, key: str, cache_entry: Dict[str, Any], namespace: str) -> None:
        """Save a cache entry to disk."""
        try:
            file_path = self._get_cache_file_path(key, namespace)
            
            # Ensure cache_entry is serializable
            serializable_entry = cache_entry.copy()
            
            with open(file_path, 'w') as f:
                json.dump(serializable_entry, f)
        except Exception as e:
            logger.warning(f"Failed to save cache entry to disk: {e}")
    
    async def _load_from_disk(self, key: str, namespace: str) -> Optional[Dict[str, Any]]:
        """Load a cache entry from disk."""
        try:
            file_path = self._get_cache_file_path(key, namespace)
            if not file_path.exists():
                return None
            
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache entry from disk: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "total_entries": 0,
            "total_namespaces": 0,
            "namespaces": {},
            "persistence_enabled": self.enable_persistence,
            "default_ttl": self.default_ttl,
        }
        
        for namespace, entries in self.cache.items():
            namespace_stats = {
                "entries": len(entries),
                "expired_entries": 0,
                "valid_entries": 0,
            }
            
            current_time = time.time()
            for entry in entries.values():
                expiration = entry.get("expiration")
                if expiration is None or expiration > current_time:
                    namespace_stats["valid_entries"] += 1
                else:
                    namespace_stats["expired_entries"] += 1
            
            stats["namespaces"][namespace] = namespace_stats
            stats["total_entries"] += namespace_stats["entries"]
        
        stats["total_namespaces"] = len(stats["namespaces"])
        
        return stats 