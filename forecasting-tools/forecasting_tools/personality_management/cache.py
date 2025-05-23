"""
Personality Configuration Cache

This module provides caching mechanisms for personality configurations
to improve performance by reducing redundant loading operations.
"""

import logging
import time
from typing import Dict, Optional, Any, Callable
from functools import wraps

from forecasting_tools.personality_management.config import PersonalityConfig

logger = logging.getLogger(__name__)

class PersonalityCache:
    """
    Cache for personality configurations to improve performance.
    
    This cache stores personality configurations in memory to avoid redundant 
    file loading and parsing operations.
    """
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern for the cache."""
        if cls._instance is None:
            cls._instance = super(PersonalityCache, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Initialize the cache."""
        self._cache: Dict[str, PersonalityConfig] = {}
        self._timestamp_cache: Dict[str, float] = {}
        self._config_file_paths: Dict[str, str] = {}
        self._cache_ttl = 300  # 5 minutes TTL by default
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        logger.debug("Personality cache initialized")
    
    def set_ttl(self, ttl_seconds: int) -> None:
        """
        Set the time-to-live for cache entries.
        
        Args:
            ttl_seconds: Cache TTL in seconds
        """
        self._cache_ttl = ttl_seconds
        logger.debug(f"Cache TTL set to {ttl_seconds} seconds")
    
    def get(self, personality_name: str) -> Optional[PersonalityConfig]:
        """
        Get a personality configuration from the cache.
        
        Args:
            personality_name: Name of the personality to retrieve
            
        Returns:
            Cached personality configuration or None if not found
        """
        # Check if the entry exists and is not expired
        if personality_name in self._cache:
            # Check expiration
            timestamp = self._timestamp_cache.get(personality_name, 0)
            if time.time() - timestamp <= self._cache_ttl:
                self._hits += 1
                logger.debug(f"Cache hit for personality '{personality_name}'")
                return self._cache[personality_name]
            else:
                # Expired entry
                self._evict(personality_name)
                
        self._misses += 1
        logger.debug(f"Cache miss for personality '{personality_name}'")
        return None
    
    def put(self, personality_name: str, config: PersonalityConfig, file_path: Optional[str] = None) -> None:
        """
        Store a personality configuration in the cache.
        
        Args:
            personality_name: Name of the personality
            config: Personality configuration object
            file_path: Optional path to the configuration file
        """
        self._cache[personality_name] = config
        self._timestamp_cache[personality_name] = time.time()
        
        if file_path:
            self._config_file_paths[personality_name] = file_path
            
        logger.debug(f"Cached personality '{personality_name}'")
    
    def invalidate(self, personality_name: str) -> None:
        """
        Invalidate a specific personality in the cache.
        
        Args:
            personality_name: Name of the personality to invalidate
        """
        self._evict(personality_name)
        logger.debug(f"Invalidated cache for personality '{personality_name}'")
    
    def invalidate_all(self) -> None:
        """Invalidate the entire cache."""
        self._cache.clear()
        self._timestamp_cache.clear()
        self._config_file_paths.clear()
        logger.debug("Invalidated entire personality cache")
    
    def get_file_path(self, personality_name: str) -> Optional[str]:
        """
        Get the file path for a cached personality configuration.
        
        Args:
            personality_name: Name of the personality
            
        Returns:
            File path or None if not found
        """
        return self._config_file_paths.get(personality_name)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests) * 100 if total_requests > 0 else 0
        
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
            "ttl_seconds": self._cache_ttl
        }
    
    def _evict(self, personality_name: str) -> None:
        """
        Remove a personality from the cache.
        
        Args:
            personality_name: Name of the personality to evict
        """
        if personality_name in self._cache:
            del self._cache[personality_name]
        if personality_name in self._timestamp_cache:
            del self._timestamp_cache[personality_name]
        if personality_name in self._config_file_paths:
            del self._config_file_paths[personality_name]
        self._evictions += 1


def cached_personality(func: Callable) -> Callable:
    """
    Decorator for caching personality loading functions.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with caching
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Assume first argument after self is personality_name
        if len(args) > 1:
            personality_name = args[1]
        else:
            personality_name = kwargs.get("personality_name")
            
        if not personality_name:
            return func(*args, **kwargs)
            
        # Try to get from cache
        cache = PersonalityCache()
        cached_config = cache.get(personality_name)
        
        if cached_config:
            return cached_config
            
        # Not in cache, call original function
        config = func(*args, **kwargs)
        
        # Add to cache if it's a PersonalityConfig
        if isinstance(config, PersonalityConfig):
            cache.put(personality_name, config)
            
        return config
    
    return wrapper 