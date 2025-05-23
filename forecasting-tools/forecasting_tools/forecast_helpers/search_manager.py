"""
Search Manager

This module provides a unified interface for coordinating multiple search
providers, including Perplexity, Crawl4ai, and other custom searchers.
It handles provider selection, rate limiting, error handling, and caching.
"""

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Union, Literal, Any, cast

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.forecast_helpers.perplexity_searcher import PerplexitySearcher
from forecasting_tools.forecast_helpers.smart_searcher import SmartSearcher
from forecasting_tools.forecast_helpers.crawl4ai_searcher import Crawl4AISearcher
from forecasting_tools.config import SearchConfig

logger = logging.getLogger(__name__)


class SearchManager:
    """
    A unified interface for coordinating multiple search providers.
    
    This class:
    - Manages different search providers (Perplexity, Smart Searcher with Exa, Crawl4ai)
    - Handles provider selection based on availability and configuration
    - Implements rate limiting to prevent API overuse
    - Provides error handling and fallback mechanisms
    - Caches search results to reduce API calls and costs
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """Implement the Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(SearchManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        default_provider: str = None,
        use_cache: bool = None,
        rate_limit: int = None,
        timeout: int = None,
        fallback_order: List[str] = None,
    ):
        """
        Initialize the SearchManager.
        
        Args:
            default_provider: Default search provider to use ("perplexity", "smart", "crawl4ai", or "auto")
            use_cache: Whether to cache search results
            rate_limit: Maximum number of requests per minute
            timeout: Timeout in seconds for search requests
            fallback_order: List of providers to try in order if the default fails
        """
        # Skip initialization if already initialized (singleton pattern)
        if SearchManager._initialized:
            return
        
        SearchManager._initialized = True
        
        # Use provided values or fall back to config
        self.default_provider = default_provider or SearchConfig.DEFAULT_SEARCH_PROVIDER
        self.use_cache = use_cache if use_cache is not None else SearchConfig.ENABLE_SEARCH_CACHE
        self.rate_limit = rate_limit or SearchConfig.SEARCH_RATE_LIMIT
        self.timeout = timeout or SearchConfig.SEARCH_TIMEOUT
        self.fallback_order = fallback_order or SearchConfig.SEARCH_FALLBACK_ORDER
        
        # Initialize rate limiting
        self.request_timestamps: List[float] = []
        self.rate_limit_lock = asyncio.Lock()
        
        # Initialize cache
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_lock = asyncio.Lock()
        
        # Initialize providers
        self.providers = {}
        self._initialize_providers()
        
        # Available providers based on API keys
        self.available_providers = self._get_available_providers()
        
        if not self.available_providers:
            logger.warning("No search providers available! Research functionality will be limited.")
        else:
            logger.info(f"Available search providers: {', '.join(self.available_providers)}")
    
    def _initialize_providers(self):
        """Initialize all supported search providers."""
        # Initialize Perplexity if API key is available
        perplexity_api_key = SearchConfig.PERPLEXITY_API_KEY
        openrouter_api_key = SearchConfig.OPENROUTER_API_KEY
        
        if perplexity_api_key or openrouter_api_key:
            try:
                self.providers["perplexity"] = PerplexitySearcher(
                    api_key=perplexity_api_key,
                    use_open_router=not perplexity_api_key and bool(openrouter_api_key)
                )
                logger.info("Initialized Perplexity search provider")
            except Exception as e:
                logger.warning(f"Failed to initialize Perplexity search provider: {e}")
        
        # Initialize Smart Searcher if EXA API key is available
        exa_api_key = SearchConfig.EXA_API_KEY
        if exa_api_key:
            try:
                settings = SearchConfig.SMART_SEARCHER_SETTINGS
                self.providers["smart"] = SmartSearcher(
                    include_works_cited_list=settings["include_works_cited_list"],
                    use_brackets_around_citations=settings["use_brackets_around_citations"],
                    num_searches_to_run=settings["num_searches_to_run"],
                    num_sites_per_search=settings["num_sites_per_search"],
                    model=settings["model"],
                    use_advanced_filters=settings["use_advanced_filters"]
                )
                logger.info("Initialized Smart Searcher provider")
            except Exception as e:
                logger.warning(f"Failed to initialize Smart Searcher provider: {e}")
        
        # Initialize Crawl4ai if API key is available
        crawl4ai_api_key = SearchConfig.CRAWL4AI_API_KEY
        if crawl4ai_api_key:
            try:
                settings = SearchConfig.CRAWL4AI_SETTINGS
                self.providers["crawl4ai"] = Crawl4AISearcher(
                    api_key=crawl4ai_api_key,
                    timeout=settings["timeout"],
                    max_pages=settings["max_pages"],
                    crawl_depth=settings["crawl_depth"], 
                    follow_links=settings["follow_links"],
                    synthesis_model=settings["synthesis_model"]
                )
                logger.info("Initialized Crawl4AI search provider")
            except Exception as e:
                logger.warning(f"Failed to initialize Crawl4AI search provider: {e}")
    
    def _get_available_providers(self) -> List[str]:
        """Get list of available search providers based on initialized providers."""
        return list(self.providers.keys())
    
    def get_provider(self, provider_name: str = None) -> Any:
        """
        Get a search provider by name or default.
        
        Args:
            provider_name: The name of the provider to get, or None to use default logic
            
        Returns:
            The requested search provider or the best available one
        
        Raises:
            ValueError: If the requested provider is not available and no fallbacks exist
        """
        # If no provider specified, use default selection logic
        if provider_name is None:
            provider_name = self.default_provider
        
        # If "auto", select the best available provider
        if provider_name == "auto":
            for provider in self.fallback_order:
                if provider in self.available_providers:
                    return self.providers[provider]
            
            # If no providers are available, raise an error
            raise ValueError("No search providers available")
        
        # If specific provider requested, try to return it
        if provider_name in self.available_providers:
            return self.providers[provider_name]
        
        # If the requested provider isn't available, try fallbacks
        logger.warning(
            f"Requested provider '{provider_name}' not available. Trying fallbacks: {self.fallback_order}"
        )
        
        for provider in self.fallback_order:
            if provider in self.available_providers:
                logger.info(f"Using fallback provider: {provider}")
                return self.providers[provider]
        
        # If no fallbacks are available, raise an error
        raise ValueError(f"Requested provider '{provider_name}' and fallbacks not available")
    
    async def _enforce_rate_limit(self):
        """Enforce rate limits to prevent API overuse."""
        async with self.rate_limit_lock:
            now = time.time()
            
            # Remove timestamps older than 60 seconds
            self.request_timestamps = [t for t in self.request_timestamps if now - t < 60]
            
            # If we've reached the rate limit, wait until we can make another request
            if len(self.request_timestamps) >= self.rate_limit:
                oldest_timestamp = min(self.request_timestamps)
                wait_time = 60 - (now - oldest_timestamp)
                if wait_time > 0:
                    logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)
            
            # Add current timestamp to the list
            self.request_timestamps.append(time.time())
    
    async def _get_from_cache(self, query: str, search_type: str) -> Optional[Dict[str, Any]]:
        """Get a search result from the cache if available."""
        if not self.use_cache:
            return None
        
        async with self.cache_lock:
            cache_key = f"{search_type}:{query}"
            if cache_key in self.cache:
                logger.info(f"Cache hit for '{query}' with search type '{search_type}'")
                return self.cache[cache_key]
            return None
    
    async def _save_to_cache(self, query: str, search_type: str, result: Dict[str, Any]):
        """Save a search result to the cache."""
        if not self.use_cache:
            return
        
        async with self.cache_lock:
            cache_key = f"{search_type}:{query}"
            self.cache[cache_key] = result
            logger.debug(f"Cached result for '{query}' with search type '{search_type}'")
    
    async def search(
        self,
        query: str,
        provider: str = None,
        search_type: Literal["basic", "deep"] = "basic",
        search_depth: Literal["low", "medium", "high"] = "medium",
        use_cache: bool = None,
        max_results: int = 10,
        timeout: int = None,
    ) -> str:
        """
        Search for information using the configured providers.
        
        Args:
            query: The search query
            provider: The provider to use (perplexity, smart, crawl4ai, or auto)
            search_type: The type of search (basic or deep)
            search_depth: The depth of the search (low, medium, high)
            use_cache: Whether to use cached results
            max_results: Maximum number of results to return
            timeout: Timeout in seconds
            
        Returns:
            The search results as a string
        """
        # Determine if cache should be used
        use_cache = self.use_cache if use_cache is None else use_cache
        timeout = self.timeout if timeout is None else timeout
        
        # Check cache first if enabled
        if use_cache:
            cached_result = await self._get_from_cache(query, f"{search_type}:{search_depth}")
            if cached_result:
                return cached_result["result"]
        
        # Enforce rate limiting
        await self._enforce_rate_limit()
        
        # Get the provider
        try:
            search_provider = self.get_provider(provider)
        except ValueError as e:
            logger.error(f"Search failed: {e}")
            return f"ERROR: No search provider available. Please ensure at least one of the following API keys is set: PERPLEXITY_API_KEY, EXA_API_KEY, CRAWL4AI_API_KEY"
        
        # Create a timeout task
        result = None
        error = None
        
        try:
            if provider == "perplexity" or (provider is None and isinstance(search_provider, PerplexitySearcher)):
                if search_type == "basic":
                    result = await asyncio.wait_for(
                        search_provider.get_formatted_news_async(query),
                        timeout=timeout
                    )
                else:  # deep
                    result = await asyncio.wait_for(
                        search_provider.get_formatted_deep_research(
                            query, search_depth=search_depth, max_results=max_results
                        ),
                        timeout=timeout
                    )
            elif provider == "smart" or (provider is None and isinstance(search_provider, SmartSearcher)):
                result = await asyncio.wait_for(
                    search_provider.invoke(query),
                    timeout=timeout
                )
            elif provider == "crawl4ai" or (provider is None and isinstance(search_provider, Crawl4AISearcher)):
                if search_type == "basic":
                    result = await asyncio.wait_for(
                        search_provider.get_formatted_news_async(query),
                        timeout=timeout
                    )
                else:  # deep
                    result = await asyncio.wait_for(
                        search_provider.get_formatted_deep_research(
                            query, search_depth=search_depth, max_results=max_results
                        ),
                        timeout=timeout
                    )
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except asyncio.TimeoutError:
            error = f"Search timed out after {timeout} seconds"
            logger.error(error)
        except Exception as e:
            error = f"Search failed: {str(e)}"
            logger.error(f"Search error: {e}", exc_info=True)
        
        # If we got a result, cache it
        if result and use_cache:
            await self._save_to_cache(
                query, 
                f"{search_type}:{search_depth}", 
                {"result": result, "timestamp": time.time()}
            )
        
        # If error and no result, return error message
        if error and not result:
            return f"ERROR: {error}"
        
        return result or "No results found" 