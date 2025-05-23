"""
Crawl4AI Searcher

This module provides a searcher implementation that uses Crawl4AI to perform
web research with controlled crawling, document-based search, and fine-grained
content analysis capabilities.
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional, Any, Union, Literal

import aiohttp

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents

logger = logging.getLogger(__name__)


class Crawl4AISearcher:
    """
    A searcher that leverages Crawl4AI for deep web research with customizable
    crawling strategies, document searching, and content analysis.
    
    Crawl4AI provides controlled web crawling capabilities, allowing for more
    targeted and comprehensive research compared to traditional search engines.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.crawl4ai.com/v1",
        timeout: int = 90,
        max_pages: int = 10,
        crawl_depth: int = 2,
        follow_links: bool = True,
        synthesis_model: str = "gpt-4o",
    ):
        """
        Initialize the Crawl4AI searcher.
        
        Args:
            api_key: Crawl4AI API key
            base_url: Base URL for the Crawl4AI API
            timeout: Default timeout in seconds
            max_pages: Maximum number of pages to crawl
            crawl_depth: Maximum crawl depth
            follow_links: Whether to follow links during crawling
            synthesis_model: LLM model to use for synthesizing search results
        """
        self.api_key = api_key or os.getenv("CRAWL4AI_API_KEY")
        
        if not self.api_key:
            raise ValueError("CRAWL4AI_API_KEY is not set")
        
        self.base_url = base_url
        self.timeout = timeout
        self.max_pages = max_pages
        self.crawl_depth = crawl_depth
        self.follow_links = follow_links
        
        # Initialize the LLM for synthesis
        self.synthesis_model = GeneralLlm(
            model=synthesis_model,
            temperature=0.1,
        )
    
    async def get_formatted_search(
        self,
        query: str,
        search_type: Literal["search", "crawl", "both"] = "both",
        max_results: int = 10,
        follow_links: Optional[bool] = None,
        crawl_depth: Optional[int] = None,
        domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        time_frame: Optional[str] = None,
    ) -> str:
        """
        Perform a search using Crawl4AI and format the results.
        
        Args:
            query: The search query
            search_type: Type of search to perform (search, crawl, or both)
            max_results: Maximum number of results to return
            follow_links: Whether to follow links during crawling
            crawl_depth: Maximum crawl depth
            domains: List of domains to limit search to
            exclude_domains: List of domains to exclude from search
            time_frame: Time frame for search results (e.g., "past_week", "past_month")
            
        Returns:
            Formatted search results as a string
        """
        # Use provided parameters or fall back to instance defaults
        follow_links = self.follow_links if follow_links is None else follow_links
        crawl_depth = self.crawl_depth if crawl_depth is None else crawl_depth
        
        # Perform the search
        raw_results = await self._perform_search(
            query=query,
            search_type=search_type,
            max_results=max_results,
            follow_links=follow_links,
            crawl_depth=crawl_depth,
            domains=domains,
            exclude_domains=exclude_domains,
            time_frame=time_frame,
        )
        
        # Format and synthesize the results
        formatted_results = await self._synthesize_results(query, raw_results)
        
        return formatted_results
    
    async def _perform_search(
        self,
        query: str,
        search_type: Literal["search", "crawl", "both"],
        max_results: int,
        follow_links: bool,
        crawl_depth: int,
        domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        time_frame: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the search against the Crawl4AI API.
        
        Args:
            query: The search query
            search_type: Type of search to perform
            max_results: Maximum number of results
            follow_links: Whether to follow links
            crawl_depth: Maximum crawl depth
            domains: List of domains to limit search to
            exclude_domains: List of domains to exclude
            time_frame: Time frame for search results
            
        Returns:
            Raw search results from the API
        """
        # Prepare the request payload
        payload = {
            "query": query,
            "max_results": max_results,
        }
        
        # Add optional parameters if provided
        if domains:
            payload["domains"] = domains
        
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains
        
        if time_frame:
            payload["time_frame"] = time_frame
        
        # Set the appropriate endpoint based on search_type
        endpoint = ""
        if search_type == "search":
            endpoint = "/search"
        elif search_type == "crawl":
            endpoint = "/crawl"
            payload["follow_links"] = follow_links
            payload["crawl_depth"] = crawl_depth
        else:  # both
            endpoint = "/combined"
            payload["follow_links"] = follow_links
            payload["crawl_depth"] = crawl_depth
        
        # Execute the API request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}{endpoint}",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=self.timeout,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Crawl4AI API error: {error_text}")
                        return {"error": f"API error: {response.status}", "results": []}
                    
                    return await response.json()
        except asyncio.TimeoutError:
            logger.error(f"Crawl4AI request timed out after {self.timeout} seconds")
            return {"error": "Request timed out", "results": []}
        except Exception as e:
            logger.error(f"Crawl4AI request failed: {str(e)}")
            return {"error": f"Request failed: {str(e)}", "results": []}
    
    async def _synthesize_results(self, query: str, raw_results: Dict[str, Any]) -> str:
        """
        Synthesize and format the search results into a coherent response.
        
        Args:
            query: The original search query
            raw_results: Raw search results from the API
            
        Returns:
            Formatted and synthesized results
        """
        # Check if there was an error
        if "error" in raw_results:
            return f"ERROR: {raw_results['error']}"
        
        # Check if we have results
        results = raw_results.get("results", [])
        if not results:
            return "No results found for the query."
        
        # Extract content and metadata from results
        extracted_content = []
        for result in results:
            title = result.get("title", "No title")
            url = result.get("url", "")
            snippet = result.get("snippet", "")
            content = result.get("content", "")
            date = result.get("date", "")
            
            # Use snippet if content is too long, otherwise use content
            text_to_use = snippet if len(content) > 500 else content
            
            # Format result entry
            entry = {
                "title": title,
                "url": url,
                "date": date,
                "content": text_to_use,
            }
            
            extracted_content.append(entry)
        
        # Create a prompt for synthesis
        prompt = clean_indents(
            f"""
            You are a research assistant analyzing web search results for a forecaster.
            
            The search query was: 
            {query}
            
            Below are the search results:
            
            {json.dumps(extracted_content, indent=2)}
            
            Please provide a comprehensive synthesis of these results that:
            1. Extracts the most relevant information related to the query
            2. Organizes findings in a clear, structured format
            3. Highlights areas of consensus and disagreement across sources
            4. Notes the credibility and recency of sources where relevant
            5. Includes proper citations with URLs for all information
            
            Focus on extracting factual information and data points that would be useful for making a forecast
            rather than opinions or speculation. Format your response with clear sections and bullet points where appropriate.
            """
        )
        
        # Generate the synthesis
        try:
            synthesis = await self.synthesis_model.invoke(prompt)
            return synthesis
        except Exception as e:
            logger.error(f"Failed to synthesize results: {e}")
            
            # Fallback to a simple format if synthesis fails
            formatted_results = "# Search Results\n\n"
            for i, result in enumerate(extracted_content, 1):
                formatted_results += f"## {i}. {result['title']}\n"
                if result['date']:
                    formatted_results += f"Date: {result['date']}\n"
                formatted_results += f"Source: {result['url']}\n\n"
                formatted_results += f"{result['content']}\n\n"
            
            return formatted_results
    
    async def get_formatted_deep_research(
        self,
        query: str,
        search_depth: Literal["low", "medium", "high"] = "medium",
        max_results: int = 10,
    ) -> str:
        """
        Perform deep research on a query with configurable depth.
        
        Args:
            query: The research query
            search_depth: Depth of search (low, medium, high)
            max_results: Maximum number of results
            
        Returns:
            Formatted deep research results as a string
        """
        # Configure search parameters based on depth
        if search_depth == "low":
            # Basic search with minimal crawling
            return await self.get_formatted_search(
                query=query,
                search_type="search",  # Just search, no crawling
                max_results=max_results,
                follow_links=False,
                crawl_depth=0,
            )
        elif search_depth == "medium":
            # Balanced search and crawl
            return await self.get_formatted_search(
                query=query,
                search_type="both",
                max_results=max_results,
                follow_links=True,
                crawl_depth=1,  # Only first level of links
            )
        else:  # high
            # Deep crawling
            return await self.get_formatted_search(
                query=query,
                search_type="both",
                max_results=max_results,
                follow_links=True,
                crawl_depth=2,  # Follow links up to 2 levels deep
                # Increase timeout for deep crawling
                # This is handled in the _perform_search method via self.timeout
            )
    
    # Alias for compatibility with Perplexity interface
    get_formatted_news = get_formatted_search
    get_formatted_news_async = get_formatted_search 