"""
Exa Search Summarizer module.

This module provides a wrapper around ExaSearcher to summarize search results.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Union

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.exa_searcher import (
    ExaHighlightQuote,
    ExaSearcher,
    SearchInput,
    ExaSource
)
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.model_interfaces.ai_model import AiModel
from forecasting_tools.ai_models.model_interfaces.outputs_text import (
    OutputsText,
)
from forecasting_tools.forecast_helpers.works_cited_creator import (
    WorksCitedCreator,
)
from forecasting_tools.util.misc import (
    fill_in_citations,
    make_text_fragment_url,
)

logger = logging.getLogger(__name__)


class ExaSearchSummarizer(OutputsText, AiModel):
    """
    Summarizes search results from Exa.
    """

    def __init__(
        self,
        temperature: float = 0,
        num_searches_to_run: int = 2,
        num_sites_per_search: int = 7,
        include_works_cited_list: bool = False,
        model: str = "gpt-4o",
    ):
        """
        Initialize the ExaSearchSummarizer.

        Args:
            temperature: Temperature for the LLM
            num_searches_to_run: Number of search queries to run
            num_sites_per_search: Number of sites to retrieve per search
            include_works_cited_list: Whether to include a works cited list
            model: LLM model to use for summarization
        """
        self.temperature = temperature
        self.num_searches_to_run = num_searches_to_run
        self.num_sites_per_search = num_sites_per_search
        self.include_works_cited_list = include_works_cited_list
        self.model = model
        self.exa_searcher = ExaSearcher()
        self.llm = GeneralLlm(model=model, temperature=temperature)

    async def invoke(self, prompt: str) -> str:
        """
        Perform Exa searches and summarize the results.

        Args:
            prompt: The search query or instruction

        Returns:
            A summary of the search results
        """
        logger.info(f"ExaSearchSummarizer starting with prompt: {prompt[:50]}...")
        
        # Perform Exa searches
        search_input = SearchInput(query=prompt)
        search_results = await self.exa_searcher.invoke(
            search_input, 
            num_results=self.num_sites_per_search,
            use_autoprompt=True
        )
        
        if not search_results:
            return "No search results found. Please try a different query."
        
        # Format search results for the LLM
        formatted_results = self._format_search_results(search_results)
        
        # Create summarization prompt
        summarization_prompt = self._create_summarization_prompt(prompt, formatted_results)
        
        # Generate summary
        summary = await self.llm.invoke(summarization_prompt)
        
        # Add citations if needed
        if self.include_works_cited_list:
            works_cited = WorksCitedCreator.create_works_cited(search_results)
            summary += f"\n\n## Works Cited\n{works_cited}"
        
        return summary

    def _format_search_results(self, sources: List[ExaSource]) -> str:
        """Format search results for the LLM."""
        formatted_results = ""
        for i, source in enumerate(sources, 1):
            formatted_results += f"Source {i}: {source.title or 'No title'}\n"
            formatted_results += f"URL: {source.url or 'No URL'}\n"
            formatted_results += f"Published: {source.readable_publish_date}\n"
            formatted_results += "Highlights:\n"
            
            for j, highlight in enumerate(source.highlights, 1):
                formatted_results += f"  {j}. {highlight}\n"
            
            formatted_results += "\n"
        
        return formatted_results

    def _create_summarization_prompt(self, original_prompt: str, formatted_results: str) -> str:
        """Create a prompt for the LLM to summarize search results."""
        return clean_indents(f"""
            You are a helpful AI research assistant. Summarize the following search results to answer the user's query.
            Include specific details, numbers, and facts from the search results.
            Cite your sources using [Source X] notation.

            User query: {original_prompt}

            Search results:
            {formatted_results}

            Provide a comprehensive summary that answers the user's query based on these search results.
            Make sure to include appropriate citations to the sources.
        """)

    async def aestimate_tokens(self, prompt: str) -> int:
        """Estimate the number of tokens that will be used."""
        # This is a rough estimate
        base_tokens = 1000  # Base token usage for the system
        prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate of prompt tokens
        search_result_tokens = self.num_searches_to_run * self.num_sites_per_search * 200  # Rough estimate of search result tokens
        return int(base_tokens + prompt_tokens + search_result_tokens) 