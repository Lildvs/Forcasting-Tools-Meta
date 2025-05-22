from __future__ import annotations

import asyncio
import os
from typing import Literal, Optional

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents


class PerplexitySearcher:
    """
    A search helper that uses Perplexity AI models to gather real-time information
    from the web and synthesize it for forecasting questions.
    
    Perplexity is specialized for real-time web search and information synthesis,
    providing up-to-date information with source attribution and helping identify
    recent developments not in training data.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_open_router: bool = False,
    ) -> None:
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.use_open_router = use_open_router or (not self.api_key and os.getenv("OPENROUTER_API_KEY"))

        if not self.api_key and not os.getenv("OPENROUTER_API_KEY") and not use_open_router:
            raise ValueError("PERPLEXITY_API_KEY or OPENROUTER_API_KEY is not set")

    def get_formatted_news(self, query: str) -> str:
        """Synchronous wrapper for get_formatted_news_async"""
        return asyncio.run(self.get_formatted_news_async(query))

    async def get_formatted_news_async(self, query: str) -> str:
        """
        Use Perplexity's sonar model to search for and summarize relevant news about the query.
        This is the basic search functionality.
        """
        prompt = clean_indents(
            f"""
            You are a research assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.
            
            For each relevant piece of information you find, include the source with a link.
            Always format your response with clear sections, bullet points where appropriate, and include publication dates of information.
            
            Question:
            {query}
            """
        )
        
        model = self._get_perplexity_model("basic")
        response = await model.invoke(prompt)
        return response

    async def get_formatted_deep_research(
        self,
        query: str,
        search_depth: Literal["low", "medium", "high"] = "medium",
        max_results: int = 10,
    ) -> str:
        """
        Use Perplexity's deep research capability to perform in-depth analysis on a query.
        
        Parameters:
        - query: The forecasting question to research
        - search_depth: Determines which model to use and how deep the research goes
        - max_results: Maximum number of results to include
        """
        depth_description = ""
        if search_depth == "low":
            depth_description = "Conduct a basic search to find key facts and recent developments."
        elif search_depth == "medium":
            depth_description = "Conduct a thorough search including multiple perspectives and analyses."
        else:  # high
            depth_description = "Conduct an exhaustive search, digging deep into specialized sources, expert opinions, and technical details."
        
        prompt = clean_indents(
            f"""
            You are a professional research assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            
            {depth_description}
            
            To be a great assistant, you generate a detailed, well-sourced analysis that:
            1. Identifies the most relevant facts and data points
            2. Presents multiple perspectives on the question
            3. Highlights recent developments that may impact the forecast
            4. Notes important uncertainties or knowledge gaps
            5. Provides clear citations for all information
            
            You do NOT produce forecasts yourself - stick to providing information.
            
            Format your response in clear sections with proper citations.
            Include links to sources whenever possible.
            
            Question:
            {query}
            """
        )
        
        model = self._get_perplexity_model(search_depth)
        response = await model.invoke(prompt)
        return response

    def _get_perplexity_model(self, depth: str | Literal["basic", "low", "medium", "high"]) -> GeneralLlm:
        """Get the appropriate Perplexity model based on the requested depth"""
        
        if depth in ["basic", "low"]:
            model_name = "perplexity/sonar" if not self.use_open_router else "openrouter/perplexity/sonar"
        elif depth == "medium":
            model_name = "perplexity/sonar-pro" if not self.use_open_router else "openrouter/perplexity/sonar-pro"
        else:  # high or deep research
            model_name = "perplexity/sonar-reasoning-pro" if not self.use_open_router else "openrouter/perplexity/sonar-reasoning-pro"
        
        return GeneralLlm(
            model=model_name,
            temperature=0.1,
            timeout=120,  # Longer timeout for search operations
        ) 