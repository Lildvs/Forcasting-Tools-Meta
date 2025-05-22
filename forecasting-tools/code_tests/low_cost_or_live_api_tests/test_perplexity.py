import asyncio
import logging
import os

import pytest

from forecasting_tools.forecast_helpers.perplexity_searcher import PerplexitySearcher

logger = logging.getLogger(__name__)


def test_perplexity_connection() -> None:
    if not os.getenv("PERPLEXITY_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("PERPLEXITY_API_KEY or OPENROUTER_API_KEY is not set")
    
    logger.debug("Testing Perplexity connection")
    
    news = PerplexitySearcher().get_formatted_news(
        "Will the UK prime minister change before 2025?"
    )
    
    assert news != ""
    assert len(news) > 100


@pytest.mark.asyncio
async def test_perplexity_deep_research() -> None:
    if not os.getenv("PERPLEXITY_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("PERPLEXITY_API_KEY or OPENROUTER_API_KEY is not set")
    
    logger.debug("Testing Perplexity deep research")
    
    news = await PerplexitySearcher().get_formatted_deep_research(
        "Will the UK prime minister change before 2025?",
        search_depth="low",
    )
    
    assert news != ""
    assert len(news) > 100 