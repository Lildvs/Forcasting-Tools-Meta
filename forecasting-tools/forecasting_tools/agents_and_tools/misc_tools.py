import asyncio
import logging
from datetime import datetime
from typing import Literal

from forecasting_tools.agents_and_tools.question_generators.simple_question import (
    SimpleQuestion,
)
from forecasting_tools.ai_models.agent_wrappers import AgentTool, agent_tool
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_helpers.metaculus_api import (
    MetaculusApi,
    MetaculusQuestion,
)
from forecasting_tools.forecast_helpers.smart_searcher import SmartSearcher
from forecasting_tools.forecast_helpers.exa_search_summarizer import (
    ExaSearchSummarizer,
)
from forecasting_tools.forecast_helpers.perplexity_searcher import PerplexitySearcher

logger = logging.getLogger(__name__)


@agent_tool
async def get_general_news_with_perplexity(topic: str) -> str:
    """
    Get general news context for a topic using Perplexity.
    
    This searches for recent information on the given topic using Perplexity's search capabilities.
    """
    # TODO: Insert an if statement that will use Exa summaries if Perplexity keys are not enabled
    return await PerplexitySearcher().get_formatted_news_async(topic)


@agent_tool
async def perplexity_pro_search(query: str) -> str:
    """
    Use Perplexity (sonar-reasoning-pro) to search for information on a topic.
    
    This is Perplexity's highest quality search model.
    """
    model = GeneralLlm(
        model="openrouter/perplexity/sonar-reasoning-pro",
        temperature=0.0,
        timeout=180,
    )
    return await model.invoke(query)


@agent_tool
async def perplexity_quick_search(query: str) -> str:
    """
    Use Perplexity (sonar) to search for information on a topic.
    
    This is Perplexity's fastest but lowest quality search model.
    """
    model = GeneralLlm(
        model="openrouter/perplexity/sonar",
        temperature=0.0,
    )
    return await model.invoke(query)


@agent_tool
async def smart_searcher_search(query: str) -> str:
    """
    Use SmartSearcher to search for information on a topic.
    This will provide a LLM answer with citations.
    Citations will include url text fragments for faster fact checking.
    """
    return await SmartSearcher(model="openrouter/openai/o4-mini").invoke(query)


@agent_tool
def grab_question_details_from_metaculus(
    url_or_id: str | int,
) -> MetaculusQuestion:
    """
    This function grabs the details of a question from a Metaculus URL or ID.
    """
    if isinstance(url_or_id, int):
        question = MetaculusApi.get_question_by_post_id(url_or_id)
    else:
        question = MetaculusApi.get_question_by_url(url_or_id)
    question.api_json = {}
    return question


@agent_tool
def grab_open_questions_from_tournament(
    tournament_id_or_slug: int | str,
) -> list[MetaculusQuestion]:
    """
    This function grabs the details of all questions from a Metaculus tournament.
    """
    questions = MetaculusApi.get_all_open_questions_from_tournament(
        tournament_id_or_slug
    )
    for question in questions:
        question.api_json = {}
    return questions


def create_tool_for_forecasting_bot(
    bot_or_class: type[ForecastBot] | ForecastBot,
) -> AgentTool:
    if isinstance(bot_or_class, type):
        bot = bot_or_class()
    else:
        bot = bot_or_class

    description = clean_indents(
        """
        Forecast a SimpleQuestion (simplified binary, numeric, or multiple choice question) using a forecasting bot.
        """
    )

    @agent_tool(description_override=description)
    def forecast_question_tool(question: SimpleQuestion) -> str:
        metaculus_question = (
            SimpleQuestion.simple_questions_to_metaculus_questions([question])[
                0
            ]
        )
        task = bot.forecast_question(metaculus_question)
        report = asyncio.run(task)
        return report.explanation

    return forecast_question_tool
