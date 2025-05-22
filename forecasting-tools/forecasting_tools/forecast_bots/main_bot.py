import logging

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.forecast_report import ReasonedPrediction
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
)
from forecasting_tools.forecast_bots.template_bot import TemplateBot
from forecasting_tools.forecast_helpers.perplexity_searcher import PerplexitySearcher

logger = logging.getLogger(__name__)


class MainBot(TemplateBot):
    """
    Just a bot that does something simple to demonstrate the package.
    """

    def __init__(
        self,
        *,
        research_reports_per_question: int = 1,
        predictions_per_research_report: int = 5,
        use_research_summary_to_forecast: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            research_reports_per_question=research_reports_per_question,
            predictions_per_research_report=predictions_per_research_report,
            use_research_summary_to_forecast=use_research_summary_to_forecast,
            **kwargs,
        )

    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        We get basic news context for the question.
        """
        research = await PerplexitySearcher().get_formatted_news_async(
            question.question_text
        )
        return research

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        """Simplified forecast function that always returns 50%."""
        reasoning = "I'm not sure, so I'll just say 50%."
        return ReasonedPrediction(prediction_value=0.5, reasoning=reasoning)

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm]:
        return {
            "default": GeneralLlm(model="openai/o1", temperature=1),
            "summarizer": GeneralLlm(
                model="openai/gpt-4o-mini", temperature=0
            ),
        }
