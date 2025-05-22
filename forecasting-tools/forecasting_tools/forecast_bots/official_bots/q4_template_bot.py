from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.forecast_bots.official_bots.q3_template_bot import (
    Q3TemplateBot2024,
)
from forecasting_tools.forecast_helpers.perplexity_searcher import PerplexitySearcher


class Q4TemplateBot2024(Q3TemplateBot2024):
    """
    Q4 Template Bot was the same as Q3 other than switching out for Perplexity
    """

    async def run_research(self, question: MetaculusQuestion) -> str:
        news = PerplexitySearcher().get_formatted_news(question.question_text)
        return news
