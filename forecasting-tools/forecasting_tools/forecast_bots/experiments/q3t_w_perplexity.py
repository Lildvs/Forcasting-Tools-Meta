from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.forecast_bots.official_bots.q3_template_bot import (
    Q3TemplateBot2024,
)
from forecasting_tools.forecast_helpers.perplexity_searcher import PerplexitySearcher


class Q3TemplateWithPerplexity(Q3TemplateBot2024):

    async def run_research(self, question: MetaculusQuestion) -> str:
        response = PerplexitySearcher().get_formatted_news(question.question_text)
        return response
