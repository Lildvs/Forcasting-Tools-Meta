from forecasting_tools.forecast_bots.official_bots.q2_template_bot import (
    Q2TemplateBot2025,
)
import asyncio
from forecasting_tools.data_models.questions import BinaryQuestion, NumericQuestion, MultipleChoiceQuestion


class TemplateBot(Q2TemplateBot2025):
    """A concrete implementation of ForecastBot with sync methods for the Streamlit app."""
    
    def forecast_binary(self, question: BinaryQuestion):
        """Forecast on a binary question."""
        return asyncio.run(self._run_forecast_on_binary(question, ""))
    
    def forecast_numeric(self, question: NumericQuestion):
        """Forecast on a numeric question."""
        return asyncio.run(self._run_forecast_on_numeric(question, ""))
    
    def forecast_multiple_choice(self, question: MultipleChoiceQuestion):
        """Forecast on a multiple choice question."""
        return asyncio.run(self._run_forecast_on_multiple_choice(question, ""))
