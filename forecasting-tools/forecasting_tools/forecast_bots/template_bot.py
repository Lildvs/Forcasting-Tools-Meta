from forecasting_tools.forecast_bots.official_bots.q2_template_bot import (
    Q2TemplateBot2025,
)
import asyncio
from forecasting_tools.data_models.questions import BinaryQuestion, NumericQuestion, MultipleChoiceQuestion
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class BinaryForecast:
    """Forecast result for binary questions with the expected attributes for the Streamlit app."""
    binary_prob: float
    reasoning: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class NumericForecast:
    """Forecast result for numeric questions with the expected attributes for the Streamlit app."""
    mean: float
    low: float
    high: float
    reasoning: str
    metadata: Optional[Dict[str, Any]] = None


class TemplateBot(Q2TemplateBot2025):
    """A concrete implementation of ForecastBot with sync methods for the Streamlit app."""
    
    def forecast_binary(self, question: BinaryQuestion):
        """Forecast on a binary question."""
        reasoned_prediction = asyncio.run(self._run_forecast_on_binary(question, ""))
        # Convert to the format expected by the Streamlit app
        return BinaryForecast(
            binary_prob=reasoned_prediction.prediction_value,
            reasoning=reasoned_prediction.reasoning,
            metadata=getattr(reasoned_prediction, 'metadata', None)
        )
    
    def forecast_numeric(self, question: NumericQuestion):
        """Forecast on a numeric question."""
        reasoned_prediction = asyncio.run(self._run_forecast_on_numeric(question, ""))
        numeric_dist = reasoned_prediction.prediction_value
        
        # Extract the mean, low, and high values from the numeric distribution
        mean = numeric_dist.mean
        
        # Get the range (5th to 95th percentile as a reasonable default)
        percentiles = numeric_dist.declared_percentiles
        low = percentiles.get(5, mean * 0.5)  # Default to half of mean if 5th percentile is missing
        high = percentiles.get(95, mean * 1.5)  # Default to 1.5x mean if 95th percentile is missing
        
        return NumericForecast(
            mean=mean,
            low=low,
            high=high,
            reasoning=reasoned_prediction.reasoning,
            metadata=getattr(reasoned_prediction, 'metadata', None)
        )
    
    def forecast_multiple_choice(self, question: MultipleChoiceQuestion):
        """Forecast on a multiple choice question."""
        return asyncio.run(self._run_forecast_on_multiple_choice(question, ""))
