"""
Ensemble Bot

This module provides a forecast bot that creates and manages ensembles of
forecasters with diverse personalities to generate more robust predictions.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.forecast_report import (
    NumericDistribution,
    PredictedOptionList,
    ReasonedPrediction,
    ForecastReport
)
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_bots.basic_bot import BasicBot
from forecasting_tools.forecast_bots.research_bot import ResearchBot
from forecasting_tools.forecast_bots.calibrated_bot import CalibratedBot
from forecasting_tools.forecast_bots.economist_bot import EconomistBot
from forecasting_tools.forecast_bots.bayesian_bot import BayesianBot
from forecasting_tools.forecast_helpers.prediction_extractor import PredictionExtractor
from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.processors import ProcessorFactory


logger = logging.getLogger(__name__)


class EnsembleBot(ForecastBot):
    """
    A forecast bot that creates and manages ensembles of forecasters with
    diverse personalities to generate more robust predictions.
    
    This bot:
    - Creates an ensemble of bots with different personalities
    - Manages the diversity of the ensemble
    - Applies weighted aggregation based on past performance
    - Provides detailed reporting on ensemble dynamics
    """
    
    def __init__(
        self,
        *,
        bot_name: str | None = None,
        research_reports_per_question: int = 1,
        predictions_per_research_report: int = 1,
        use_research_summary_to_forecast: bool = False,
        publish_reports_to_metaculus: bool = False,
        folder_to_save_reports_to: str | None = None,
        skip_previously_forecasted_questions: bool = False,
        llms: dict[str, str | GeneralLlm] | None = None,
        logger_name: Optional[str] = None,
        ensemble_config: Optional[Dict[str, Any]] = None,
        personality_names: Optional[List[str]] = None,
        bot_types: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize the EnsembleBot.
        
        Args:
            bot_name: Optional name for the bot
            research_reports_per_question: Number of research reports per question
            predictions_per_research_report: Number of predictions per research report
            use_research_summary_to_forecast: Whether to use research summaries
            publish_reports_to_metaculus: Whether to publish to Metaculus
            folder_to_save_reports_to: Folder to save reports to
            skip_previously_forecasted_questions: Whether to skip previously forecasted questions
            llms: Dictionary mapping purposes to LLMs
            logger_name: Name for the logger
            ensemble_config: Configuration for the ensemble
            personality_names: List of personality names to include in the ensemble
            bot_types: List of bot types to include in the ensemble
            weights: Dictionary mapping bot identifiers to weights
        """
        super().__init__(
            research_reports_per_question=research_reports_per_question,
            predictions_per_research_report=predictions_per_research_report,
            use_research_summary_to_forecast=use_research_summary_to_forecast,
            publish_reports_to_metaculus=publish_reports_to_metaculus,
            folder_to_save_reports_to=folder_to_save_reports_to,
            skip_previously_forecasted_questions=skip_previously_forecasted_questions,
            llms=llms,
            logger_name=logger_name,
        )
        
        self.bot_name = bot_name or "EnsembleBot"
        self.ensemble_config = ensemble_config or {}
        
        # Set up personality diversity
        self.personality_manager = PersonalityManager()
        self.personality_names = personality_names or [
            "balanced", "bayesian", "economist", "creative", "cautious"
        ]
        
        # Set up bot types
        self.bot_types = bot_types or [
            "basic", "research", "calibrated", "economist", "bayesian"
        ]
        
        # Set up weights (default to equal weighting)
        self.weights = weights or {}
        
        # Create the ensemble
        self.ensemble = self._create_ensemble()
        
        logger.info(f"Initialized {self.bot_name} with {len(self.ensemble)} members")
        
    def _create_ensemble(self) -> List[Tuple[str, ForecastBot]]:
        """
        Create an ensemble of bots with diverse personalities.
        
        Returns:
            List of (identifier, bot) tuples
        """
        ensemble = []
        
        # Create bot type mapping
        bot_type_map = {
            "basic": BasicBot,
            "research": ResearchBot,
            "calibrated": CalibratedBot,
            "economist": EconomistBot,
            "bayesian": BayesianBot,
        }
        
        # Create different combinations of personality and bot type
        for personality_name in self.personality_names:
            for bot_type in self.bot_types:
                if bot_type in bot_type_map:
                    bot_class = bot_type_map[bot_type]
                    
                    # Customize bot parameters based on type
                    kwargs = {
                        "bot_name": f"{bot_type.capitalize()}_{personality_name}",
                        "personality_name": personality_name,
                        "llms": self.llms.copy() if self.llms else None,
                    }
                    
                    # Add bot type specific parameters
                    if bot_type == "calibrated":
                        kwargs["calibration_strength"] = 0.5
                    elif bot_type == "research":
                        kwargs["research_depth"] = "moderate"
                    elif bot_type == "economist":
                        kwargs["economic_focus"] = None  # Let it be determined by personality
                    elif bot_type == "bayesian":
                        kwargs["sensitivity_analysis"] = True
                    
                    # Create the bot
                    bot = bot_class(**kwargs)
                    
                    # Create a unique identifier
                    identifier = f"{bot_type}_{personality_name}"
                    
                    # Add to ensemble
                    ensemble.append((identifier, bot))
        
        return ensemble
    
    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Run research using the most comprehensive bot in the ensemble.
        
        Args:
            question: The question to research
            
        Returns:
            The research results
        """
        # Use the research bot with balanced personality for base research
        for identifier, bot in self.ensemble:
            if "research_balanced" in identifier:
                logger.info(f"Using {identifier} for ensemble research")
                return await bot.run_research(question)
        
        # Fallback to first bot if no research_balanced bot exists
        logger.info(f"Using {self.ensemble[0][0]} for ensemble research (fallback)")
        return await self.ensemble[0][1].run_research(question)
    
    async def get_forecast_from_research(
        self, question: MetaculusQuestion, research: str
    ) -> ReasonedPrediction:
        """
        Get forecasts from all ensemble members and aggregate them.
        
        Args:
            question: The question to forecast
            research: The research results
            
        Returns:
            An aggregated reasoned prediction
        """
        # Collect forecasts from all ensemble members
        ensemble_forecasts = await self._collect_ensemble_forecasts(question, research)
        
        # Aggregate the forecasts
        aggregated_forecast = await self._aggregate_ensemble_forecasts(
            question, ensemble_forecasts
        )
        
        return aggregated_forecast
    
    async def _collect_ensemble_forecasts(
        self, question: MetaculusQuestion, research: str
    ) -> Dict[str, ReasonedPrediction]:
        """
        Collect forecasts from all ensemble members.
        
        Args:
            question: The question to forecast
            research: The research results
            
        Returns:
            Dictionary mapping identifiers to reasoned predictions
        """
        forecast_tasks = []
        
        for identifier, bot in self.ensemble:
            task = asyncio.create_task(
                self._get_forecast_from_bot(bot, question, research, identifier)
            )
            forecast_tasks.append(task)
        
        # Wait for all forecasts
        results = await asyncio.gather(*forecast_tasks, return_exceptions=True)
        
        # Process results
        forecasts = {}
        for i, (identifier, _) in enumerate(self.ensemble):
            result = results[i]
            if isinstance(result, Exception):
                logger.error(f"Error from {identifier}: {str(result)}")
            else:
                forecasts[identifier] = result
        
        return forecasts
    
    async def _get_forecast_from_bot(
        self, bot: ForecastBot, question: MetaculusQuestion, research: str, identifier: str
    ) -> ReasonedPrediction:
        """
        Get a forecast from a specific bot.
        
        Args:
            bot: The bot to get a forecast from
            question: The question to forecast
            research: The research results
            identifier: The bot identifier
            
        Returns:
            A reasoned prediction
        """
        try:
            logger.info(f"Getting forecast from {identifier}")
            return await bot.get_forecast_from_research(question, research)
        except Exception as e:
            logger.error(f"Error getting forecast from {identifier}: {str(e)}")
            raise
    
    async def _aggregate_ensemble_forecasts(
        self, 
        question: MetaculusQuestion, 
        forecasts: Dict[str, ReasonedPrediction]
    ) -> ReasonedPrediction:
        """
        Aggregate forecasts from ensemble members.
        
        Args:
            question: The question being forecasted
            forecasts: Dictionary mapping identifiers to reasoned predictions
            
        Returns:
            An aggregated reasoned prediction
        """
        if not forecasts:
            raise ValueError("No forecasts to aggregate")
        
        # Determine the type of forecast
        if isinstance(question, BinaryQuestion):
            return await self._aggregate_binary_forecasts(question, forecasts)
        elif isinstance(question, NumericQuestion):
            return await self._aggregate_numeric_forecasts(question, forecasts)
        elif isinstance(question, MultipleChoiceQuestion):
            return await self._aggregate_multiple_choice_forecasts(question, forecasts)
        else:
            raise ValueError(f"Unsupported question type: {type(question)}")
    
    async def _aggregate_binary_forecasts(
        self, 
        question: BinaryQuestion, 
        forecasts: Dict[str, ReasonedPrediction]
    ) -> ReasonedPrediction:
        """
        Aggregate binary forecasts from ensemble members.
        
        Args:
            question: The binary question
            forecasts: Dictionary mapping identifiers to reasoned predictions
            
        Returns:
            An aggregated reasoned prediction
        """
        # Extract probability values
        values = []
        weights = []
        
        for identifier, forecast in forecasts.items():
            # Get the probability value
            value = forecast.prediction_value
            
            # Get the weight (default to 1.0)
            weight = self.weights.get(identifier, 1.0)
            
            values.append(value)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
        else:
            normalized_weights = [1.0 / len(weights)] * len(weights)
        
        # Calculate weighted average
        weighted_avg = sum(v * w for v, w in zip(values, normalized_weights))
        
        # Create combined reasoning
        combined_reasoning = self._create_ensemble_reasoning(forecasts)
        
        return ReasonedPrediction(
            prediction_value=weighted_avg,
            reasoning=combined_reasoning
        )
    
    async def _aggregate_numeric_forecasts(
        self, 
        question: NumericQuestion, 
        forecasts: Dict[str, ReasonedPrediction]
    ) -> ReasonedPrediction:
        """
        Aggregate numeric forecasts from ensemble members.
        
        Args:
            question: The numeric question
            forecasts: Dictionary mapping identifiers to reasoned predictions
            
        Returns:
            An aggregated reasoned prediction
        """
        # Extract distribution parameters
        means = []
        stdevs = []
        weights = []
        
        for identifier, forecast in forecasts.items():
            # Get the distribution
            distribution = forecast.prediction_value
            
            # Get the weight (default to 1.0)
            weight = self.weights.get(identifier, 1.0)
            
            means.append(distribution.mean)
            stdevs.append(distribution.stdev)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
        else:
            normalized_weights = [1.0 / len(weights)] * len(weights)
        
        # Calculate weighted mean
        weighted_mean = sum(m * w for m, w in zip(means, normalized_weights))
        
        # Calculate weighted standard deviation (using variance pooling)
        # This is a heuristic that accounts for both the variance within each forecast
        # and the variance between forecasts
        within_variance = sum((s**2) * w for s, w in zip(stdevs, normalized_weights))
        between_variance = sum(w * (m - weighted_mean)**2 for m, w in zip(means, normalized_weights))
        combined_variance = within_variance + between_variance
        combined_stdev = (combined_variance)**0.5
        
        # Create combined distribution
        combined_distribution = NumericDistribution(
            mean=weighted_mean,
            stdev=combined_stdev
        )
        
        # Create combined reasoning
        combined_reasoning = self._create_ensemble_reasoning(forecasts)
        
        return ReasonedPrediction(
            prediction_value=combined_distribution,
            reasoning=combined_reasoning
        )
    
    async def _aggregate_multiple_choice_forecasts(
        self, 
        question: MultipleChoiceQuestion, 
        forecasts: Dict[str, ReasonedPrediction]
    ) -> ReasonedPrediction:
        """
        Aggregate multiple choice forecasts from ensemble members.
        
        Args:
            question: The multiple choice question
            forecasts: Dictionary mapping identifiers to reasoned predictions
            
        Returns:
            An aggregated reasoned prediction
        """
        # Extract probability values for each option
        option_probabilities = []
        weights = []
        
        for identifier, forecast in forecasts.items():
            # Get the probabilities
            probabilities = forecast.prediction_value.probabilities
            
            # Get the weight (default to 1.0)
            weight = self.weights.get(identifier, 1.0)
            
            option_probabilities.append(probabilities)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
        else:
            normalized_weights = [1.0 / len(weights)] * len(weights)
        
        # Calculate weighted probabilities
        num_options = len(option_probabilities[0])
        weighted_probabilities = [0.0] * num_options
        
        for probs, weight in zip(option_probabilities, normalized_weights):
            for i, prob in enumerate(probs):
                weighted_probabilities[i] += prob * weight
        
        # Create combined prediction
        combined_prediction = PredictedOptionList(
            probabilities=weighted_probabilities,
            option_names=question.option_names
        )
        
        # Create combined reasoning
        combined_reasoning = self._create_ensemble_reasoning(forecasts)
        
        return ReasonedPrediction(
            prediction_value=combined_prediction,
            reasoning=combined_reasoning
        )
    
    def _create_ensemble_reasoning(
        self, forecasts: Dict[str, ReasonedPrediction]
    ) -> str:
        """
        Create a combined reasoning string from ensemble forecasts.
        
        Args:
            forecasts: Dictionary mapping identifiers to reasoned predictions
            
        Returns:
            Combined reasoning string
        """
        reasoning_parts = [
            "# Ensemble Forecast\n\n"
            "This forecast combines predictions from multiple forecasters with diverse personalities and approaches.\n\n"
        ]
        
        # Add each bot's reasoning
        for identifier, forecast in forecasts.items():
            bot_type, personality = identifier.split("_", 1)
            
            # Extract a summary of the reasoning (first paragraph or first 250 chars)
            reasoning_text = forecast.reasoning
            reasoning_summary = reasoning_text.split("\n\n")[0]
            if len(reasoning_summary) > 250:
                reasoning_summary = reasoning_summary[:247] + "..."
                
            # Format prediction value based on type
            if isinstance(forecast.prediction_value, float):
                # Binary prediction
                value_str = f"{forecast.prediction_value:.1%}"
            elif hasattr(forecast.prediction_value, "mean"):
                # Numeric prediction
                value_str = f"{forecast.prediction_value.mean} ± {forecast.prediction_value.stdev}"
            elif hasattr(forecast.prediction_value, "probabilities"):
                # Multiple choice prediction
                top_option_idx = forecast.prediction_value.probabilities.index(
                    max(forecast.prediction_value.probabilities)
                )
                top_option = forecast.prediction_value.option_names[top_option_idx]
                top_prob = forecast.prediction_value.probabilities[top_option_idx]
                value_str = f"{top_option} ({top_prob:.1%})"
            else:
                value_str = str(forecast.prediction_value)
                
            # Add to reasoning parts
            reasoning_parts.append(
                f"## {bot_type.capitalize()} Bot with {personality.capitalize()} Personality\n"
                f"**Prediction:** {value_str}\n"
                f"**Reasoning:** {reasoning_summary}\n"
            )
        
        # Add ensemble methodology explanation
        reasoning_parts.append(
            "## Ensemble Methodology\n"
            "The final prediction is a weighted average of all forecasts, accounting for "
            "both the central tendency and the spread of opinions. This approach captures "
            "diverse perspectives and helps mitigate individual biases.\n"
        )
        
        return "\n".join(reasoning_parts) 