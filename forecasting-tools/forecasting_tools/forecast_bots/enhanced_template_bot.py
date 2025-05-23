"""
Enhanced Template Bot

This module provides an enhanced template bot that uses the personality templates system
for more flexible and configurable forecasting.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.forecast_report import (
    PredictedOptionList,
    ReasonedPrediction,
)
from forecasting_tools.data_models.numeric_report import NumericDistribution
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_helpers.perplexity_searcher import Perplexity
from forecasting_tools.forecast_helpers.prediction_extractor import PredictionExtractor
from forecasting_tools.personality_templates import PersonalityManager

logger = logging.getLogger(__name__)


class EnhancedTemplateBot(ForecastBot):
    """
    An enhanced template bot that uses the personality templates system
    for more flexible and configurable forecasting.
    """

    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    RESEARCH_TYPE = "default_research"  # can be "default_research", "perplexity_research", or "smart_searcher_research"

    def __init__(
        self,
        *,
        research_reports_per_question: int = 1,
        predictions_per_research_report: int = 1,
        use_research_summary_to_forecast: bool = False,
        publish_reports_to_metaculus: bool = False,
        folder_to_save_reports_to: str | None = None,
        skip_previously_forecasted_questions: bool = False,
        llms: dict[str, str | GeneralLlm] | None = None,
        logger_name: Optional[str] = None,
        personality_name: Optional[str] = None,
        bot_version: str = "q2",
    ) -> None:
        super().__init__(
            research_reports_per_question=research_reports_per_question,
            predictions_per_research_report=predictions_per_research_report,
            use_research_summary_to_forecast=use_research_summary_to_forecast,
            publish_reports_to_metaculus=publish_reports_to_metaculus,
            folder_to_save_reports_to=folder_to_save_reports_to,
            skip_previously_forecasted_questions=skip_previously_forecasted_questions,
            llms=llms,
            logger_name=logger_name,
            personality_name=personality_name,
        )
        
        # Initialize the personality manager
        self.personality_manager = PersonalityManager(
            bot_version=bot_version,
            personality_name=personality_name,
            research_type=self.RESEARCH_TYPE
        )
        
        logger.info(f"Initialized EnhancedTemplateBot with personality: {personality_name or 'balanced'}, bot_version: {bot_version}")

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm]:
        return {
            "default": GeneralLlm(model="gpt-4o", temperature=0.1),
            "summarizer": GeneralLlm(model="gpt-4o-mini", temperature=0.1),
        }

    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Run research on a question using Perplexity with the personality-based prompt.
        """
        # Get the research prompt from the personality manager
        system_prompt = self.personality_manager.get_prompt("research_prompt", question=question.question_text)
        
        # Use Perplexity for research with our system prompt
        response = await Perplexity(
            temperature=0.1, system_prompt=system_prompt
        ).invoke(question.question_text)
        
        return response

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        """
        Run a forecast on a binary question using the personality-based prompt.
        """
        # Get the binary forecast prompt from the personality manager
        prompt = self.personality_manager.get_prompt(
            "binary_forecast_prompt",
            question_text=question.question_text,
            background_info=question.background_info,
            resolution_criteria=question.resolution_criteria,
            fine_print=question.fine_print,
            research=research,
            current_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Apply thinking configuration if available
        thinking_config = self.personality_manager.get_thinking_config()
        
        # Run the forecast
        reasoning = await self.get_llm("default", "llm").invoke(
            prompt, 
            **thinking_config
        )
        
        # Extract prediction
        prediction = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=0.99, min_prediction=0.01
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        """
        Run a forecast on a multiple choice question using the personality-based prompt.
        """
        # Get the multiple choice forecast prompt from the personality manager
        prompt = self.personality_manager.get_prompt(
            "multiple_choice_forecast_prompt",
            question_text=question.question_text,
            options=", ".join([opt.option for opt in question.options]),
            background_info=question.background_info,
            resolution_criteria=question.resolution_criteria,
            fine_print=question.fine_print,
            research=research,
            current_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Apply thinking configuration if available
        thinking_config = self.personality_manager.get_thinking_config()
        
        # Run the forecast
        reasoning = await self.get_llm("default", "llm").invoke(
            prompt,
            **thinking_config
        )
        
        # Extract prediction
        probabilities = PredictionExtractor.extract_probabilities_for_options(
            reasoning,
            [opt.option for opt in question.options],
        )
        
        predictions_list = PredictedOptionList(
            probabilities=probabilities, option_names=question.option_names
        )
        return ReasonedPrediction(
            prediction_value=predictions_list, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        """
        Run a forecast on a numeric question using the personality-based prompt.
        """
        # Get lower/upper bound messages
        lower_bound_message = ""
        upper_bound_message = ""
        
        if question.has_lower_bound:
            lower_bound_message = f"Lower bound: {question.lower_bound} {question.unit_of_measure}"
        if question.has_upper_bound:
            upper_bound_message = f"Upper bound: {question.upper_bound} {question.unit_of_measure}"
        
        # Get the numeric forecast prompt from the personality manager
        prompt = self.personality_manager.get_prompt(
            "numeric_forecast_prompt",
            question_text=question.question_text,
            background_info=question.background_info,
            resolution_criteria=question.resolution_criteria,
            fine_print=question.fine_print,
            unit_of_measure=question.unit_of_measure,
            lower_bound_message=lower_bound_message,
            upper_bound_message=upper_bound_message,
            research=research,
            current_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Apply thinking configuration if available
        thinking_config = self.personality_manager.get_thinking_config()
        
        # Run the forecast
        reasoning = await self.get_llm("default", "llm").invoke(
            prompt,
            **thinking_config
        )
        
        # Extract prediction
        try:
            percentiles = PredictionExtractor.extract_numbered_percentiles(
                reasoning,
                expected_percentiles=[10, 20, 40, 60, 80, 90],
                min_percentile_value=question.lower_bound if question.has_lower_bound else None,
                max_percentile_value=question.upper_bound if question.has_upper_bound else None,
            )
            
            prediction = NumericDistribution(
                percentile_values=percentiles
            )
            return ReasonedPrediction(
                prediction_value=prediction, reasoning=reasoning
            )
        except Exception as e:
            logger.error(f"Error extracting percentiles: {e}")
            raise ValueError(f"Failed to extract percentiles from reasoning: {e}")


class EnhancedTemplateBotQ1(EnhancedTemplateBot):
    """Q1 version of the enhanced template bot."""
    
    def __init__(self, personality_name: Optional[str] = None, **kwargs):
        super().__init__(bot_version="q1", personality_name=personality_name, **kwargs)


class EnhancedTemplateBotQ2(EnhancedTemplateBot):
    """Q2 version of the enhanced template bot."""
    
    def __init__(self, personality_name: Optional[str] = None, **kwargs):
        super().__init__(bot_version="q2", personality_name=personality_name, **kwargs)


class EnhancedTemplateBotQ3(EnhancedTemplateBot):
    """Q3 version of the enhanced template bot."""
    
    def __init__(self, personality_name: Optional[str] = None, **kwargs):
        super().__init__(bot_version="q3", personality_name=personality_name, **kwargs)


class EnhancedTemplateBotQ4(EnhancedTemplateBot):
    """Q4 version of the enhanced template bot with Veritas personality."""
    
    def __init__(self, personality_name: Optional[str] = None, **kwargs):
        super().__init__(bot_version="q4", personality_name=personality_name, **kwargs)
        
    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm]:
        # Use GPT-4 with higher temperature for Q4 Veritas-style bot
        return {
            "default": GeneralLlm(model="gpt-4", temperature=0.3),
            "summarizer": GeneralLlm(model="gpt-4o-mini", temperature=0.1),
        } 