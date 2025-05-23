"""
Personality-Aware Bot

This module provides a forecast bot that is explicitly designed to work with
the personality management system and can adapt its forecasting approach
based on different personality traits.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.forecast_report import (
    ReasonedPrediction,
)
from forecasting_tools.data_models.numeric_report import NumericDistribution
from forecasting_tools.data_models.multiple_choice_report import PredictedOptionList
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.ai_models.deprecated_model_classes.perplexity import Perplexity
from forecasting_tools.forecast_helpers.prediction_extractor import PredictionExtractor
from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.processors import ProcessorFactory


logger = logging.getLogger(__name__)


class PersonalityAwareBot(ForecastBot):
    """
    A forecast bot that is aware of and can adapt to different personality configurations.
    
    This bot uses the personality management system to customize prompts,
    research approaches, and forecasting methods based on the selected personality.
    """
    
    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    
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
        personality_name: str = "balanced",
        research_type: str = "perplexity",
    ) -> None:
        """
        Initialize the PersonalityAwareBot.
        
        Args:
            research_reports_per_question: Number of research reports to generate per question
            predictions_per_research_report: Number of predictions to make per research report
            use_research_summary_to_forecast: Whether to use a summary of research for forecasting
            publish_reports_to_metaculus: Whether to publish reports to Metaculus
            folder_to_save_reports_to: Folder to save reports to
            skip_previously_forecasted_questions: Whether to skip questions that have been previously forecasted
            llms: Dictionary of LLMs to use for different purposes
            logger_name: Name for the logger
            personality_name: Name of the personality to use
            research_type: Type of research to use ("perplexity", "smart_search", or "default")
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
        
        # Initialize the personality manager
        self.personality_manager = PersonalityManager(personality_name=personality_name)
        self.personality = self.personality_manager.get_current_personality()
        
        # Create the processor for the personality
        self.processor = ProcessorFactory.create_processor(self.personality)
        
        # Set the research type
        self.research_type = research_type
        
        logger.info(
            f"Initialized PersonalityAwareBot with personality: {personality_name}, "
            f"research_type: {research_type}"
        )
    
    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm]:
        """
        Get the default LLM configuration.
        
        Returns:
            Dictionary mapping purpose to LLM
        """
        return {
            "default": GeneralLlm(model="gpt-4o", temperature=0.2),
            "summarizer": GeneralLlm(model="gpt-4o-mini", temperature=0.1),
            "research": GeneralLlm(model="gpt-4o", temperature=0.3),
        }
    
    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Run research on a question using the personality-specific approach.
        
        Args:
            question: The question to research
            
        Returns:
            The research results
        """
        # Process the research query based on personality
        processed_query = self.processor.process_research_query(question)
        
        # Get the research prompt from the personality manager
        system_prompt = self.personality_manager.get_prompt(
            "research_prompt", 
            question_text=processed_query
        )
        
        # Use the appropriate research method based on configuration
        if self.research_type == "perplexity":
            # Use Perplexity with our personality-specific system prompt
            response = await Perplexity(
                temperature=0.2, 
                system_prompt=system_prompt
            ).invoke(processed_query)
        else:
            # Use the default LLM for research
            response = await self.get_llm("research", "llm").invoke(
                f"System: {system_prompt}\n\nUser: {processed_query}\n\nAssistant:"
            )
        
        return response
    
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        """
        Run a forecast on a binary question using the personality-specific approach.
        
        Args:
            question: The binary question
            research: The research results
            
        Returns:
            A reasoned prediction
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
        
        # Apply personality-specific processing
        prompt = self.processor.process_prompt(prompt, question_type="binary")
        
        # Apply thinking configuration based on personality
        thinking_config = self.personality_manager.get_thinking_config()
        
        # Run the forecast
        reasoning = await self.get_llm("default", "llm").invoke(
            prompt, 
            **thinking_config
        )
        
        # Extract prediction with appropriate bounds based on personality
        min_prediction = 0.01
        max_prediction = 0.99
        
        # Adjust bounds based on personality traits
        if self.personality.uncertainty_approach.value == "overconfident":
            min_prediction = 0.001
            max_prediction = 0.999
        elif self.personality.uncertainty_approach.value == "cautious":
            min_prediction = 0.05
            max_prediction = 0.95
        
        prediction = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=max_prediction, min_prediction=min_prediction
        )
        
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )
    
    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        """
        Run a forecast on a multiple choice question using the personality-specific approach.
        
        Args:
            question: The multiple choice question
            research: The research results
            
        Returns:
            A reasoned prediction
        """
        # Get the multiple choice forecast prompt from the personality manager
        prompt = self.personality_manager.get_prompt(
            "multiple_choice_prompt",
            question_text=question.question_text,
            options=", ".join([opt.option for opt in question.options]),
            background_info=question.background_info,
            resolution_criteria=question.resolution_criteria,
            fine_print=question.fine_print,
            research=research,
            current_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Apply personality-specific processing
        prompt = self.processor.process_prompt(prompt, question_type="multiple_choice")
        
        # Apply thinking configuration based on personality
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
        Run a forecast on a numeric question using the personality-specific approach.
        
        Args:
            question: The numeric question
            research: The research results
            
        Returns:
            A reasoned prediction
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
        
        # Apply personality-specific processing
        prompt = self.processor.process_prompt(prompt, question_type="numeric")
        
        # Apply thinking configuration based on personality
        thinking_config = self.personality_manager.get_thinking_config()
        
        # Run the forecast
        reasoning = await self.get_llm("default", "llm").invoke(
            prompt,
            **thinking_config
        )
        
        # Process the forecast output based on personality if needed
        reasoning = self.processor.process_forecast_output(reasoning, "numeric")
        
        # Extract prediction
        num_prediction = PredictionExtractor.extract_percentile_distribution(
            reasoning
        )
        
        return ReasonedPrediction(
            prediction_value=num_prediction, reasoning=reasoning
        ) 