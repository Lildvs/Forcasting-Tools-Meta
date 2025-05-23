"""
Basic Bot

This module provides a basic forecasting bot that leverages personality traits
for its reasoning process. It serves as a simple implementation of the forecast
bot concept with personality-driven forecasting capabilities.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.forecast_report import (
    NumericDistribution,
    PredictedOptionList,
    ReasonedPrediction,
)
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_helpers.prediction_extractor import PredictionExtractor
from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.processors import ProcessorFactory


logger = logging.getLogger(__name__)


class BasicBot(ForecastBot):
    """
    A basic implementation of a forecast bot that leverages personality traits
    for its reasoning process.
    
    This bot adjusts its reasoning approach based on personality traits like
    risk tolerance, creativity, and uncertainty handling.
    """
    
    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    
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
        personality_name: str = "balanced",
    ) -> None:
        """
        Initialize the BasicBot.
        
        Args:
            bot_name: Optional name for the bot
            research_reports_per_question: Number of research reports to generate per question
            predictions_per_research_report: Number of predictions to make per research report
            use_research_summary_to_forecast: Whether to use a summary of research for forecasting
            publish_reports_to_metaculus: Whether to publish reports to Metaculus
            folder_to_save_reports_to: Folder to save reports to
            skip_previously_forecasted_questions: Whether to skip questions that have been previously forecasted
            llms: Dictionary of LLMs to use for different purposes
            logger_name: Name for the logger
            personality_name: Name of the personality to use
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
            personality_name=personality_name,
        )
        
        self.bot_name = bot_name or f"BasicBot_{personality_name}"
        
        # Initialize the personality traits processor
        self.personality = self.personality_manager.get_current_personality()
        self.processor = ProcessorFactory.create_processor(self.personality)
        
        logger.info(f"Initialized {self.bot_name} with personality: {personality_name}")
    
    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm]:
        """
        Get the default LLM configuration.
        
        Returns:
            Dictionary mapping purpose to LLM
        """
        return {
            "default": "gpt-4o",
            "summarizer": "gpt-4o-mini",
        }
    
    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Run research on a question using the integrated research pipeline.
        
        This implementation:
        1. Uses the search infrastructure for web research via research_question
        2. Applies personality-specific processing to the query
        3. Formats the results according to personality traits
        
        Args:
            question: The question to research
            
        Returns:
            Research results as a formatted string
        """
        # Process the research query based on personality
        processed_query = self.processor.process_research_query(question)
        
        # Log research attempt
        logger.info(f"Running research for question {question.question_id} with {self.personality_manager.personality_name} personality")
        
        # Select search depth based on personality traits
        search_depth = "medium"  # Default
        if hasattr(self.personality, "traits"):
            traits = self.personality.traits
            if "thoroughness" in traits:
                thoroughness = traits["thoroughness"].value
                if isinstance(thoroughness, (int, float)):
                    if thoroughness > 0.7:
                        search_depth = "high"
                    elif thoroughness < 0.3:
                        search_depth = "low"
        
        # Select search type based on personality and question complexity
        search_type = "basic"
        if question.resolution_criteria and len(question.resolution_criteria) > 100:
            # More complex questions need deeper research
            search_type = "deep"
        
        # Perform the research using integrated search
        try:
            research_results = await self.research_question(
                question,
                search_type=search_type,
                search_depth=search_depth
            )
            
            # If research failed or returned empty results, fall back to traditional method
            if research_results.startswith("ERROR:") or not research_results.strip():
                logger.warning(f"Web research failed for question {question.question_id}, falling back to non-web research")
                research_results = await self._fallback_research(question)
        except Exception as e:
            logger.error(f"Research failed with error: {e}")
            research_results = await self._fallback_research(question)
        
        # Format the research based on personality
        formatted_research = self._format_research_results(question, research_results)
        
        return formatted_research
    
    async def _fallback_research(self, question: MetaculusQuestion) -> str:
        """
        Fallback research method when web research fails.
        
        Args:
            question: The question to research
            
        Returns:
            Research results as a string
        """
        # Get the personality-specific system prompt
        research_prompt = self.personality_manager.get_prompt(
            "research_prompt",
            question_text=question.question_text,
            background_info=question.background_info or "",
            resolution_criteria=question.resolution_criteria or "",
            current_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Process the query based on personality
        processed_query = self.processor.process_research_query(question)
        
        # Get thinking configuration based on personality
        thinking_config = self.personality_manager.get_thinking_config()
        
        # Run the research using the LLM directly
        research_results = await self.get_llm("default", "llm").invoke(
            f"{research_prompt}\n\n{processed_query}",
            **thinking_config
        )
        
        return research_results
    
    def _format_research_results(self, question: MetaculusQuestion, research: str) -> str:
        """
        Format research results based on personality traits.
        
        Args:
            question: The question that was researched
            research: The raw research results
            
        Returns:
            Formatted research results
        """
        # Add headers and structure
        formatted_research = f"# Research on: {question.question_text}\n\n"
        
        # Add date and source information
        formatted_research += f"*Research conducted on: {datetime.now().strftime('%Y-%m-%d')}*\n\n"
        
        # Add the actual research content
        formatted_research += research
        
        return formatted_research
    
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        """
        Run a forecast on a binary question with personality-aware reasoning.
        
        Args:
            question: The binary question
            research: The research results
            
        Returns:
            A reasoned prediction with probability between 0 and 1
        """
        # Get the binary forecast prompt from the personality manager
        prompt = self.personality_manager.get_prompt(
            "binary_forecast_prompt",
            question_text=question.question_text,
            background_info=question.background_info or "",
            resolution_criteria=question.resolution_criteria or "",
            fine_print=question.fine_print or "",
            research=research,
            current_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Process the prompt based on personality
        prompt = self.processor.process_prompt(prompt, question_type="binary")
        
        # Apply thinking configuration based on personality
        thinking_config = self.personality_manager.get_thinking_config()
        
        # Run the forecast
        reasoning = await self.get_llm("default", "llm").invoke(
            prompt, 
            **thinking_config
        )
        
        # Set prediction bounds based on personality traits
        min_prediction = 0.01
        max_prediction = 0.99
        
        # Adjust bounds based on personality traits (more cautious = narrower bounds)
        if self.personality.traits.get("risk_tolerance"):
            risk_tolerance = self.personality.traits["risk_tolerance"].value
            if isinstance(risk_tolerance, (int, float)) and 0 <= risk_tolerance <= 1:
                # Higher risk tolerance = wider bounds (more extreme probabilities)
                if risk_tolerance > 0.7:  # High risk tolerance
                    min_prediction = 0.001
                    max_prediction = 0.999
                elif risk_tolerance < 0.3:  # Low risk tolerance
                    min_prediction = 0.05
                    max_prediction = 0.95
        
        # Extract the prediction
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
        Run a forecast on a multiple choice question with personality-aware reasoning.
        
        Args:
            question: The multiple choice question
            research: The research results
            
        Returns:
            A reasoned prediction with probabilities for each option
        """
        # Get the multiple choice forecast prompt
        prompt = self.personality_manager.get_prompt(
            "multiple_choice_forecast_prompt",
            question_text=question.question_text,
            options=", ".join([opt.option for opt in question.options]),
            background_info=question.background_info or "",
            resolution_criteria=question.resolution_criteria or "",
            fine_print=question.fine_print or "",
            research=research,
            current_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Process the prompt based on personality
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
        Run a forecast on a numeric question with personality-aware reasoning.
        
        Args:
            question: The numeric question
            research: The research results
            
        Returns:
            A reasoned prediction with a distribution
        """
        # Get lower/upper bound messages
        lower_bound_message = ""
        upper_bound_message = ""
        
        if question.has_lower_bound:
            lower_bound_message = f"Lower bound: {question.lower_bound} {question.unit_of_measure}"
        if question.has_upper_bound:
            upper_bound_message = f"Upper bound: {question.upper_bound} {question.unit_of_measure}"
        
        # Get the numeric forecast prompt
        prompt = self.personality_manager.get_prompt(
            "numeric_forecast_prompt",
            question_text=question.question_text,
            background_info=question.background_info or "",
            resolution_criteria=question.resolution_criteria or "",
            fine_print=question.fine_print or "",
            unit_of_measure=question.unit_of_measure,
            lower_bound_message=lower_bound_message,
            upper_bound_message=upper_bound_message,
            research=research,
            current_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Process the prompt based on personality
        prompt = self.processor.process_prompt(prompt, question_type="numeric")
        
        # Apply thinking configuration based on personality
        thinking_config = self.personality_manager.get_thinking_config()
        
        # Run the forecast
        reasoning = await self.get_llm("default", "llm").invoke(
            prompt, 
            **thinking_config
        )
        
        # Adjust prediction extraction based on personality traits
        # (more creative = wider distributions, more cautious = narrower distributions)
        prediction = PredictionExtractor.extract_normal_distribution(reasoning)
        
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        ) 