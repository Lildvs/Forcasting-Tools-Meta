"""
Calibrated Bot

This module provides a specialized forecasting bot that adjusts its calibration
based on personality traits, particularly focusing on balancing caution and
creativity in its probability estimates.
"""

import asyncio
import logging
import json
import math
import re
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Union, cast

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
from forecasting_tools.forecast_bots.basic_bot import BasicBot
from forecasting_tools.forecast_helpers.prediction_extractor import PredictionExtractor
from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.processors import ProcessorFactory


logger = logging.getLogger(__name__)


class CalibratedBot(BasicBot):
    """
    A specialized forecasting bot that adjusts its calibration based on personality traits.
    
    This bot extends the BasicBot with calibration-focused features:
    - Explicit consideration of base rates and reference classes
    - Personality-driven calibration adjustments
    - Multi-stage forecasting with initial estimate and calibration steps
    - Confidence interval adjustments based on personality traits
    """
    
    def __init__(
        self,
        *,
        bot_name: str | None = None,
        research_reports_per_question: int = 1,
        predictions_per_research_report: int = 2,  # Make multiple predictions
        use_research_summary_to_forecast: bool = False,
        publish_reports_to_metaculus: bool = False,
        folder_to_save_reports_to: str | None = None,
        skip_previously_forecasted_questions: bool = False,
        llms: dict[str, str | GeneralLlm] | None = None,
        logger_name: Optional[str] = None,
        personality_name: str = "balanced",
        calibration_strength: float = 0.5,  # 0.0 = none, 1.0 = maximum
    ) -> None:
        """
        Initialize the CalibratedBot.
        
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
            calibration_strength: How strongly to apply calibration adjustments (0.0-1.0)
        """
        super().__init__(
            bot_name=bot_name or f"CalibratedBot_{personality_name}",
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
        
        self.calibration_strength = calibration_strength
        
        # Set calibration parameters based on personality traits
        self.calibration_params = self._initialize_calibration_params()
        
        logger.info(
            f"Initialized {self.bot_name} with personality: {personality_name}, "
            f"calibration_strength: {calibration_strength}, "
            f"calibration_params: {self.calibration_params}"
        )
    
    def _initialize_calibration_params(self) -> Dict[str, Any]:
        """
        Initialize calibration parameters based on personality traits.
        
        Returns:
            Dictionary of calibration parameters
        """
        params = {
            "binary_extremeness_adjustment": 0.0,  # -0.2 to 0.2, negative = more moderate
            "confidence_interval_width": 0.0,  # -0.2 to 0.2, negative = narrower
            "base_rate_weight": 0.5,  # 0.0 to 1.0, higher = more weight on base rates
            "explicit_calibration": True,  # Whether to do explicit calibration reasoning
        }
        
        # Adjust based on personality traits
        if self.personality.uncertainty_approach.value == "cautious":
            # Cautious: more moderate, wider intervals, more base rate weight
            params["binary_extremeness_adjustment"] = -0.15 * self.calibration_strength
            params["confidence_interval_width"] = 0.15 * self.calibration_strength
            params["base_rate_weight"] = 0.7
        elif self.personality.uncertainty_approach.value == "overconfident":
            # Overconfident: more extreme, narrower intervals, less base rate weight
            params["binary_extremeness_adjustment"] = 0.15 * self.calibration_strength
            params["confidence_interval_width"] = -0.15 * self.calibration_strength
            params["base_rate_weight"] = 0.3
            
        # Adjust based on thinking style
        if self.personality.thinking_style.value == "creative":
            # Creative: slightly more extreme, more varied intervals
            params["binary_extremeness_adjustment"] += 0.05 * self.calibration_strength
            params["confidence_interval_width"] += 0.05 * self.calibration_strength
            params["base_rate_weight"] -= 0.1
        elif self.personality.thinking_style.value == "bayesian":
            # Bayesian: more emphasis on base rates
            params["base_rate_weight"] += 0.2
            params["explicit_calibration"] = True
            
        # Ensure values stay within bounds
        params["binary_extremeness_adjustment"] = max(-0.2, min(0.2, params["binary_extremeness_adjustment"]))
        params["confidence_interval_width"] = max(-0.2, min(0.2, params["confidence_interval_width"]))
        params["base_rate_weight"] = max(0.0, min(1.0, params["base_rate_weight"]))
        
        return params
    
    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm]:
        """
        Get the default LLM configuration optimized for calibration.
        
        Returns:
            Dictionary mapping purpose to LLM
        """
        return {
            "default": "gpt-4o",
            "summarizer": "gpt-4o-mini",
            "calibrator": "gpt-4o",
        }
    
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        """
        Run a forecast on a binary question with calibration adjustments.
        
        Args:
            question: The binary question
            research: The research results
            
        Returns:
            A calibrated reasoned prediction
        """
        # First, generate an initial forecast
        initial_forecast = await self._generate_initial_binary_forecast(question, research)
        
        # Then apply calibration
        if self.calibration_params["explicit_calibration"]:
            calibrated_forecast = await self._calibrate_binary_forecast(
                question, research, initial_forecast
            )
            return calibrated_forecast
        else:
            # Apply simple calibration adjustment without a separate calibration step
            adjusted_prediction = self._adjust_binary_prediction(
                initial_forecast.prediction_value
            )
            return ReasonedPrediction(
                prediction_value=adjusted_prediction,
                reasoning=initial_forecast.reasoning + "\n\n(Prediction calibrated based on historical accuracy.)"
            )
    
    async def _generate_initial_binary_forecast(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        """
        Generate an initial binary forecast before calibration.
        
        Args:
            question: The binary question
            research: The research results
            
        Returns:
            An initial reasoned prediction
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
        
        # Add base rate instructions based on personality
        if self.personality.thinking_style.value == "bayesian" or self.calibration_params["base_rate_weight"] > 0.6:
            prompt += "\n\nIMPORTANT: Begin your analysis by establishing appropriate reference classes and base rates before considering specific evidence."
        
        # Process the prompt based on personality
        prompt = self.processor.process_prompt(prompt, question_type="binary")
        
        # Apply thinking configuration based on personality
        thinking_config = self.personality_manager.get_thinking_config()
        
        # Run the forecast
        reasoning = await self.get_llm("default", "llm").invoke(
            prompt, 
            **thinking_config
        )
        
        # Extract the prediction
        prediction = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=0.99, min_prediction=0.01
        )
        
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )
    
    async def _calibrate_binary_forecast(
        self, 
        question: BinaryQuestion, 
        research: str, 
        initial_forecast: ReasonedPrediction[float]
    ) -> ReasonedPrediction[float]:
        """
        Apply explicit calibration to a binary forecast.
        
        Args:
            question: The binary question
            research: The research results
            initial_forecast: The initial forecast
            
        Returns:
            A calibrated reasoned prediction
        """
        # Get the calibration prompt from the personality manager
        calibration_prompt = self.personality_manager.get_prompt(
            "calibration_prompt",
            question_text=question.question_text,
            initial_probability=f"{initial_forecast.prediction_value:.1%}",
            initial_reasoning=initial_forecast.reasoning,
            current_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Customize calibration approach based on personality traits
        if self.personality.uncertainty_approach.value == "cautious":
            calibration_prompt += "\n\nConsider whether you might be overconfident. Carefully examine potential biases that could lead to extreme probabilities."
        elif self.personality.thinking_style.value == "creative":
            calibration_prompt += "\n\nConsider creative scenarios that might contradict your initial assessment."
        elif self.personality.thinking_style.value == "bayesian":
            calibration_prompt += "\n\nReassess your prior probabilities and likelihood ratios to ensure proper Bayesian updating."
        
        # Apply thinking configuration based on personality
        thinking_config = self.personality_manager.get_thinking_config()
        
        # Run the calibration
        calibration_reasoning = await self.get_llm("calibrator", "llm").invoke(
            calibration_prompt,
            **thinking_config
        )
        
        # Extract the calibrated prediction
        calibrated_prediction = PredictionExtractor.extract_last_percentage_value(
            calibration_reasoning, max_prediction=0.99, min_prediction=0.01
        )
        
        # Combine the reasoning
        combined_reasoning = (
            f"## Initial Assessment\n{initial_forecast.reasoning}\n\n"
            f"## Calibration\n{calibration_reasoning}"
        )
        
        return ReasonedPrediction(
            prediction_value=calibrated_prediction, 
            reasoning=combined_reasoning
        )
    
    def _adjust_binary_prediction(self, prediction: float) -> float:
        """
        Apply a simple calibration adjustment to a binary prediction.
        
        Args:
            prediction: The original prediction (0-1)
            
        Returns:
            The adjusted prediction (0-1)
        """
        # No adjustment if extremeness_adjustment is 0
        if self.calibration_params["binary_extremeness_adjustment"] == 0:
            return prediction
            
        # Convert to log odds for better adjustment of extreme probabilities
        def to_log_odds(p: float) -> float:
            # Clamp to prevent infinity
            p = max(0.001, min(0.999, p))
            return math.log(p / (1 - p))
            
        def from_log_odds(lo: float) -> float:
            return 1 / (1 + math.exp(-lo))
        
        # Convert to log odds
        log_odds = to_log_odds(prediction)
        
        # Apply adjustment: negative adjustment makes log odds closer to 0 (more moderate)
        adjustment_factor = 1.0 - abs(self.calibration_params["binary_extremeness_adjustment"])
        adjusted_log_odds = log_odds * adjustment_factor
        
        # Convert back to probability
        adjusted_prediction = from_log_odds(adjusted_log_odds)
        
        # Ensure within bounds
        return max(0.01, min(0.99, adjusted_prediction))
    
    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        """
        Run a forecast on a multiple choice question with calibration adjustments.
        
        Args:
            question: The multiple choice question
            research: The research results
            
        Returns:
            A calibrated reasoned prediction
        """
        # First, generate an initial forecast
        initial_forecast = await self._generate_initial_multiple_choice_forecast(question, research)
        
        # Then apply calibration
        if self.calibration_params["explicit_calibration"]:
            calibrated_forecast = await self._calibrate_multiple_choice_forecast(
                question, research, initial_forecast
            )
            return calibrated_forecast
        else:
            # Apply simple calibration adjustment without a separate calibration step
            adjusted_prediction = self._adjust_multiple_choice_prediction(
                initial_forecast.prediction_value
            )
            return ReasonedPrediction(
                prediction_value=adjusted_prediction,
                reasoning=initial_forecast.reasoning + "\n\n(Prediction calibrated based on historical accuracy.)"
            )
    
    async def _generate_initial_multiple_choice_forecast(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        """
        Generate an initial multiple choice forecast before calibration.
        
        Args:
            question: The multiple choice question
            research: The research results
            
        Returns:
            An initial reasoned prediction
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
    
    async def _calibrate_multiple_choice_forecast(
        self, 
        question: MultipleChoiceQuestion, 
        research: str, 
        initial_forecast: ReasonedPrediction[PredictedOptionList]
    ) -> ReasonedPrediction[PredictedOptionList]:
        """
        Apply explicit calibration to a multiple choice forecast.
        
        Args:
            question: The multiple choice question
            research: The research results
            initial_forecast: The initial forecast
            
        Returns:
            A calibrated reasoned prediction
        """
        # Format initial probabilities for the prompt
        initial_probs_str = ""
        for i, (name, prob) in enumerate(zip(
            initial_forecast.prediction_value.option_names,
            initial_forecast.prediction_value.probabilities
        )):
            initial_probs_str += f"Option {i+1}: {name} - {prob:.1%}\n"
        
        # Get the calibration prompt
        calibration_prompt = self.personality_manager.get_prompt(
            "multiple_choice_calibration_prompt",
            question_text=question.question_text,
            options=", ".join([opt.option for opt in question.options]),
            initial_probabilities=initial_probs_str,
            initial_reasoning=initial_forecast.reasoning,
            current_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Apply thinking configuration based on personality
        thinking_config = self.personality_manager.get_thinking_config()
        
        # Run the calibration
        calibration_reasoning = await self.get_llm("calibrator", "llm").invoke(
            calibration_prompt,
            **thinking_config
        )
        
        # Extract calibrated probabilities
        calibrated_probabilities = PredictionExtractor.extract_probabilities_for_options(
            calibration_reasoning,
            [opt.option for opt in question.options],
        )
        
        # Combine the reasoning
        combined_reasoning = (
            f"## Initial Assessment\n{initial_forecast.reasoning}\n\n"
            f"## Calibration\n{calibration_reasoning}"
        )
        
        # Create calibrated prediction
        calibrated_predictions_list = PredictedOptionList(
            probabilities=calibrated_probabilities, 
            option_names=question.option_names
        )
        
        return ReasonedPrediction(
            prediction_value=calibrated_predictions_list, 
            reasoning=combined_reasoning
        )
    
    def _adjust_multiple_choice_prediction(
        self, prediction: PredictedOptionList
    ) -> PredictedOptionList:
        """
        Apply a simple calibration adjustment to a multiple choice prediction.
        
        Args:
            prediction: The original prediction
            
        Returns:
            The adjusted prediction
        """
        # If no adjustment needed, return original
        if self.calibration_params["binary_extremeness_adjustment"] == 0:
            return prediction
            
        # Calculate adjustment direction: negative = more uniform distribution
        direction = self.calibration_params["binary_extremeness_adjustment"]
        strength = abs(direction) * 2  # Scale up for more noticeable effect
        
        # Get original probabilities
        original_probs = prediction.probabilities.copy()
        n_options = len(original_probs)
        
        # Create uniform distribution
        uniform_prob = 1.0 / n_options
        uniform_probs = [uniform_prob] * n_options
        
        # Interpolate between original and uniform based on direction and strength
        if direction < 0:  # Move toward uniform (more conservative)
            weight_uniform = min(1.0, strength)
            weight_original = 1.0 - weight_uniform
            adjusted_probs = [
                (p * weight_original) + (uniform_prob * weight_uniform)
                for p in original_probs
            ]
        else:  # Move away from uniform (more extreme)
            # Amplify differences from uniform
            adjusted_probs = []
            for p in original_probs:
                diff = p - uniform_prob
                adjusted = p + (diff * strength)
                adjusted_probs.append(adjusted)
                
        # Normalize to ensure sum is 1.0
        total = sum(adjusted_probs)
        if total > 0:
            adjusted_probs = [p / total for p in adjusted_probs]
        else:
            adjusted_probs = original_probs.copy()
        
        # Create new prediction with adjusted probabilities
        return PredictedOptionList(
            probabilities=adjusted_probs,
            option_names=prediction.option_names
        )
    
    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        """
        Run a forecast on a numeric question with calibration adjustments.
        
        Args:
            question: The numeric question
            research: The research results
            
        Returns:
            A calibrated reasoned prediction
        """
        # First, generate an initial forecast
        initial_forecast = await self._generate_initial_numeric_forecast(question, research)
        
        # Then apply calibration
        if self.calibration_params["explicit_calibration"]:
            calibrated_forecast = await self._calibrate_numeric_forecast(
                question, research, initial_forecast
            )
            return calibrated_forecast
        else:
            # Apply simple calibration adjustment without a separate calibration step
            adjusted_prediction = self._adjust_numeric_prediction(
                initial_forecast.prediction_value
            )
            return ReasonedPrediction(
                prediction_value=adjusted_prediction,
                reasoning=initial_forecast.reasoning + "\n\n(Prediction calibrated with adjusted confidence intervals.)"
            )
    
    async def _generate_initial_numeric_forecast(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        """
        Generate an initial numeric forecast before calibration.
        
        Args:
            question: The numeric question
            research: The research results
            
        Returns:
            An initial reasoned prediction
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
        
        # Extract prediction
        prediction = PredictionExtractor.extract_normal_distribution(reasoning)
        
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )
    
    async def _calibrate_numeric_forecast(
        self, 
        question: NumericQuestion, 
        research: str, 
        initial_forecast: ReasonedPrediction[NumericDistribution]
    ) -> ReasonedPrediction[NumericDistribution]:
        """
        Apply explicit calibration to a numeric forecast.
        
        Args:
            question: The numeric question
            research: The research results
            initial_forecast: The initial forecast
            
        Returns:
            A calibrated reasoned prediction
        """
        # Format the initial distribution for the prompt
        dist = initial_forecast.prediction_value
        initial_dist_str = (
            f"Mean: {dist.mean} {question.unit_of_measure}\n"
            f"Standard Deviation: {dist.stdev} {question.unit_of_measure}\n"
            f"90% Confidence Interval: {dist.mean - 1.645*dist.stdev} to {dist.mean + 1.645*dist.stdev} {question.unit_of_measure}"
        )
        
        # Get the calibration prompt
        calibration_prompt = self.personality_manager.get_prompt(
            "numeric_calibration_prompt",
            question_text=question.question_text,
            initial_distribution=initial_dist_str,
            initial_reasoning=initial_forecast.reasoning,
            unit_of_measure=question.unit_of_measure,
            current_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Customize calibration based on personality
        if self.personality.uncertainty_approach.value == "cautious":
            calibration_prompt += "\n\nConsider whether your confidence interval is too narrow. Most forecasters tend to be overconfident."
        elif self.personality.thinking_style.value == "creative":
            calibration_prompt += "\n\nConsider unusual or unexpected scenarios that might push the value outside your estimated range."
        
        # Apply thinking configuration based on personality
        thinking_config = self.personality_manager.get_thinking_config()
        
        # Run the calibration
        calibration_reasoning = await self.get_llm("calibrator", "llm").invoke(
            calibration_prompt,
            **thinking_config
        )
        
        # Extract calibrated distribution
        calibrated_distribution = PredictionExtractor.extract_normal_distribution(calibration_reasoning)
        
        # Combine the reasoning
        combined_reasoning = (
            f"## Initial Assessment\n{initial_forecast.reasoning}\n\n"
            f"## Calibration\n{calibration_reasoning}"
        )
        
        return ReasonedPrediction(
            prediction_value=calibrated_distribution, 
            reasoning=combined_reasoning
        )
    
    def _adjust_numeric_prediction(
        self, prediction: NumericDistribution
    ) -> NumericDistribution:
        """
        Apply a simple calibration adjustment to a numeric prediction.
        
        Args:
            prediction: The original prediction
            
        Returns:
            The adjusted prediction
        """
        # If no adjustment needed, return original
        if self.calibration_params["confidence_interval_width"] == 0:
            return prediction
            
        # Calculate adjustment factor for standard deviation
        # Positive width adjustment = wider intervals
        width_adjustment = 1.0 + self.calibration_params["confidence_interval_width"] * 2
        
        # Adjust standard deviation
        adjusted_stdev = prediction.stdev * width_adjustment
        
        # Create adjusted distribution
        return NumericDistribution(
            mean=prediction.mean,
            stdev=adjusted_stdev
        ) 