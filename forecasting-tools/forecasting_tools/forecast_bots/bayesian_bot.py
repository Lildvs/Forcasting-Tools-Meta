"""
Bayesian Bot

This module provides a specialized forecasting bot that implements structured
Bayesian updating for generating forecasts. It explicitly models prior probabilities,
likelihood ratios, and updates beliefs based on new evidence.
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
from forecasting_tools.forecast_bots.calibrated_bot import CalibratedBot
from forecasting_tools.forecast_helpers.prediction_extractor import PredictionExtractor
from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.processors import BayesianProcessor


logger = logging.getLogger(__name__)


class BayesianBot(CalibratedBot):
    """
    A specialized forecasting bot that implements structured Bayesian updating.
    
    This bot extends the CalibratedBot with Bayesian-specific features:
    - Explicit modeling of prior probabilities
    - Structured evaluation of evidence
    - Likelihood ratio calculations
    - Formal Bayesian updating
    - Sensitivity analysis for priors and likelihoods
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
        personality_name: str = "bayesian",  # Default to bayesian personality
        calibration_strength: float = 0.5,
        sensitivity_analysis: bool = True,  # Whether to perform sensitivity analysis
        prior_source: str = "reference_class",  # reference_class, historical, model
    ) -> None:
        """
        Initialize the BayesianBot.
        
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
            sensitivity_analysis: Whether to perform sensitivity analysis on priors and likelihoods
            prior_source: Method for determining priors (reference_class, historical, model)
        """
        # Ensure bayesian personality is used if not explicitly overridden
        if personality_name != "bayesian":
            logger.info(f"Using specified personality '{personality_name}' instead of default 'bayesian'")
            
        super().__init__(
            bot_name=bot_name or f"BayesianBot_{personality_name}",
            research_reports_per_question=research_reports_per_question,
            predictions_per_research_report=predictions_per_research_report,
            use_research_summary_to_forecast=use_research_summary_to_forecast,
            publish_reports_to_metaculus=publish_reports_to_metaculus,
            folder_to_save_reports_to=folder_to_save_reports_to,
            skip_previously_forecasted_questions=skip_previously_forecasted_questions,
            llms=llms,
            logger_name=logger_name,
            personality_name=personality_name,
            calibration_strength=calibration_strength,
        )
        
        self.sensitivity_analysis = sensitivity_analysis
        self.prior_source = prior_source
        
        # Ensure we're using the BayesianProcessor
        if not isinstance(self.processor, BayesianProcessor):
            self.processor = BayesianProcessor(self.personality)
            
        logger.info(
            f"Initialized {self.bot_name} with sensitivity_analysis: {sensitivity_analysis}, "
            f"prior_source: {prior_source}"
        )
    
    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm]:
        """
        Get the default LLM configuration optimized for Bayesian reasoning.
        
        Returns:
            Dictionary mapping purpose to LLM
        """
        return {
            "default": GeneralLlm(model="gpt-4o", temperature=0.1),  # Lower temperature for more precise reasoning
            "summarizer": GeneralLlm(model="gpt-4o-mini", temperature=0.1),
            "calibrator": GeneralLlm(model="gpt-4o", temperature=0.1),
            "prior_estimator": GeneralLlm(model="gpt-4o", temperature=0.1),
            "likelihood_estimator": GeneralLlm(model="gpt-4o", temperature=0.1),
            "sensitivity_analyzer": GeneralLlm(model="gpt-4o", temperature=0.1),
        }
    
    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Run research with a focus on reference classes and base rates.
        
        Args:
            question: The question to research
            
        Returns:
            The research results
        """
        # Get basic research
        basic_research = await super().run_research(question)
        
        # If using reference classes for priors, gather additional information
        if self.prior_source == "reference_class":
            reference_class_info = await self._gather_reference_class_information(question)
            research = f"{basic_research}\n\n## Reference Class Information\n{reference_class_info}"
        elif self.prior_source == "historical":
            historical_data = await self._gather_historical_information(question)
            research = f"{basic_research}\n\n## Historical Data\n{historical_data}"
        else:
            research = basic_research
            
        return research
    
    async def _gather_reference_class_information(
        self, question: MetaculusQuestion
    ) -> str:
        """
        Gather information about relevant reference classes for the question.
        
        Args:
            question: The question to analyze
            
        Returns:
            Information about relevant reference classes
        """
        prompt = (
            f"Identify appropriate reference classes for the following forecasting question:\n\n"
            f"Question: {question.question_text}\n\n"
            f"Background: {question.background_info or 'Not provided'}\n\n"
            f"For each reference class you identify, provide:\n"
            f"1. A clear definition of the reference class\n"
            f"2. The base rate (historical frequency) of similar events in this reference class\n"
            f"3. The strength of analogy between this question and the reference class (strong, moderate, weak)\n"
            f"4. Any important caveats or adjustments needed when applying this reference class\n\n"
            f"Identify at least 3 reference classes if possible. Focus on reference classes with clear, "
            f"quantifiable base rates rather than vague analogies."
        )
        
        try:
            reference_class_info = await self.get_llm("prior_estimator", "llm").invoke(prompt)
            return reference_class_info
        except Exception as e:
            logger.error(f"Error gathering reference class information: {str(e)}")
            return "Error gathering reference class information."
    
    async def _gather_historical_information(
        self, question: MetaculusQuestion
    ) -> str:
        """
        Gather historical information relevant to the question.
        
        Args:
            question: The question to analyze
            
        Returns:
            Historical information relevant to the question
        """
        prompt = (
            f"Research and summarize historical data relevant to the following forecasting question:\n\n"
            f"Question: {question.question_text}\n\n"
            f"Background: {question.background_info or 'Not provided'}\n\n"
            f"Focus on:\n"
            f"1. Long-term historical trends in this domain\n"
            f"2. Frequency of similar events in the past\n"
            f"3. Quantitative data that could inform a prior probability\n"
            f"4. Relevant time series or frequency data\n\n"
            f"Provide specific numbers, frequencies, and probabilities whenever possible."
        )
        
        try:
            historical_info = await self.get_llm("prior_estimator", "llm").invoke(prompt)
            return historical_info
        except Exception as e:
            logger.error(f"Error gathering historical information: {str(e)}")
            return "Error gathering historical information."
    
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        """
        Run a forecast on a binary question using structured Bayesian updating.
        
        Args:
            question: The binary question
            research: The research results
            
        Returns:
            A reasoned prediction
        """
        # First, estimate a prior probability
        prior_probability, prior_reasoning = await self._estimate_prior_probability(question, research)
        
        # Then, identify key pieces of evidence
        evidence_list = await self._identify_key_evidence(question, research)
        
        # Apply Bayesian updating to each piece of evidence
        final_probability, updating_reasoning = await self._apply_bayesian_updating(
            question, prior_probability, evidence_list
        )
        
        # Optionally perform sensitivity analysis
        if self.sensitivity_analysis:
            sensitivity_reasoning = await self._perform_sensitivity_analysis(
                question, prior_probability, evidence_list, final_probability
            )
            combined_reasoning = (
                f"## Prior Probability Estimation\n{prior_reasoning}\n\n"
                f"## Bayesian Updating\n{updating_reasoning}\n\n"
                f"## Sensitivity Analysis\n{sensitivity_reasoning}"
            )
        else:
            combined_reasoning = (
                f"## Prior Probability Estimation\n{prior_reasoning}\n\n"
                f"## Bayesian Updating\n{updating_reasoning}"
            )
        
        return ReasonedPrediction(
            prediction_value=final_probability, reasoning=combined_reasoning
        )
    
    async def _estimate_prior_probability(
        self, question: BinaryQuestion, research: str
    ) -> Tuple[float, str]:
        """
        Estimate a prior probability for a binary question.
        
        Args:
            question: The binary question
            research: The research results
            
        Returns:
            Tuple of (prior probability, reasoning)
        """
        # Create a prompt for estimating the prior
        prompt = (
            f"Estimate a prior probability for the following forecasting question based on reference classes and base rates.\n\n"
            f"Question: {question.question_text}\n\n"
            f"Background: {question.background_info or 'Not provided'}\n\n"
            f"Resolution criteria: {question.resolution_criteria or 'Not provided'}\n\n"
            f"Research: {research[:5000]}...\n\n"  # Truncate research to avoid token limits
            f"Follow these steps:\n"
            f"1. Identify the most relevant reference classes for this question\n"
            f"2. Determine the base rates (historical frequencies) for these reference classes\n"
            f"3. Discuss how to weight and combine these reference classes\n"
            f"4. Estimate a prior probability based on these reference classes\n\n"
            f"Your response should be structured with clear headings for each step, and end with a single "
            f"probability estimate (a number between 0 and 1) that represents your prior probability."
        )
        
        # Get the prior estimation
        try:
            prior_reasoning = await self.get_llm("prior_estimator", "llm").invoke(prompt)
            
            # Extract the prior probability
            prior_probability = PredictionExtractor.extract_last_percentage_value(
                prior_reasoning, max_prediction=0.99, min_prediction=0.01
            )
            
            return prior_probability, prior_reasoning
        except Exception as e:
            logger.error(f"Error estimating prior probability: {str(e)}")
            # Default to 0.5 if extraction fails
            return 0.5, f"Error estimating prior probability: {str(e)}"
    
    async def _identify_key_evidence(
        self, question: BinaryQuestion, research: str
    ) -> List[Dict[str, Any]]:
        """
        Identify key pieces of evidence from the research.
        
        Args:
            question: The binary question
            research: The research results
            
        Returns:
            List of evidence items with descriptions and likelihood ratios
        """
        # Create a prompt for identifying evidence
        prompt = (
            f"Identify key pieces of evidence from the research that are relevant to this forecasting question.\n\n"
            f"Question: {question.question_text}\n\n"
            f"Research: {research[:8000]}...\n\n"  # Truncate research to avoid token limits
            f"For each piece of evidence:\n"
            f"1. Provide a clear description of the evidence\n"
            f"2. Assess whether this evidence supports or contradicts the question's resolution as 'Yes'\n"
            f"3. Estimate a likelihood ratio for this evidence (how much more/less likely this evidence would be if the answer is 'Yes' vs 'No')\n"
            f"4. Provide brief reasoning for your likelihood ratio estimate\n\n"
            f"Identify 3-5 distinct, important pieces of evidence. Format your response as a JSON array of objects, "
            f"where each object has the fields 'description', 'supports_yes', 'likelihood_ratio', and 'reasoning'."
        )
        
        # Get the evidence identification
        try:
            evidence_response = await self.get_llm("likelihood_estimator", "llm").invoke(prompt)
            
            # Extract JSON from the response
            json_match = re.search(r'```json\s*(.*?)\s*```', evidence_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find anything that looks like a JSON array
                json_match = re.search(r'(\[.*\])', evidence_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Fallback to the whole response
                    json_str = evidence_response
            
            # Clean up the string to make it valid JSON
            json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)  # Remove comments
            
            # Try to parse, with fallback for errors
            try:
                evidence_list = json.loads(json_str)
                return evidence_list
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse evidence JSON: {json_str}")
                # Create a default evidence item
                return [{"description": "Research findings (details could not be parsed)", 
                         "supports_yes": None, 
                         "likelihood_ratio": 1.0, 
                         "reasoning": "Error parsing evidence details."}]
                
        except Exception as e:
            logger.error(f"Error identifying key evidence: {str(e)}")
            return [{"description": "Research findings", 
                     "supports_yes": None, 
                     "likelihood_ratio": 1.0, 
                     "reasoning": f"Error identifying evidence: {str(e)}"}]
    
    async def _apply_bayesian_updating(
        self, 
        question: BinaryQuestion, 
        prior_probability: float, 
        evidence_list: List[Dict[str, Any]]
    ) -> Tuple[float, str]:
        """
        Apply Bayesian updating to the evidence.
        
        Args:
            question: The binary question
            prior_probability: The prior probability
            evidence_list: List of evidence items
            
        Returns:
            Tuple of (posterior probability, reasoning)
        """
        # Convert prior to odds
        prior_odds = prior_probability / (1 - prior_probability)
        
        # Initialize updating reasoning
        updating_reasoning = (
            f"Starting with a prior probability of {prior_probability:.2%} "
            f"(odds of {prior_odds:.2f}), I will apply Bayesian updating to each piece of evidence.\n\n"
        )
        
        # Track posterior odds through updates
        posterior_odds = prior_odds
        
        # Apply each piece of evidence
        for i, evidence in enumerate(evidence_list, 1):
            # Get likelihood ratio (default to 1.0 if missing or invalid)
            likelihood_ratio = evidence.get("likelihood_ratio", 1.0)
            if not isinstance(likelihood_ratio, (int, float)) or likelihood_ratio <= 0:
                likelihood_ratio = 1.0
                
            # Apply Bayes' rule
            posterior_odds = posterior_odds * likelihood_ratio
            
            # Calculate new probability
            new_probability = posterior_odds / (1 + posterior_odds)
            
            # Document the update
            updating_reasoning += (
                f"Evidence {i}: {evidence.get('description', 'Unnamed evidence')}\n"
                f"Likelihood ratio: {likelihood_ratio:.2f} "
                f"({evidence.get('reasoning', 'No reasoning provided')})\n"
                f"Updated odds: {posterior_odds:.2f} (probability: {new_probability:.2%})\n\n"
            )
        
        # Convert final odds to probability
        final_probability = posterior_odds / (1 + posterior_odds)
        
        # Add final summary
        updating_reasoning += (
            f"Final posterior probability after Bayesian updating: {final_probability:.2%}\n"
            f"This represents a {abs(final_probability - prior_probability) * 100:.1f} percentage point "
            f"{'increase' if final_probability > prior_probability else 'decrease'} from the prior."
        )
        
        return final_probability, updating_reasoning
    
    async def _perform_sensitivity_analysis(
        self,
        question: BinaryQuestion,
        prior_probability: float,
        evidence_list: List[Dict[str, Any]],
        final_probability: float
    ) -> str:
        """
        Perform sensitivity analysis on the Bayesian updating.
        
        Args:
            question: The binary question
            prior_probability: The prior probability
            evidence_list: List of evidence items
            final_probability: The final posterior probability
            
        Returns:
            Sensitivity analysis reasoning
        """
        # Create a prompt for sensitivity analysis
        prompt = (
            f"Perform a sensitivity analysis on the Bayesian updating process for this question.\n\n"
            f"Question: {question.question_text}\n\n"
            f"Prior probability: {prior_probability:.2%}\n"
            f"Final probability: {final_probability:.2%}\n\n"
            f"Evidence used (with likelihood ratios):\n"
        )
        
        # Add evidence details
        for i, evidence in enumerate(evidence_list, 1):
            prompt += (
                f"{i}. {evidence.get('description', 'Unnamed evidence')} - "
                f"LR: {evidence.get('likelihood_ratio', 1.0)}\n"
            )
            
        prompt += (
            f"\nAnalyze how sensitive the final probability is to:\n"
            f"1. Changes in the prior probability (test with both higher and lower priors)\n"
            f"2. Changes in the likelihood ratios for the most influential pieces of evidence\n"
            f"3. The potential impact of evidence that might have been overlooked\n\n"
            f"Provide a quantitative assessment where possible and discuss the robustness of the final probability."
        )
        
        # Get the sensitivity analysis
        try:
            sensitivity_analysis = await self.get_llm("sensitivity_analyzer", "llm").invoke(prompt)
            return sensitivity_analysis
        except Exception as e:
            logger.error(f"Error performing sensitivity analysis: {str(e)}")
            return f"Error performing sensitivity analysis: {str(e)}"
    
    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        """
        Run a forecast on a numeric question using Bayesian methods.
        
        Args:
            question: The numeric question
            research: The research results
            
        Returns:
            A reasoned prediction with a distribution
        """
        # For numeric questions, estimate a prior distribution
        prior_distribution, prior_reasoning = await self._estimate_prior_distribution(question, research)
        
        # Identify key evidence that would update the distribution
        posterior_distribution, update_reasoning = await self._update_numeric_distribution(
            question, prior_distribution, research
        )
        
        # Combine the reasoning
        combined_reasoning = (
            f"## Prior Distribution Estimation\n{prior_reasoning}\n\n"
            f"## Distribution Update Based on Evidence\n{update_reasoning}"
        )
        
        return ReasonedPrediction(
            prediction_value=posterior_distribution, reasoning=combined_reasoning
        )
    
    async def _estimate_prior_distribution(
        self, question: NumericQuestion, research: str
    ) -> Tuple[NumericDistribution, str]:
        """
        Estimate a prior distribution for a numeric question.
        
        Args:
            question: The numeric question
            research: The research results
            
        Returns:
            Tuple of (prior distribution, reasoning)
        """
        # Get lower/upper bound messages
        lower_bound_message = ""
        upper_bound_message = ""
        
        if question.has_lower_bound:
            lower_bound_message = f"Lower bound: {question.lower_bound} {question.unit_of_measure}"
        if question.has_upper_bound:
            upper_bound_message = f"Upper bound: {question.upper_bound} {question.unit_of_measure}"
        
        # Create a prompt for estimating the prior distribution
        prompt = (
            f"Estimate a prior probability distribution for the following numeric forecasting question.\n\n"
            f"Question: {question.question_text}\n\n"
            f"Background: {question.background_info or 'Not provided'}\n\n"
            f"{lower_bound_message}\n{upper_bound_message}\n\n"
            f"Follow these steps:\n"
            f"1. Identify relevant historical or reference data for this type of question\n"
            f"2. Based on this data, estimate a prior distribution as a normal distribution\n"
            f"3. Specify the mean and standard deviation of this distribution\n"
            f"4. Explain your reasoning for these parameters\n\n"
            f"Your response should include clear numerical estimates for the mean and standard deviation "
            f"in {question.unit_of_measure}."
        )
        
        # Get the prior distribution estimation
        try:
            prior_reasoning = await self.get_llm("prior_estimator", "llm").invoke(prompt)
            
            # Extract the distribution parameters
            distribution = PredictionExtractor.extract_normal_distribution(prior_reasoning)
            
            return distribution, prior_reasoning
        except Exception as e:
            logger.error(f"Error estimating prior distribution: {str(e)}")
            
            # Create a default distribution based on bounds if available
            if question.has_lower_bound and question.has_upper_bound:
                mean = (question.lower_bound + question.upper_bound) / 2
                stdev = (question.upper_bound - question.lower_bound) / 4
            elif question.has_lower_bound:
                mean = question.lower_bound * 1.5
                stdev = question.lower_bound / 2
            elif question.has_upper_bound:
                mean = question.upper_bound / 2
                stdev = question.upper_bound / 4
            else:
                mean = 100  # Default placeholder
                stdev = 30  # Default placeholder
                
            default_distribution = NumericDistribution(mean=mean, stdev=stdev)
            return default_distribution, f"Error estimating prior distribution: {str(e)}"
    
    async def _update_numeric_distribution(
        self,
        question: NumericQuestion,
        prior_distribution: NumericDistribution,
        research: str
    ) -> Tuple[NumericDistribution, str]:
        """
        Update a numeric distribution based on evidence.
        
        Args:
            question: The numeric question
            prior_distribution: The prior distribution
            research: The research results
            
        Returns:
            Tuple of (posterior distribution, reasoning)
        """
        # Create a prompt for updating the distribution
        prompt = (
            f"Update the prior distribution for this numeric question based on the research evidence.\n\n"
            f"Question: {question.question_text}\n\n"
            f"Prior distribution:\n"
            f"- Mean: {prior_distribution.mean} {question.unit_of_measure}\n"
            f"- Standard deviation: {prior_distribution.stdev} {question.unit_of_measure}\n\n"
            f"Research (excerpts): {research[:8000]}...\n\n"  # Truncate research to avoid token limits
            f"Follow these steps:\n"
            f"1. Identify key pieces of evidence from the research that would update our distribution\n"
            f"2. For each piece of evidence, assess how it should shift the mean or affect uncertainty\n"
            f"3. Combine these updates to determine a posterior distribution\n"
            f"4. Specify the new mean and standard deviation\n\n"
            f"Your response should include clear numerical estimates for the updated mean and standard deviation "
            f"in {question.unit_of_measure}."
        )
        
        # Get the distribution update
        try:
            update_reasoning = await self.get_llm("likelihood_estimator", "llm").invoke(prompt)
            
            # Extract the updated distribution parameters
            posterior_distribution = PredictionExtractor.extract_normal_distribution(update_reasoning)
            
            return posterior_distribution, update_reasoning
        except Exception as e:
            logger.error(f"Error updating numeric distribution: {str(e)}")
            return prior_distribution, f"Error updating distribution: {str(e)}" 