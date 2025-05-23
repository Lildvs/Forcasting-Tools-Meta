"""
Economist Bot

This module provides a specialized forecasting bot that focuses on economic reasoning
and analysis. It leverages economic frameworks, data, and models to generate forecasts.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Set

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
from forecasting_tools.forecast_bots.research_bot import ResearchBot
from forecasting_tools.forecast_helpers.prediction_extractor import PredictionExtractor
from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.processors import EconomistProcessor


logger = logging.getLogger(__name__)


class EconomistBot(ResearchBot):
    """
    A specialized forecasting bot that focuses on economic reasoning and analysis.
    
    This bot extends the ResearchBot with economics-specific features:
    - Economic framework selection based on question type
    - Economic data prioritization in research
    - Market-based reasoning approaches
    - Focus on incentives, institutional factors, and systemic dynamics
    """
    
    # Economic frameworks that can be applied to different questions
    ECONOMIC_FRAMEWORKS = {
        "microeconomic": {
            "description": "Analysis focused on individual decision-making, markets, and resource allocation",
            "key_concepts": ["supply and demand", "price elasticity", "market equilibrium", 
                           "consumer/producer surplus", "game theory", "incentives"],
        },
        "macroeconomic": {
            "description": "Analysis focused on aggregate economic phenomena and policies",
            "key_concepts": ["GDP", "inflation", "unemployment", "monetary policy", 
                           "fiscal policy", "business cycles", "aggregate demand/supply"],
        },
        "behavioral": {
            "description": "Analysis incorporating psychological insights into economic behavior",
            "key_concepts": ["bounded rationality", "heuristics", "biases", "framing effects", 
                           "prospect theory", "mental accounting", "nudging"],
        },
        "institutional": {
            "description": "Analysis focused on the role of institutions and rules in economic outcomes",
            "key_concepts": ["transaction costs", "property rights", "governance", 
                           "institutional quality", "path dependence", "political economy"],
        },
        "financial": {
            "description": "Analysis of financial markets, instruments, and systems",
            "key_concepts": ["efficient market hypothesis", "risk premium", "time value of money", 
                           "portfolio theory", "derivatives", "financial stability"],
        },
    }
    
    # Economic indicators to check for different domains
    ECONOMIC_INDICATORS = {
        "general": ["GDP growth", "inflation rate", "unemployment rate", "interest rates"],
        "monetary": ["interest rates", "money supply", "exchange rates", "central bank statements"],
        "fiscal": ["government spending", "tax revenues", "budget deficit/surplus", "debt-to-GDP ratio"],
        "trade": ["trade balance", "import/export growth", "tariff levels", "trade agreements"],
        "labor": ["unemployment rate", "labor force participation", "wage growth", "job creation"],
        "industry": ["industrial production", "capacity utilization", "purchasing managers index", "inventory levels"],
        "housing": ["housing starts", "home sales", "home price indices", "mortgage rates"],
        "consumer": ["consumer confidence", "retail sales", "personal income", "savings rate"],
    }
    
    def __init__(
        self,
        *,
        bot_name: str | None = None,
        research_reports_per_question: int = 2,
        predictions_per_research_report: int = 1,
        use_research_summary_to_forecast: bool = True,
        publish_reports_to_metaculus: bool = False,
        folder_to_save_reports_to: str | None = None,
        skip_previously_forecasted_questions: bool = False,
        llms: dict[str, str | GeneralLlm] | None = None,
        logger_name: Optional[str] = None,
        personality_name: str = "economist",  # Default to economist personality
        research_depth: str = "deep",
        economic_focus: Optional[str] = None,  # micro, macro, behavioral, institutional, financial
    ) -> None:
        """
        Initialize the EconomistBot.
        
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
            research_depth: Depth of research to perform (shallow, moderate, deep)
            economic_focus: Specific economic framework to focus on
        """
        # Ensure economist personality is used if not explicitly overridden
        if personality_name != "economist":
            logger.info(f"Using specified personality '{personality_name}' instead of default 'economist'")
            
        super().__init__(
            bot_name=bot_name or f"EconomistBot_{personality_name}",
            research_reports_per_question=research_reports_per_question,
            predictions_per_research_report=predictions_per_research_report,
            use_research_summary_to_forecast=use_research_summary_to_forecast,
            publish_reports_to_metaculus=publish_reports_to_metaculus,
            folder_to_save_reports_to=folder_to_save_reports_to,
            skip_previously_forecasted_questions=skip_previously_forecasted_questions,
            llms=llms,
            logger_name=logger_name,
            personality_name=personality_name,
            research_depth=research_depth,
        )
        
        self.economic_focus = economic_focus
        
        # Ensure we're using the EconomistProcessor
        if not isinstance(self.processor, EconomistProcessor):
            self.processor = EconomistProcessor(self.personality)
            
        logger.info(
            f"Initialized {self.bot_name} with economic_focus: {economic_focus or 'general'}"
        )
    
    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm]:
        """
        Get the default LLM configuration optimized for economic analysis.
        
        Returns:
            Dictionary mapping purpose to LLM
        """
        return {
            "default": GeneralLlm(model="gpt-4o", temperature=0.2),
            "summarizer": GeneralLlm(model="gpt-4o", temperature=0.1),
            "research": "perplexity/sonar-deep-research",
            "research_planner": GeneralLlm(model="gpt-4o", temperature=0.3),
            "economic_analyzer": GeneralLlm(model="gpt-4o", temperature=0.2),
        }
    
    async def _generate_research_plan(self, question: MetaculusQuestion) -> List[str]:
        """
        Generate an economics-focused research plan.
        
        Args:
            question: The question to research
            
        Returns:
            List of research queries to execute
        """
        # Base research queries
        base_queries = await super()._generate_research_plan(question)
        
        # Add economic-specific research queries
        economic_queries = []
        
        # Analyze the question to determine relevant economic indicators and data
        relevant_indicators = await self._identify_relevant_economic_indicators(question)
        
        # Add queries for economic data
        for domain, indicators in relevant_indicators.items():
            if indicators:
                # Select up to 3 most relevant indicators per domain
                selected_indicators = indicators[:3]
                indicator_str = ", ".join(selected_indicators)
                economic_queries.append(f"Latest data and trends for {indicator_str} related to {question.question_text}")
                
        # Add query for the specific economic framework if set
        if self.economic_focus and self.economic_focus in self.ECONOMIC_FRAMEWORKS:
            framework = self.ECONOMIC_FRAMEWORKS[self.economic_focus]
            economic_queries.append(
                f"Analyze {question.question_text} using {self.economic_focus} economics framework, "
                f"focusing on {', '.join(framework['key_concepts'][:3])}"
            )
            
        # Add a query for market-based predictions if applicable
        economic_queries.append(f"Market predictions, forecasts, and expert economic analyses related to {question.question_text}")
        
        # Combine base and economic queries, removing duplicates
        all_queries = base_queries + economic_queries
        unique_queries = []
        seen = set()
        for query in all_queries:
            query_key = query.lower()
            if query_key not in seen:
                seen.add(query_key)
                unique_queries.append(query)
                
        # Limit to maximum 5 queries to avoid too much research overhead
        return unique_queries[:5]
    
    async def _identify_relevant_economic_indicators(self, question: MetaculusQuestion) -> Dict[str, List[str]]:
        """
        Identify relevant economic indicators for a question.
        
        Args:
            question: The question to analyze
            
        Returns:
            Dictionary mapping domains to lists of relevant indicators
        """
        # Construct prompt to identify relevant economic domains and indicators
        prompt = (
            f"Analyze the following forecasting question and identify which economic domains and indicators "
            f"would be most relevant for forecasting it. Focus only on the most relevant domains.\n\n"
            f"Question: {question.question_text}\n\n"
            f"Background: {question.background_info or 'Not provided'}\n\n"
            f"List the top 2-3 most relevant economic domains from the following options: "
            f"general, monetary, fiscal, trade, labor, industry, housing, consumer.\n\n"
            f"For each relevant domain, list the most important indicators to research. "
            f"Format your response as a JSON dictionary with domains as keys and lists of indicators as values."
        )
        
        # Get economic analysis
        try:
            analysis_response = await self.get_llm("economic_analyzer", "llm").invoke(prompt)
            
            # Extract JSON dictionary from response
            import json
            import re
            
            # Look for JSON-like content
            json_match = re.search(r'```json\s*(.*?)\s*```', analysis_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find anything that looks like a dictionary
                json_match = re.search(r'(\{.*\})', analysis_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Fallback to the whole response
                    json_str = analysis_response
            
            # Clean up the string to make it valid JSON
            json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)  # Remove comments
            json_str = re.sub(r'[\n\r\t]', '', json_str)  # Remove newlines, tabs
            
            # Try to parse, with fallback for errors
            try:
                result = json.loads(json_str)
                # Filter to ensure only valid domains
                valid_domains = set(self.ECONOMIC_INDICATORS.keys())
                result = {k: v for k, v in result.items() if k in valid_domains}
                return result
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse economic indicator JSON: {json_str}")
                return {"general": self.ECONOMIC_INDICATORS["general"]}
                
        except Exception as e:
            logger.error(f"Error identifying economic indicators: {str(e)}")
            return {"general": self.ECONOMIC_INDICATORS["general"]}
    
    async def _synthesize_research(
        self, question: MetaculusQuestion, research_results: List[str]
    ) -> str:
        """
        Synthesize research results with an economic perspective.
        
        Args:
            question: The original question
            research_results: List of research results to synthesize
            
        Returns:
            Synthesized research as a single string
        """
        # Get basic synthesis
        synthesis = await super()._synthesize_research(question, research_results)
        
        # Add economic framework analysis
        if self.economic_focus:
            framework_analysis = await self._apply_economic_framework(question, synthesis)
            synthesis = f"{synthesis}\n\n## Economic Framework Analysis\n{framework_analysis}"
        
        return synthesis
    
    async def _apply_economic_framework(
        self, question: MetaculusQuestion, research: str
    ) -> str:
        """
        Apply a specific economic framework to analyze the research.
        
        Args:
            question: The question being analyzed
            research: The research to analyze
            
        Returns:
            Analysis using the specified economic framework
        """
        # Skip if no specific framework is set
        if not self.economic_focus or self.economic_focus not in self.ECONOMIC_FRAMEWORKS:
            return ""
            
        # Get framework details
        framework = self.ECONOMIC_FRAMEWORKS[self.economic_focus]
        
        # Create prompt for framework analysis
        prompt = (
            f"You are an economist specializing in {self.economic_focus} economics. "
            f"Apply {self.economic_focus} economic analysis to the following research "
            f"on this question: '{question.question_text}'\n\n"
            f"Focus specifically on the following key concepts from {self.economic_focus} economics: "
            f"{', '.join(framework['key_concepts'])}\n\n"
            f"Research to analyze:\n{research[:10000]}\n\n"  # Limit research to avoid token limits
            f"Provide a concise economic analysis using the {self.economic_focus} framework. "
            f"Focus on the most relevant insights for forecasting the question."
        )
        
        # Get the analysis
        try:
            framework_analysis = await self.get_llm("economic_analyzer", "llm").invoke(prompt)
            return framework_analysis
        except Exception as e:
            logger.error(f"Error applying economic framework: {str(e)}")
            return f"Error applying {self.economic_focus} economic framework."
            
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        """
        Run a forecast on a binary question using economic reasoning.
        
        Args:
            question: The binary question
            research: The research results
            
        Returns:
            A reasoned prediction with economic analysis
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
        
        # Add economic reasoning guidance
        prompt += "\n\nIMPORTANT: Apply economic reasoning to this forecast. Consider incentives, market mechanisms, and relevant economic indicators. Analyze how economic actors are likely to behave given the current context."
        
        # Process the prompt with the economist processor
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
            reasoning, max_prediction=0.97, min_prediction=0.03
        )
        
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )
    
    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        """
        Run a forecast on a numeric question using economic reasoning.
        
        Args:
            question: The numeric question
            research: The research results
            
        Returns:
            A reasoned prediction with economic analysis
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
        
        # Add economic reasoning guidance
        prompt += "\n\nIMPORTANT: Apply economic reasoning to this forecast. For economic variables, consider historical patterns, relevant economic theories, and the current economic context. For distribution estimation, consider both central tendencies and potential variance based on economic uncertainty."
        
        # Process the prompt with the economist processor
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