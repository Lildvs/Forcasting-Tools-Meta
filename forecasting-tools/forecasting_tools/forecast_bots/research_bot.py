"""
Research Bot

This module provides a specialized forecasting bot that adapts its research strategies
based on personality traits. It focuses on in-depth research methods and
utilizes different search techniques based on the personality configuration.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

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
from forecasting_tools.forecast_helpers.perplexity_searcher import Perplexity
from forecasting_tools.forecast_helpers.prediction_extractor import PredictionExtractor
from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.processors import ProcessorFactory


logger = logging.getLogger(__name__)


class ResearchBot(BasicBot):
    """
    A specialized forecasting bot that adapts its research strategies based on personality traits.
    
    This bot extends the BasicBot with enhanced research capabilities, including:
    - Adapting research depth based on personality traits
    - Using different search techniques based on personality
    - Generating multiple research queries based on personality traits
    - Synthesizing research results in a personality-specific way
    """
    
    def __init__(
        self,
        *,
        bot_name: str | None = None,
        research_reports_per_question: int = 2,  # More research by default
        predictions_per_research_report: int = 1,
        use_research_summary_to_forecast: bool = True,  # Enable research summary
        publish_reports_to_metaculus: bool = False,
        folder_to_save_reports_to: str | None = None,
        skip_previously_forecasted_questions: bool = False,
        llms: dict[str, str | GeneralLlm] | None = None,
        logger_name: Optional[str] = None,
        personality_name: str = "balanced",
        research_depth: str = "deep",  # shallow, moderate, deep
    ) -> None:
        """
        Initialize the ResearchBot.
        
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
        """
        super().__init__(
            bot_name=bot_name or f"ResearchBot_{personality_name}",
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
        
        self.research_depth = research_depth
        logger.info(f"Initialized {self.bot_name} with personality: {personality_name}, research_depth: {research_depth}")
    
    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm]:
        """
        Get the default LLM configuration with enhanced research capabilities.
        
        Returns:
            Dictionary mapping purpose to LLM
        """
        return {
            "default": "gpt-4o",
            "summarizer": "gpt-4o",
            "research": "perplexity/sonar-deep-research",
            "research_planner": "gpt-4o",
        }
    
    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Run comprehensive research on a question using personality-specific strategies.
        
        This method:
        1. Generates research queries based on personality traits
        2. Performs deep research with specialized tools
        3. Synthesizes results based on personality traits
        
        Args:
            question: The question to research
            
        Returns:
            The synthesized research results
        """
        # Generate research plan based on personality
        research_plan = await self._generate_research_plan(question)
        
        # Execute the research plan
        research_results = await self._execute_research_plan(question, research_plan)
        
        # Synthesize results based on personality
        synthesized_research = await self._synthesize_research(question, research_results)
        
        return synthesized_research
    
    async def _generate_research_plan(self, question: MetaculusQuestion) -> List[str]:
        """
        Generate a personality-based research plan with multiple queries.
        
        Args:
            question: The question to research
            
        Returns:
            List of research queries to execute
        """
        # Get the personality-specific system prompt for research planning
        planning_prompt = self.personality_manager.get_prompt(
            "research_planning_prompt",
            question_text=question.question_text,
            background_info=question.background_info or "",
            resolution_criteria=question.resolution_criteria or "",
            current_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Adapt research depth based on personality traits
        if self.personality.traits.get("thoroughness"):
            thoroughness = self.personality.traits["thoroughness"].value
            if isinstance(thoroughness, (int, float)):
                if thoroughness > 0.7:  # Very thorough
                    planning_prompt += "\n\nCreate a VERY THOROUGH research plan with 4-5 focused research queries."
                elif thoroughness < 0.3:  # Less thorough
                    planning_prompt += "\n\nCreate a FOCUSED research plan with 2-3 essential research queries."
                else:  # Moderate thoroughness
                    planning_prompt += "\n\nCreate a BALANCED research plan with 3-4 research queries."
        
        # Apply thinking configuration based on personality
        thinking_config = self.personality_manager.get_thinking_config()
        
        # Generate the research plan
        plan_response = await self.get_llm("research_planner", "llm").invoke(
            planning_prompt,
            **thinking_config
        )
        
        # Extract research queries from the plan
        research_queries = []
        for line in plan_response.split("\n"):
            line = line.strip()
            if line.startswith("- ") or line.startswith("* ") or (line.startswith(("Query", "Research")) and ":" in line):
                query_text = line.split(":", 1)[1].strip() if ":" in line else line[2:].strip()
                if query_text and len(query_text) > 10:  # Ensure it's a meaningful query
                    research_queries.append(query_text)
        
        # If no queries were extracted, use the question text as a fallback
        if not research_queries:
            research_queries = [question.question_text]
            
        # Apply personality-specific processing to each query
        processed_queries = []
        for query in research_queries:
            processed_query = self.processor.process_research_query(
                MetaculusQuestion(question_text=query)
            )
            processed_queries.append(processed_query)
            
        return processed_queries
    
    async def _execute_research_plan(
        self, question: MetaculusQuestion, research_queries: List[str]
    ) -> List[str]:
        """
        Execute a research plan by running searches for each query.
        
        Args:
            question: The original question
            research_queries: List of queries to research
            
        Returns:
            List of research results for each query
        """
        # Determine research method based on personality and configured depth
        research_method = "perplexity"
        if self.personality.traits.get("data_reliance"):
            data_reliance = self.personality.traits["data_reliance"].value
            if isinstance(data_reliance, (int, float)):
                if data_reliance > 0.7:  # Heavy data reliance
                    research_method = "deep_research"
                elif data_reliance < 0.3:  # Light data reliance
                    research_method = "standard"
                else:
                    research_method = "perplexity"
        
        # Configure research depth based on settings
        depth_setting = {
            "shallow": {"search_context_size": "low", "reasoning_effort": "low"},
            "moderate": {"search_context_size": "medium", "reasoning_effort": "medium"},
            "deep": {"search_context_size": "high", "reasoning_effort": "high"},
        }.get(self.research_depth, {"search_context_size": "medium", "reasoning_effort": "medium"})
        
        # Execute each research query
        results = []
        for query in research_queries:
            # Get the personality-specific system prompt
            system_prompt = self.personality_manager.get_prompt(
                "research_prompt",
                question_text=query,
                current_date=datetime.now().strftime("%Y-%m-%d")
            )
            
            # Run the appropriate research method
            if research_method == "perplexity":
                # Use Perplexity with depth settings
                perplexity = Perplexity(
                    temperature=0.2,
                    system_prompt=system_prompt,
                    web_search_options={"search_context_size": depth_setting["search_context_size"]},
                    reasoning_effort=depth_setting["reasoning_effort"]
                )
                result = await perplexity.invoke(query)
                
            elif research_method == "deep_research":
                # Use a specialized deep research model/service
                if isinstance(self.get_llm("research"), GeneralLlm):
                    result = await self.get_llm("research", "llm").invoke(
                        f"System: {system_prompt}\n\nUser: {query}\n\nAssistant:"
                    )
                else:
                    # Fallback to default LLM
                    result = await self.get_llm("default", "llm").invoke(
                        f"System: {system_prompt}\n\nUser: {query}\n\nAssistant:"
                    )
                    
            else:  # standard
                # Use the default LLM
                result = await self.get_llm("default", "llm").invoke(
                    f"System: {system_prompt}\n\nUser: {query}\n\nAssistant:"
                )
                
            results.append(result)
            
        return results
    
    async def _synthesize_research(
        self, question: MetaculusQuestion, research_results: List[str]
    ) -> str:
        """
        Synthesize multiple research results based on personality traits.
        
        Args:
            question: The original question
            research_results: List of research results to synthesize
            
        Returns:
            Synthesized research as a single string
        """
        # Get the personality-specific system prompt for synthesis
        synthesis_prompt = self.personality_manager.get_prompt(
            "research_synthesis_prompt",
            question_text=question.question_text,
            background_info=question.background_info or "",
            resolution_criteria=question.resolution_criteria or "",
            current_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Add all research results to the prompt
        synthesis_prompt += "\n\n## Research Results\n\n"
        for i, result in enumerate(research_results, 1):
            synthesis_prompt += f"### Research Result {i}\n{result}\n\n"
            
        synthesis_prompt += "\n\n## Instructions\n"
        synthesis_prompt += "Synthesize these research findings into a comprehensive report that addresses the original question."
        
        # Customize synthesis approach based on personality traits
        if self.personality.thinking_style.value == "creative":
            synthesis_prompt += " Focus on identifying unusual patterns and creative connections between different pieces of research."
        elif self.personality.thinking_style.value == "analytical":
            synthesis_prompt += " Focus on methodically analyzing and comparing the findings, identifying areas of consensus and disagreement."
        elif self.personality.thinking_style.value == "bayesian":
            synthesis_prompt += " Focus on establishing base rates and updating based on new evidence found in the research."
            
        # Apply thinking configuration based on personality
        thinking_config = self.personality_manager.get_thinking_config()
        
        # Generate the synthesis
        synthesis = await self.get_llm("summarizer", "llm").invoke(
            synthesis_prompt,
            **thinking_config
        )
        
        return synthesis
    
    async def summarize_research(
        self, question: MetaculusQuestion, research: str
    ) -> str:
        """
        Override the base class method to provide personality-specific research summaries.
        
        Args:
            question: The question being forecasted
            research: The research to summarize
            
        Returns:
            A summary of the research
        """
        # Get the personality-specific system prompt for summarization
        summary_prompt = self.personality_manager.get_prompt(
            "research_summary_prompt",
            question_text=question.question_text,
            current_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        summary_prompt += f"\n\n## Research to Summarize\n{research}\n\n"
        summary_prompt += "\n\n## Instructions\n"
        summary_prompt += "Create a concise summary of this research that highlights the most important information for forecasting."
        
        # Customize summary approach based on personality traits
        if self.personality.traits.get("data_reliance"):
            data_reliance = self.personality.traits["data_reliance"].value
            if isinstance(data_reliance, (int, float)):
                if data_reliance > 0.7:  # Heavy data reliance
                    summary_prompt += " Focus particularly on quantitative data, statistics, and hard evidence."
                elif data_reliance < 0.3:  # Light data reliance
                    summary_prompt += " Focus on high-level insights, trends, and qualitative factors."
        
        # Apply thinking configuration based on personality
        thinking_config = self.personality_manager.get_thinking_config()
        
        # Generate the summary
        summary = await self.get_llm("summarizer", "llm").invoke(
            summary_prompt,
            **thinking_config
        )
        
        return summary 