"""
Personality Processors Module

This module provides specialized processors for adapting templates and
forecasting processes based on personality traits.
"""

import re
from typing import Dict, List, Optional, Any, Union, Callable

from forecasting_tools.personality_management.config import PersonalityConfig
from forecasting_tools.data_models.questions import MetaculusQuestion, BinaryQuestion, MultipleChoiceQuestion, NumericQuestion


class PersonalityProcessor:
    """
    Base class for personality-specific processors that modify prompts or
    forecast outputs based on personality traits.
    """
    
    def __init__(self, personality: PersonalityConfig):
        """
        Initialize the processor with a personality configuration.
        
        Args:
            personality: The personality configuration to use
        """
        self.personality = personality
    
    def process_prompt(self, prompt: str, **kwargs) -> str:
        """
        Process a prompt based on personality traits.
        
        Args:
            prompt: The original prompt
            **kwargs: Additional context parameters
            
        Returns:
            The processed prompt
        """
        # Base implementation just returns the original prompt
        return prompt
    
    def process_research_query(self, question: MetaculusQuestion) -> str:
        """
        Process a research query based on personality traits.
        
        Args:
            question: The question to research
            
        Returns:
            The processed research query
        """
        # Base implementation just returns the original question text
        query = question.question_text
        
        # Adjust query based on personality traits
        if hasattr(self.personality, "research_focus") and self.personality.research_focus:
            query += f" Focus on {self.personality.research_focus}."
            
        return query
    
    def process_forecast_output(self, output: str, question_type: str) -> str:
        """
        Process a forecast output based on personality traits.
        
        Args:
            output: The original forecast output
            question_type: The type of question (binary, numeric, multiple_choice)
            
        Returns:
            The processed forecast output
        """
        # Base implementation just returns the original output
        return output


class BayesianProcessor(PersonalityProcessor):
    """Processor for Bayesian personality."""
    
    def process_prompt(self, prompt: str, **kwargs) -> str:
        """
        Enhance prompts for Bayesian reasoning.
        
        Args:
            prompt: The original prompt
            **kwargs: Additional context parameters
            
        Returns:
            The processed prompt
        """
        # Add Bayesian-specific instructions
        prompt = prompt.replace(
            "INSTRUCTIONS:",
            "BAYESIAN ANALYSIS INSTRUCTIONS:\n"
            "1. Start by stating your prior probability based on relevant reference classes\n"
            "2. Identify key pieces of evidence and assess likelihood ratios\n"
            "3. Apply Bayes' rule to update your prior with each piece of evidence\n"
            "4. Calculate your posterior probability\n"
            "5. Analyze the sensitivity of your result to different priors\n\n"
            "INSTRUCTIONS:"
        )
        
        return prompt
    
    def process_research_query(self, question: MetaculusQuestion) -> str:
        """
        Adapt research queries to focus on Bayesian evidence.
        
        Args:
            question: The question to research
            
        Returns:
            The processed research query
        """
        query = super().process_research_query(question)
        query += " Include base rates, historical precedents, and statistical data that could inform a Bayesian analysis."
        return query


class EconomistProcessor(PersonalityProcessor):
    """Processor for Economist personality."""
    
    def process_prompt(self, prompt: str, **kwargs) -> str:
        """
        Enhance prompts for economic reasoning.
        
        Args:
            prompt: The original prompt
            **kwargs: Additional context parameters
            
        Returns:
            The processed prompt
        """
        # Add economist-specific instructions
        prompt = prompt.replace(
            "INSTRUCTIONS:",
            "ECONOMIC ANALYSIS INSTRUCTIONS:\n"
            "1. Identify key economic incentives and constraints\n"
            "2. Consider relevant market mechanisms and price signals\n"
            "3. Analyze institutional factors and policy influences\n"
            "4. Assess behavioral economic considerations\n"
            "5. Apply relevant economic models or frameworks\n\n"
            "INSTRUCTIONS:"
        )
        
        return prompt
    
    def process_research_query(self, question: MetaculusQuestion) -> str:
        """
        Adapt research queries to focus on economic factors.
        
        Args:
            question: The question to research
            
        Returns:
            The processed research query
        """
        query = super().process_research_query(question)
        query += " Include economic data, market trends, policy impacts, and institutional factors relevant to this question."
        return query


class CreativeProcessor(PersonalityProcessor):
    """Processor for Creative personality."""
    
    def process_prompt(self, prompt: str, **kwargs) -> str:
        """
        Enhance prompts for creative thinking.
        
        Args:
            prompt: The original prompt
            **kwargs: Additional context parameters
            
        Returns:
            The processed prompt
        """
        # Add creative-specific instructions
        prompt = prompt.replace(
            "INSTRUCTIONS:",
            "CREATIVE THINKING INSTRUCTIONS:\n"
            "1. Consider unconventional angles and perspectives\n"
            "2. Identify unusual or overlooked factors that might influence the outcome\n"
            "3. Explore novel scenarios, especially those with low probability but high impact\n"
            "4. Make connections between seemingly unrelated domains\n"
            "5. Challenge conventional wisdom and status quo assumptions\n\n"
            "INSTRUCTIONS:"
        )
        
        return prompt
    
    def process_research_query(self, question: MetaculusQuestion) -> str:
        """
        Adapt research queries to encourage creative exploration.
        
        Args:
            question: The question to research
            
        Returns:
            The processed research query
        """
        query = super().process_research_query(question)
        query += " Include unconventional perspectives, emerging trends, and creative angles that might be overlooked."
        return query


class CautiousProcessor(PersonalityProcessor):
    """Processor for Cautious personality."""
    
    def process_prompt(self, prompt: str, **kwargs) -> str:
        """
        Enhance prompts for cautious analysis.
        
        Args:
            prompt: The original prompt
            **kwargs: Additional context parameters
            
        Returns:
            The processed prompt
        """
        # Add cautious-specific instructions
        prompt = prompt.replace(
            "INSTRUCTIONS:",
            "CAUTIOUS ANALYSIS INSTRUCTIONS:\n"
            "1. Identify all major sources of uncertainty\n"
            "2. Consider worst-case scenarios and potential biases\n"
            "3. Be careful not to overstate confidence in any conclusion\n"
            "4. Examine evidence quality and potential methodological flaws\n"
            "5. Use wider confidence intervals to account for unknown unknowns\n\n"
            "INSTRUCTIONS:"
        )
        
        return prompt
    
    def process_research_query(self, question: MetaculusQuestion) -> str:
        """
        Adapt research queries to emphasize potential risks and uncertainties.
        
        Args:
            question: The question to research
            
        Returns:
            The processed research query
        """
        query = super().process_research_query(question)
        query += " Include potential risks, limitations in current knowledge, and critical examinations of evidence quality."
        return query


class ProcessorFactory:
    """Factory for creating personality-specific processors."""
    
    @staticmethod
    def create_processor(personality: PersonalityConfig) -> PersonalityProcessor:
        """
        Create a processor based on personality traits.
        
        Args:
            personality: The personality configuration
            
        Returns:
            An appropriate processor for the personality
        """
        # Select processor based on thinking style and personality name
        if personality.thinking_style.value == "bayesian" or personality.name == "bayesian":
            return BayesianProcessor(personality)
        elif personality.name == "economist" or (personality.expert_persona and "economist" in personality.expert_persona.lower()):
            return EconomistProcessor(personality)
        elif personality.thinking_style.value == "creative" or personality.name == "creative":
            return CreativeProcessor(personality)
        elif personality.uncertainty_approach.value == "cautious" or personality.name == "cautious":
            return CautiousProcessor(personality)
        else:
            # Default processor
            return PersonalityProcessor(personality) 