"""
Personality Recommender Module

This module provides a recommendation system for matching optimal personalities
to specific forecasting questions and domains.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.config import PersonalityConfig


logger = logging.getLogger(__name__)


class PersonalityRecommender:
    """
    A recommendation system for matching optimal personalities to forecasting questions.
    
    This class analyzes questions and domains to recommend the most appropriate
    personality for generating accurate forecasts.
    """
    
    # Domain classifications and recommended personalities
    DOMAIN_RECOMMENDATIONS = {
        "economics": {"primary": "economist", "secondary": "bayesian"},
        "finance": {"primary": "economist", "secondary": "bayesian"},
        "politics": {"primary": "bayesian", "secondary": "creative"},
        "technology": {"primary": "creative", "secondary": "balanced"},
        "science": {"primary": "bayesian", "secondary": "balanced"},
        "health": {"primary": "cautious", "secondary": "balanced"},
        "sports": {"primary": "bayesian", "secondary": "balanced"},
        "entertainment": {"primary": "creative", "secondary": "balanced"},
        "geopolitics": {"primary": "bayesian", "secondary": "cautious"},
        "environment": {"primary": "cautious", "secondary": "balanced"},
        "energy": {"primary": "economist", "secondary": "balanced"},
        "social": {"primary": "creative", "secondary": "balanced"},
        "general": {"primary": "balanced", "secondary": "bayesian"},
    }
    
    # Question type recommendations
    QUESTION_TYPE_RECOMMENDATIONS = {
        "binary": {
            "trend_based": "bayesian",
            "complex_causal": "creative",
            "data_rich": "bayesian",
            "high_uncertainty": "cautious",
            "expert_driven": "balanced",
            "market_based": "economist",
        },
        "numeric": {
            "time_series": "bayesian",
            "economic_indicator": "economist",
            "continuous_variable": "bayesian",
            "discrete_count": "bayesian",
            "high_variance": "cautious",
            "novel_domain": "creative",
        },
        "multiple_choice": {
            "few_options": "bayesian",
            "many_options": "creative",
            "market_driven": "economist",
            "policy_outcome": "bayesian",
            "event_based": "balanced",
        },
    }
    
    # Timeframe recommendations
    TIMEFRAME_RECOMMENDATIONS = {
        "short_term": "bayesian",  # Days to weeks
        "medium_term": "balanced",  # Months to a year
        "long_term": "creative",    # Years or more
    }
    
    def __init__(
        self, 
        llm_model: str = "gpt-4o",
        personality_manager: Optional[PersonalityManager] = None
    ):
        """
        Initialize the PersonalityRecommender.
        
        Args:
            llm_model: Model to use for question analysis
            personality_manager: Optional personality manager instance
        """
        self.llm = GeneralLlm(model=llm_model, temperature=0.2)
        self.personality_manager = personality_manager or PersonalityManager()
        self.available_personalities = self.personality_manager.get_all_personalities()
        
        logger.info(f"Initialized PersonalityRecommender with available personalities: {self.available_personalities}")
    
    async def recommend_for_question(
        self, 
        question: MetaculusQuestion,
        explain: bool = True
    ) -> Dict[str, Any]:
        """
        Recommend the best personality for a specific question.
        
        Args:
            question: The question to analyze
            explain: Whether to include explanation in the response
            
        Returns:
            Dict with recommended personality and optionally explanation
        """
        # Analyze the question
        analysis = await self._analyze_question(question)
        
        # Get recommendations based on analysis
        domain_rec = self._get_domain_recommendation(analysis["domain"])
        question_type_rec = self._get_question_type_recommendation(
            analysis["question_type"], analysis["subtype"]
        )
        timeframe_rec = self._get_timeframe_recommendation(analysis["timeframe"])
        
        # Combine recommendations with weights
        weighted_recs = {
            domain_rec["primary"]: 3,
            domain_rec["secondary"]: 1,
            question_type_rec: 2,
            timeframe_rec: 1,
        }
        
        # Get the most recommended personality
        best_personality = max(weighted_recs.items(), key=lambda x: x[1])[0]
        
        # Ensure the recommended personality is available
        if best_personality not in self.available_personalities:
            best_personality = "balanced"  # Default fallback
        
        result = {"recommended_personality": best_personality}
        
        if explain:
            explanation = (
                f"Recommended personality: {best_personality}\n\n"
                f"Analysis:\n"
                f"- Domain: {analysis['domain']} (suggests {domain_rec['primary']} or {domain_rec['secondary']})\n"
                f"- Question type: {analysis['question_type']}/{analysis['subtype']} (suggests {question_type_rec})\n"
                f"- Timeframe: {analysis['timeframe']} (suggests {timeframe_rec})\n\n"
                f"The {best_personality} personality was selected because it best matches the combination "
                f"of domain knowledge, question characteristics, and timeframe considerations."
            )
            result["explanation"] = explanation
            result["analysis"] = analysis
            
        return result
    
    async def recommend_for_batch(
        self, 
        questions: List[MetaculusQuestion]
    ) -> Dict[str, str]:
        """
        Recommend personalities for a batch of questions.
        
        Args:
            questions: List of questions to analyze
            
        Returns:
            Dict mapping question IDs to recommended personalities
        """
        recommendations = {}
        
        for question in questions:
            try:
                rec = await self.recommend_for_question(question, explain=False)
                recommendations[str(question.id)] = rec["recommended_personality"]
            except Exception as e:
                logger.error(f"Error recommending personality for question {question.id}: {str(e)}")
                recommendations[str(question.id)] = "balanced"  # Default fallback
                
        return recommendations
    
    async def _analyze_question(self, question: MetaculusQuestion) -> Dict[str, str]:
        """
        Analyze a question to determine its domain, type, and timeframe.
        
        Args:
            question: The question to analyze
            
        Returns:
            Dict with domain, question_type, subtype, and timeframe
        """
        # Create analysis prompt
        prompt = (
            f"Analyze this forecasting question and classify it according to domain, type, and timeframe:\n\n"
            f"Question: {question.question_text}\n\n"
            f"Background: {question.background_info or 'Not provided'}\n\n"
            f"Resolution criteria: {question.resolution_criteria or 'Not provided'}\n\n"
            f"1. Domain: Choose the primary domain from these options: economics, finance, politics, technology, "
            f"science, health, sports, entertainment, geopolitics, environment, energy, social, general\n\n"
            f"2. Question Type: The question is {self._get_question_type_name(question)}. "
            f"Select the most appropriate subtype.\n\n"
            f"For binary questions: trend_based, complex_causal, data_rich, high_uncertainty, expert_driven, market_based\n"
            f"For numeric questions: time_series, economic_indicator, continuous_variable, discrete_count, high_variance, novel_domain\n"
            f"For multiple choice questions: few_options, many_options, market_driven, policy_outcome, event_based\n\n"
            f"3. Timeframe: short_term (days to weeks), medium_term (months to a year), long_term (years or more)\n\n"
            f"Format your response as a JSON dictionary with the keys 'domain', 'question_type', 'subtype', and 'timeframe'."
        )
        
        # Get analysis from LLM
        try:
            analysis_response = await self.llm.invoke(prompt)
            
            # Extract domain, question_type, subtype, and timeframe
            domain_match = re.search(r'"domain"\s*:\s*"([^"]+)"', analysis_response)
            question_type_match = re.search(r'"question_type"\s*:\s*"([^"]+)"', analysis_response)
            subtype_match = re.search(r'"subtype"\s*:\s*"([^"]+)"', analysis_response)
            timeframe_match = re.search(r'"timeframe"\s*:\s*"([^"]+)"', analysis_response)
            
            domain = domain_match.group(1) if domain_match else "general"
            question_type = question_type_match.group(1) if question_type_match else self._get_question_type_name(question)
            subtype = subtype_match.group(1) if subtype_match else "general"
            timeframe = timeframe_match.group(1) if timeframe_match else "medium_term"
            
            # Normalize values
            domain = domain.lower()
            question_type = question_type.lower()
            subtype = subtype.lower()
            timeframe = timeframe.lower()
            
            # Ensure domain is valid
            if domain not in self.DOMAIN_RECOMMENDATIONS:
                domain = "general"
                
            # Ensure subtype is valid
            valid_subtypes = list(self.QUESTION_TYPE_RECOMMENDATIONS.get(question_type, {}).keys())
            if not valid_subtypes or subtype not in valid_subtypes:
                subtype = valid_subtypes[0] if valid_subtypes else "general"
                
            # Ensure timeframe is valid
            if timeframe not in self.TIMEFRAME_RECOMMENDATIONS:
                timeframe = "medium_term"
                
            return {
                "domain": domain,
                "question_type": question_type,
                "subtype": subtype,
                "timeframe": timeframe,
            }
                
        except Exception as e:
            logger.error(f"Error analyzing question: {str(e)}")
            return {
                "domain": "general",
                "question_type": self._get_question_type_name(question),
                "subtype": "general",
                "timeframe": "medium_term",
            }
    
    def _get_question_type_name(self, question: MetaculusQuestion) -> str:
        """
        Get the question type name based on the question class.
        
        Args:
            question: The question to analyze
            
        Returns:
            String representation of the question type
        """
        class_name = question.__class__.__name__
        if "Binary" in class_name:
            return "binary"
        elif "Numeric" in class_name:
            return "numeric"
        elif "MultipleChoice" in class_name:
            return "multiple_choice"
        else:
            return "general"
    
    def _get_domain_recommendation(self, domain: str) -> Dict[str, str]:
        """
        Get personality recommendations for a domain.
        
        Args:
            domain: The domain to get recommendations for
            
        Returns:
            Dict with primary and secondary personality recommendations
        """
        return self.DOMAIN_RECOMMENDATIONS.get(domain, self.DOMAIN_RECOMMENDATIONS["general"])
    
    def _get_question_type_recommendation(self, question_type: str, subtype: str) -> str:
        """
        Get personality recommendations for a question type and subtype.
        
        Args:
            question_type: The question type (binary, numeric, multiple_choice)
            subtype: The specific subtype within the question type
            
        Returns:
            Recommended personality
        """
        type_recs = self.QUESTION_TYPE_RECOMMENDATIONS.get(question_type, {})
        return type_recs.get(subtype, "balanced")
    
    def _get_timeframe_recommendation(self, timeframe: str) -> str:
        """
        Get personality recommendations based on question timeframe.
        
        Args:
            timeframe: The timeframe (short_term, medium_term, long_term)
            
        Returns:
            Recommended personality
        """
        return self.TIMEFRAME_RECOMMENDATIONS.get(timeframe, "balanced")
    
    def get_personality_traits(self, personality_name: str) -> Dict[str, Any]:
        """
        Get the traits of a specific personality.
        
        Args:
            personality_name: Name of the personality to get traits for
            
        Returns:
            Dict of personality traits
        """
        # Load the personality if needed
        personality = self.personality_manager.load_personality(personality_name)
        
        # Extract traits
        traits = {}
        for name, trait in personality.traits.items():
            traits[name] = trait.value
            
        return {
            "name": personality.name,
            "description": personality.description,
            "reasoning_depth": personality.reasoning_depth.value,
            "uncertainty_approach": personality.uncertainty_approach.value,
            "thinking_style": personality.thinking_style.value,
            "traits": traits
        }
    
    def get_personality_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all available personalities.
        
        Returns:
            Dict mapping personality names to descriptions
        """
        descriptions = {}
        
        for name in self.available_personalities:
            try:
                personality = self.personality_manager.load_personality(name)
                descriptions[name] = personality.description
            except Exception as e:
                logger.error(f"Error getting description for personality {name}: {str(e)}")
                descriptions[name] = f"Unknown personality: {name}"
                
        return descriptions
    
    def get_recommended_bot_type(self, personality_name: str) -> str:
        """
        Get the recommended bot type for a personality.
        
        Args:
            personality_name: Name of the personality
            
        Returns:
            Recommended bot type
        """
        # Map personalities to optimal bot types
        bot_type_map = {
            "bayesian": "bayesian",
            "economist": "economist",
            "creative": "research",
            "cautious": "calibrated",
            "balanced": "basic",
        }
        
        return bot_type_map.get(personality_name, "template") 