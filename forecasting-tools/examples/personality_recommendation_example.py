#!/usr/bin/env python3
"""
Personality Recommendation Example

This script demonstrates how to use the personality recommendation system
to automatically select the optimal personality and bot type for different
forecasting questions.
"""

import asyncio
import logging
from typing import Dict, List, Optional

from forecasting_tools.data_models.questions import BinaryQuestion, NumericQuestion, MultipleChoiceQuestion
from forecasting_tools.personality_management.recommender import PersonalityRecommender


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Sample questions from different domains
SAMPLE_QUESTIONS = [
    # Economic domain
    BinaryQuestion(
        question_text="Will the Federal Reserve cut interest rates by more than 0.5% in total during 2024?",
        background_info="The Federal Reserve has been maintaining high interest rates to combat inflation, but there are signs that inflation is decreasing.",
        resolution_criteria="This question resolves positively if the Federal Reserve reduces the federal funds rate by more than 0.5 percentage points in total during calendar year 2024.",
    ),
    
    # Technology domain
    BinaryQuestion(
        question_text="Will Apple release a dedicated AR/VR headset before the end of 2024?",
        background_info="Apple has been rumored to be developing augmented reality (AR) and virtual reality (VR) products for several years.",
        resolution_criteria="This question resolves positively if Apple publicly announces and makes available for purchase a standalone AR or VR headset before December 31, 2024.",
    ),
    
    # Health domain
    NumericQuestion(
        question_text="What percentage of the US population will be fully vaccinated against COVID-19 by December 31, 2024?",
        background_info="Vaccination rates in the US have slowed down after initial rapid uptake. Boosters and updated vaccines continue to be developed.",
        resolution_criteria="This question resolves to the percentage of the US population that has received all recommended doses of a COVID-19 vaccine as reported by the CDC on December 31, 2024.",
        lower_bound=0,
        upper_bound=100,
        unit_of_measure="%",
        has_lower_bound=True,
        has_upper_bound=True,
    ),
    
    # Geopolitical domain
    MultipleChoiceQuestion(
        question_text="Which country will be the next to join NATO?",
        background_info="Several countries have expressed interest in joining NATO, while the alliance has expanded in recent years to include new members.",
        resolution_criteria="This question resolves to the next country that formally joins NATO after August 1, 2023. If no country joins by December 31, 2025, the question resolves as 'None of the above'.",
        options=[
            {"id": 1, "option": "Ukraine"},
            {"id": 2, "option": "Sweden"},
            {"id": 3, "option": "Finland"},
            {"id": 4, "option": "Georgia"},
            {"id": 5, "option": "Moldova"},
            {"id": 6, "option": "None of the above"},
        ],
        option_names=["Ukraine", "Sweden", "Finland", "Georgia", "Moldova", "None of the above"],
    ),
    
    # Environmental domain
    NumericQuestion(
        question_text="What will be the Arctic sea ice minimum extent in September 2024, in millions of square kilometers?",
        background_info="Arctic sea ice reaches its minimum extent each September. The minimum has been declining over recent decades due to climate change.",
        resolution_criteria="This question resolves to the minimum Arctic sea ice extent in September 2024 as reported by the National Snow and Ice Data Center (NSIDC), measured in millions of square kilometers.",
        lower_bound=0,
        upper_bound=10,
        unit_of_measure="million sq km",
        has_lower_bound=True,
        has_upper_bound=True,
    ),
    
    # Social domain with high uncertainty
    BinaryQuestion(
        question_text="Will a significant social media platform (with over 100 million users) shut down completely before the end of 2025?",
        background_info="Social media platforms can face various challenges including competition, regulatory issues, financial problems, or changing user preferences.",
        resolution_criteria="This question resolves positively if a social media platform that had over 100 million monthly active users at some point in 2023 completely ceases operations by December 31, 2025.",
    ),
]


async def demonstrate_personality_recommendation():
    """Demonstrate the personality recommendation system for various questions."""
    logger.info("===== Personality Recommendation System Demonstration =====\n")
    
    # Initialize the recommender
    recommender = PersonalityRecommender()
    
    # Process each question
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        logger.info(f"QUESTION {i}: {question.question_text}")
        logger.info(f"Type: {question.__class__.__name__}")
        
        # Get recommendation
        try:
            result = await recommender.recommend_for_question(question)
            
            # Display recommendation
            logger.info(f"RECOMMENDATION:")
            logger.info(f"  Personality: {result['recommended_personality']}")
            logger.info(f"  Bot type: {recommender.get_recommended_bot_type(result['recommended_personality'])}")
            logger.info(f"  Domain: {result['analysis']['domain']}")
            logger.info(f"  Question subtype: {result['analysis']['subtype']}")
            logger.info(f"  Timeframe: {result['analysis']['timeframe']}")
            
            # Get personality traits
            traits = recommender.get_personality_traits(result['recommended_personality'])
            logger.info(f"PERSONALITY TRAITS:")
            logger.info(f"  Description: {traits['description']}")
            logger.info(f"  Reasoning depth: {traits['reasoning_depth']}")
            logger.info(f"  Thinking style: {traits['thinking_style']}")
            logger.info(f"  Uncertainty approach: {traits['uncertainty_approach']}")
            
            # Show how to use this in code
            logger.info(f"CODE USAGE:")
            logger.info(f"  from forecasting_tools.run_bot import create_bot")
            logger.info(f"  bot = create_bot(")
            logger.info(f"      personality_name='{result['recommended_personality']}',")
            logger.info(f"      bot_type='{recommender.get_recommended_bot_type(result['recommended_personality'])}'")
            logger.info(f"  )")
            
        except Exception as e:
            logger.error(f"Error getting recommendation: {str(e)}")
            
        logger.info("\n" + "="*60 + "\n")


async def main():
    """Run the demonstration."""
    await demonstrate_personality_recommendation()


if __name__ == "__main__":
    asyncio.run(main()) 