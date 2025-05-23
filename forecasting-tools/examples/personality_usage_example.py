#!/usr/bin/env python3
"""
Personality Usage Example

This script demonstrates how to use the personality management system with different
forecasting bots. It shows how to select personalities, customize bot behavior,
and get recommendations for optimal personality-bot combinations.
"""

import asyncio
import logging
from typing import Dict, List, Optional

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.questions import BinaryQuestion, NumericQuestion
from forecasting_tools.forecast_bots.basic_bot import BasicBot
from forecasting_tools.forecast_bots.bayesian_bot import BayesianBot
from forecasting_tools.forecast_bots.calibrated_bot import CalibratedBot
from forecasting_tools.forecast_bots.economist_bot import EconomistBot
from forecasting_tools.forecast_bots.research_bot import ResearchBot
from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.recommender import PersonalityRecommender


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Sample questions for demonstration
SAMPLE_QUESTIONS = {
    "economic": BinaryQuestion(
        question_text="Will the Federal Reserve cut interest rates by more than 1% in total during 2023?",
        background_info="The Federal Reserve has been raising interest rates to combat inflation, but there are concerns about economic growth slowing down.",
        resolution_criteria="This question resolves positively if the Federal Reserve reduces the federal funds rate by more than 1 percentage point in total during calendar year 2023.",
    ),
    "technology": BinaryQuestion(
        question_text="Will AI systems capable of writing novels that are indistinguishable from human-written novels be widely available by the end of 2025?",
        background_info="Large language models like GPT-4 have shown significant capabilities in generating coherent text, but still have limitations in creating long-form narratives with consistent plots and characters.",
        resolution_criteria="This question resolves positively if, by December 31, 2025, there are commercially available AI systems that can generate novel-length fiction (50,000+ words) that literary experts cannot reliably distinguish from human-written novels in blind tests.",
    ),
    "geopolitical": BinaryQuestion(
        question_text="Will any country formally withdraw from the Paris Climate Agreement before 2024?",
        background_info="The Paris Agreement is a legally binding international treaty on climate change, adopted by 196 Parties at COP 21 in Paris. The United States previously withdrew under the Trump administration but rejoined under President Biden.",
        resolution_criteria="This question resolves positively if any country that is currently a party to the Paris Agreement formally notifies the UN of its withdrawal from the agreement before January 1, 2024.",
    ),
    "numeric_example": NumericQuestion(
        question_text="What will be the global average temperature anomaly for 2023 according to NASA GISTEMP?",
        background_info="NASA's Goddard Institute for Space Studies (GISS) maintains the GISTEMP dataset of global surface temperature anomalies relative to the 1951-1980 average.",
        resolution_criteria="This question resolves to the global annual mean temperature anomaly for 2023 as reported by NASA GISTEMP, rounded to the nearest 0.01°C.",
        lower_bound=0.5,
        upper_bound=2.0,
        unit_of_measure="°C",
        has_lower_bound=True,
        has_upper_bound=True,
    ),
}


async def demonstrate_personality_recommendation():
    """Demonstrate the personality recommendation system."""
    logger.info("Demonstrating personality recommendation system...")
    
    # Initialize the recommender
    recommender = PersonalityRecommender()
    
    # Get recommendations for each question
    for name, question in SAMPLE_QUESTIONS.items():
        try:
            recommendation = await recommender.recommend_for_question(question)
            logger.info(f"\nRecommendation for {name} question:")
            logger.info(f"  Recommended personality: {recommendation['recommended_personality']}")
            logger.info(f"  Recommended bot type: {recommender.get_recommended_bot_type(recommendation['recommended_personality'])}")
            logger.info(f"  Domain: {recommendation['analysis']['domain']}")
            logger.info(f"  Question type: {recommendation['analysis']['question_type']}/{recommendation['analysis']['subtype']}")
            logger.info(f"  Timeframe: {recommendation['analysis']['timeframe']}")
        except Exception as e:
            logger.error(f"Error getting recommendation for {name}: {str(e)}")


async def demonstrate_bot_with_personality(question_name: str = "economic"):
    """
    Demonstrate forecasting with different bot and personality combinations.
    
    Args:
        question_name: Name of the sample question to use
    """
    question = SAMPLE_QUESTIONS.get(question_name)
    if not question:
        logger.error(f"Question '{question_name}' not found")
        return
        
    logger.info(f"\nDemonstrating forecasting with different bots for question: {question.question_text}")
    
    # Define bot configurations to test
    bot_configs = [
        {"bot_class": BasicBot, "personality": "balanced", "description": "Baseline bot with balanced personality"},
        {"bot_class": BasicBot, "personality": "creative", "description": "Basic bot with creative personality"},
        {"bot_class": ResearchBot, "personality": "balanced", "description": "Research-focused bot with balanced personality"},
        {"bot_class": CalibratedBot, "personality": "cautious", "description": "Calibrated bot with cautious personality"},
        {"bot_class": EconomistBot, "personality": "economist", "description": "Economist bot with economic focus"},
        {"bot_class": BayesianBot, "personality": "bayesian", "description": "Bayesian bot with structured updating"},
    ]
    
    # Run forecasts with each configuration
    for config in bot_configs:
        bot_class = config["bot_class"]
        personality = config["personality"]
        description = config["description"]
        
        logger.info(f"\nTesting: {description}")
        
        # Create the bot with specified personality
        bot = bot_class(
            bot_name=f"{bot_class.__name__}_{personality}",
            personality_name=personality,
            llms={"default": GeneralLlm(model="gpt-4o", temperature=0.2)},
        )
        
        try:
            # Run research (simplified for example)
            logger.info("  Running research...")
            research = await bot.run_research(question)
            research_summary = research[:100] + "..." if len(research) > 100 else research
            logger.info(f"  Research summary: {research_summary}")
            
            # Generate forecast
            logger.info("  Generating forecast...")
            prediction = await bot.get_forecast_from_research(question, research)
            
            # Display results
            if hasattr(prediction.prediction_value, "mean"):
                # Numeric prediction
                mean = prediction.prediction_value.mean
                stdev = prediction.prediction_value.stdev
                logger.info(f"  Prediction: {mean} ± {stdev} {getattr(question, 'unit_of_measure', '')}")
            else:
                # Binary prediction
                value = prediction.prediction_value
                logger.info(f"  Prediction: {value:.1%}")
                
            # Show reasoning excerpt
            reasoning_excerpt = prediction.reasoning.split("\n")[0][:100] + "..."
            logger.info(f"  Reasoning: {reasoning_excerpt}")
            
        except Exception as e:
            logger.error(f"  Error forecasting with {bot_class.__name__}/{personality}: {str(e)}")


async def demonstrate_personality_customization():
    """Demonstrate how to customize personalities for specific use cases."""
    logger.info("\nDemonstrating personality customization...")
    
    # Initialize the personality manager
    personality_manager = PersonalityManager()
    
    # Get all available personalities
    available_personalities = personality_manager.get_all_personalities()
    logger.info(f"Available personalities: {', '.join(available_personalities)}")
    
    # Load a specific personality and view its traits
    bayesian_personality = personality_manager.load_personality("bayesian")
    logger.info(f"\nBayesian personality traits:")
    logger.info(f"  Reasoning depth: {bayesian_personality.reasoning_depth.value}")
    logger.info(f"  Thinking style: {bayesian_personality.thinking_style.value}")
    logger.info(f"  Uncertainty approach: {bayesian_personality.uncertainty_approach.value}")
    
    # Create a bot that uses the Bayesian personality
    bayesian_bot = BayesianBot(
        personality_name="bayesian",
        sensitivity_analysis=True,
        prior_source="reference_class",
    )
    
    # Show how to get a personality-specific prompt
    binary_forecast_prompt = personality_manager.get_prompt(
        "binary_forecast_prompt",
        question_text="Will X happen?",
        background_info="Some background",
        research="Research findings",
        current_date="2023-08-01"
    )
    
    logger.info(f"\nPersonality-specific prompt example (excerpt):")
    logger.info(f"  {binary_forecast_prompt[:100]}...")


async def main():
    """Run the demonstration examples."""
    # Demonstrate personality recommendation
    await demonstrate_personality_recommendation()
    
    # Demonstrate forecasting with different bots and personalities
    # Note: In a real application, you would run full forecasts
    # This is simplified for demonstration purposes
    # await demonstrate_bot_with_personality("economic")
    
    # Demonstrate personality customization
    await demonstrate_personality_customization()


if __name__ == "__main__":
    asyncio.run(main()) 