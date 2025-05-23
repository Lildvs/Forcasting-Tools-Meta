#!/usr/bin/env python
"""
Personality Demo Script

This script demonstrates how to use the personality templates system to customize
forecasting bots with different personalities.
"""

import argparse
import asyncio
import logging
from datetime import datetime
from typing import List, Optional

import dotenv

from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.questions import BinaryQuestion
from forecasting_tools.forecast_bots.enhanced_template_bot import (
    EnhancedTemplateBotQ2, 
    EnhancedTemplateBotQ4
)
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi
from forecasting_tools.personality_templates import PersonalityConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()


async def forecast_with_personality(
    question_id: int,
    personality_name: str,
    bot_version: str = "q2",
    publish_to_metaculus: bool = False,
) -> ForecastReport:
    """
    Run a forecast with a specified personality on a specific question.
    
    Args:
        question_id: The Metaculus question ID to forecast on
        personality_name: The personality to use
        bot_version: The bot version to use
        publish_to_metaculus: Whether to publish the forecast to Metaculus
        
    Returns:
        The forecast report
    """
    # Get the question from Metaculus
    question = MetaculusApi.get_question(question_id)
    logger.info(f"Forecasting on question: {question.question_text}")
    
    # Create the bot with the specified personality
    if bot_version.lower() == "q2":
        bot = EnhancedTemplateBotQ2(
            personality_name=personality_name,
            publish_reports_to_metaculus=publish_to_metaculus
        )
    elif bot_version.lower() == "q4":
        bot = EnhancedTemplateBotQ4(
            personality_name=personality_name,
            publish_reports_to_metaculus=publish_to_metaculus
        )
    else:
        raise ValueError(f"Unsupported bot version: {bot_version}")
    
    logger.info(f"Using bot version {bot_version} with personality: {personality_name}")
    
    # Run the forecast
    report = await bot.forecast_question(question)
    
    # Print the result
    if isinstance(question, BinaryQuestion):
        logger.info(f"Prediction: {report.prediction * 100:.1f}%")
    else:
        logger.info(f"Prediction: {report.prediction}")
    
    return report


async def compare_personalities(
    question_id: int,
    personalities: List[str],
    bot_version: str = "q2",
    publish_to_metaculus: bool = False,
) -> None:
    """
    Compare forecasts from different personalities on the same question.
    
    Args:
        question_id: The Metaculus question ID to forecast on
        personalities: List of personalities to compare
        bot_version: The bot version to use
        publish_to_metaculus: Whether to publish the forecasts to Metaculus
    """
    reports = []
    
    # Run forecasts with each personality
    for personality in personalities:
        try:
            report = await forecast_with_personality(
                question_id, 
                personality, 
                bot_version, 
                publish_to_metaculus
            )
            reports.append((personality, report))
        except Exception as e:
            logger.error(f"Error with personality {personality}: {e}")
    
    # Print comparison
    logger.info("\n" + "=" * 50)
    logger.info("PERSONALITY COMPARISON")
    logger.info("=" * 50)
    
    for personality, report in reports:
        if isinstance(report.prediction, float):  # Binary question
            logger.info(f"{personality}: {report.prediction * 100:.1f}%")
        else:
            logger.info(f"{personality}: {report.prediction}")
    
    logger.info("=" * 50)


async def list_personalities() -> None:
    """List all available personalities."""
    personalities = PersonalityConfig.get_available_personalities()
    
    logger.info("\n" + "=" * 50)
    logger.info("AVAILABLE PERSONALITIES")
    logger.info("=" * 50)
    
    for personality in personalities:
        config = PersonalityConfig(personality_name=personality)
        logger.info(f"{config.get_name()}: {config.get_description()}")
    
    logger.info("=" * 50)


async def main():
    parser = argparse.ArgumentParser(description="Personality demo for forecasting bots")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List personalities command
    list_parser = subparsers.add_parser("list", help="List available personalities")
    
    # Forecast command
    forecast_parser = subparsers.add_parser("forecast", help="Forecast with a specific personality")
    forecast_parser.add_argument("question_id", type=int, help="Metaculus question ID")
    forecast_parser.add_argument("personality", type=str, help="Personality to use")
    forecast_parser.add_argument("--bot-version", type=str, choices=["q2", "q4"], default="q2", help="Bot version to use")
    forecast_parser.add_argument("--publish", action="store_true", help="Publish to Metaculus")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare different personalities")
    compare_parser.add_argument("question_id", type=int, help="Metaculus question ID")
    compare_parser.add_argument("--personalities", type=str, nargs="+", default=["balanced", "cautious", "creative"], help="Personalities to compare")
    compare_parser.add_argument("--bot-version", type=str, choices=["q2", "q4"], default="q2", help="Bot version to use")
    compare_parser.add_argument("--publish", action="store_true", help="Publish to Metaculus")
    
    args = parser.parse_args()
    
    if args.command == "list":
        await list_personalities()
    elif args.command == "forecast":
        await forecast_with_personality(
            args.question_id, 
            args.personality, 
            args.bot_version, 
            args.publish
        )
    elif args.command == "compare":
        await compare_personalities(
            args.question_id, 
            args.personalities, 
            args.bot_version, 
            args.publish
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main()) 