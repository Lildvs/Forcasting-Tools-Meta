#!/usr/bin/env python3
"""
Personality Tester CLI

This script provides a command-line tool for comparing forecasts across 
different personalities, allowing users to see how different personality
traits affect forecast outputs.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

# Add the forecasting-tools directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.forecast_report import ReasonedPrediction
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots import create_bot_with_personality
from forecasting_tools.personality_management import PersonalityManager


def create_sample_binary_question() -> BinaryQuestion:
    """Create a sample binary question for testing."""
    return BinaryQuestion(
        id=0,
        title="AI Safety Political Issue",
        question_text="Will AI safety become a major political issue by the end of 2025?",
        background_info="AI capabilities have been advancing rapidly, with models like GPT-4 and Claude 3 demonstrating unprecedented capabilities. Various political figures have started expressing concerns about AI safety.",
        resolution_criteria="This question resolves positively if AI safety becomes a significant issue in the political campaigns of at least 3 major political parties across G7 countries.",
        fine_print="For the purpose of this question, 'major political issue' means that the party has official policy positions on AI safety and candidates regularly discuss it in debates and campaign speeches.",
        publish_time=datetime.now(),
        resolve_time=datetime(2025, 12, 31),
        resolution=None,
        url="https://example.com/questions/ai-safety-politics",
        possibilities=None,
    )


def create_sample_numeric_question() -> NumericQuestion:
    """Create a sample numeric question for testing."""
    return NumericQuestion(
        id=1,
        title="Global AI Investment",
        question_text="What will be the total global investment in AI startups in billions of USD in 2025?",
        background_info="Global AI investment has been growing rapidly in recent years, with an estimated $120 billion invested in 2023.",
        resolution_criteria="This question resolves to the total amount of venture capital, private equity, and corporate investment in AI startups globally in 2025, as reported by a major financial data provider.",
        fine_print="Investment figures should be adjusted for inflation and reported in 2023 USD. Only primary investments count, not secondary market transactions.",
        publish_time=datetime.now(),
        resolve_time=datetime(2025, 12, 31),
        resolution=None,
        url="https://example.com/questions/ai-investment",
        unit_of_measure="billion USD",
        has_lower_bound=True,
        lower_bound=0,
        has_upper_bound=False,
        upper_bound=None,
    )


def create_sample_multiple_choice_question() -> MultipleChoiceQuestion:
    """Create a sample multiple choice question for testing."""
    return MultipleChoiceQuestion(
        id=2,
        title="Next AI Breakthrough",
        question_text="Which area will see the most significant AI breakthrough by the end of 2025?",
        background_info="AI research is advancing rapidly across multiple domains, including language models, robotics, and scientific discovery.",
        resolution_criteria="This question resolves to the area with the most significant AI breakthrough by the end of 2025, as determined by consensus among major AI research organizations.",
        fine_print="A 'significant breakthrough' is defined as an advancement that substantially exceeds current capabilities and is recognized as such by multiple independent experts.",
        publish_time=datetime.now(),
        resolve_time=datetime(2025, 12, 31),
        resolution=None,
        url="https://example.com/questions/ai-breakthrough",
        options=[
            MultipleChoiceQuestion.Option(id=0, option="Language Models"),
            MultipleChoiceQuestion.Option(id=1, option="Computer Vision"),
            MultipleChoiceQuestion.Option(id=2, option="Robotics"),
            MultipleChoiceQuestion.Option(id=3, option="Scientific Discovery"),
            MultipleChoiceQuestion.Option(id=4, option="Healthcare Applications"),
        ],
        option_names=["Language Models", "Computer Vision", "Robotics", "Scientific Discovery", "Healthcare Applications"],
    )


def load_question_from_file(file_path: str) -> MetaculusQuestion:
    """
    Load a question from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        A question object
        
    Raises:
        ValueError: If the question type is not supported
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Determine the question type
    question_type = data.get("type", "binary")
    
    if question_type == "binary":
        return BinaryQuestion(**data)
    elif question_type == "numeric":
        return NumericQuestion(**data)
    elif question_type == "multiple_choice":
        return MultipleChoiceQuestion(**data)
    else:
        raise ValueError(f"Unsupported question type: {question_type}")


async def forecast_with_personality(
    question: MetaculusQuestion,
    personality_name: str,
    research: str,
    model_name: str = "gpt-4o",
) -> ReasonedPrediction:
    """
    Make a forecast with a specific personality.
    
    Args:
        question: The question to forecast
        personality_name: The name of the personality to use
        research: The research to use for forecasting
        model_name: The name of the LLM model to use
        
    Returns:
        A reasoned prediction
    """
    # Create a bot with the specified personality
    bot = create_bot_with_personality(
        personality_name=personality_name,
        llms={"default": GeneralLlm(model=model_name, temperature=0.2)},
    )
    
    # Make a prediction based on the question type
    if isinstance(question, BinaryQuestion):
        prediction = await bot._run_forecast_on_binary(question, research)
    elif isinstance(question, NumericQuestion):
        prediction = await bot._run_forecast_on_numeric(question, research)
    elif isinstance(question, MultipleChoiceQuestion):
        prediction = await bot._run_forecast_on_multiple_choice(question, research)
    else:
        raise ValueError(f"Unsupported question type: {type(question)}")
    
    return prediction


async def compare_personalities(
    question: MetaculusQuestion,
    personalities: List[str],
    research: str,
    model_name: str = "gpt-4o",
) -> Dict[str, ReasonedPrediction]:
    """
    Compare forecasts across different personalities.
    
    Args:
        question: The question to forecast
        personalities: List of personality names to compare
        research: The research to use for forecasting
        model_name: The name of the LLM model to use
        
    Returns:
        Dictionary mapping personality names to reasoned predictions
    """
    results = {}
    
    for personality in personalities:
        print(f"Forecasting with {personality} personality...")
        try:
            prediction = await forecast_with_personality(
                question, personality, research, model_name
            )
            results[personality] = prediction
        except Exception as e:
            print(f"Error forecasting with {personality}: {e}")
    
    return results


def format_prediction_value(prediction_value: Any) -> str:
    """
    Format a prediction value for display.
    
    Args:
        prediction_value: The prediction value to format
        
    Returns:
        A formatted string representation of the prediction value
    """
    if isinstance(prediction_value, float):
        return f"{prediction_value:.2%}"
    elif hasattr(prediction_value, "probabilities") and hasattr(prediction_value, "option_names"):
        # Format multiple choice prediction
        formatted = []
        for i, prob in enumerate(prediction_value.probabilities):
            formatted.append(f"{prediction_value.option_names[i]}: {prob:.2%}")
        return "\n    " + "\n    ".join(formatted)
    elif hasattr(prediction_value, "to_dict"):
        # Format numeric distribution
        dist_dict = prediction_value.to_dict()
        formatted = []
        for percentile, value in sorted(dist_dict.items()):
            formatted.append(f"Percentile {percentile}: {value}")
        return "\n    " + "\n    ".join(formatted)
    else:
        return str(prediction_value)


def display_comparison(
    results: Dict[str, ReasonedPrediction],
    show_reasoning: bool = False,
) -> None:
    """
    Display a comparison of forecasts across personalities.
    
    Args:
        results: Dictionary mapping personality names to reasoned predictions
        show_reasoning: Whether to display the full reasoning for each prediction
    """
    print("\n" + "=" * 80)
    print("PERSONALITY FORECAST COMPARISON")
    print("=" * 80)
    
    for personality, prediction in results.items():
        print(f"\n{personality.upper()}")
        print("-" * len(personality))
        print(f"Prediction: {format_prediction_value(prediction.prediction_value)}")
        
        if show_reasoning:
            print("\nReasoning:")
            print(prediction.reasoning)
    
    print("\n" + "=" * 80)


async def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Compare forecasts across different personalities")
    
    parser.add_argument(
        "--question-type",
        choices=["binary", "numeric", "multiple_choice"],
        default="binary",
        help="Type of question to forecast (default: binary)",
    )
    
    parser.add_argument(
        "--question-file",
        type=str,
        help="Path to a JSON file containing a question definition",
    )
    
    parser.add_argument(
        "--research-file",
        type=str,
        help="Path to a text file containing research for the question",
    )
    
    parser.add_argument(
        "--research",
        type=str,
        default="Recent data suggests growing interest in this topic. Experts have varying opinions, with some predicting significant developments while others are more cautious.",
        help="Research text to use for forecasting (default: generic research)",
    )
    
    parser.add_argument(
        "--personalities",
        type=str,
        nargs="+",
        default=["balanced", "cautious", "creative", "economist", "bayesian"],
        help="Personalities to compare (default: all available personalities)",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model to use for forecasting (default: gpt-4o)",
    )
    
    parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help="Display the full reasoning for each prediction",
    )
    
    args = parser.parse_args()
    
    # Load question
    if args.question_file:
        question = load_question_from_file(args.question_file)
    else:
        # Use a sample question based on the question type
        if args.question_type == "binary":
            question = create_sample_binary_question()
        elif args.question_type == "numeric":
            question = create_sample_numeric_question()
        else:  # multiple_choice
            question = create_sample_multiple_choice_question()
    
    # Load research
    if args.research_file:
        with open(args.research_file, "r", encoding="utf-8") as f:
            research = f.read()
    else:
        research = args.research
    
    # Validate personalities
    manager = PersonalityManager()
    available_personalities = manager.get_all_personalities()
    
    for personality in args.personalities:
        if personality not in available_personalities:
            print(f"Warning: Unknown personality '{personality}'. Available personalities: {available_personalities}")
            sys.exit(1)
    
    # Compare forecasts
    results = await compare_personalities(
        question,
        args.personalities,
        research,
        args.model,
    )
    
    # Display results
    display_comparison(results, args.show_reasoning)


if __name__ == "__main__":
    asyncio.run(main()) 