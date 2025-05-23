#!/usr/bin/env python
"""
Cost Tracking Example

This script demonstrates how to use the cost tracking functionality with ForecastingBot.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to allow running this script standalone
sys.path.insert(0, str(Path(__file__).parent.parent))

from forecasting_tools.data_models.questions import BinaryQuestion, NumericQuestion
from forecasting_tools.cost_tracking import CostTrackingBot, CostTracker

# Create data directory if it doesn't exist
data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)

# Initialize cost tracker with data file in the example directory
tracker = CostTracker(db_path=str(data_dir / "example_costs.db"))

# Print model pricing information
print("Model Pricing ($ per 1K tokens):")
for model, rate in sorted(tracker.model_rates.items()):
    print(f"  {model}: ${rate:.4f}")
print()

# Create a tracked bot
bot = CostTrackingBot(
    personality_name="analytical",
    model_name="gpt-4",
    cost_tracker=tracker
)

# Create a binary question
binary_question = BinaryQuestion(
    question_text="Will global average temperatures rise by more than 1.5°C by 2030?",
    background_info="""
    The Paris Agreement aims to limit global warming to well below 2°C, preferably to 1.5°C,
    compared to pre-industrial levels. According to the IPCC, human activities have already
    caused approximately 1.0°C of global warming above pre-industrial levels.
    """,
    resolution_criteria="Based on official IPCC or NASA temperature data for 2030",
    fine_print=""
)

# Generate a forecast
print(f"Forecasting: {binary_question.question_text}")
binary_forecast = bot.forecast_binary(binary_question)

# Display the forecast
print("\nBinary Forecast:")
print(f"Probability: {binary_forecast.binary_prob:.2%}")
print("Reasoning:")
print(binary_forecast.reasoning)

# Display cost information
if hasattr(binary_forecast, 'metadata') and 'cost_info' in binary_forecast.metadata:
    cost_info = binary_forecast.metadata['cost_info']
    print("\nCost Information:")
    print(f"Tokens used: {cost_info['tokens_used']:,}")
    print(f"Cost: ${cost_info['cost_usd']:.4f}")
    print(f"Model: {cost_info['model_name']}")
    print(f"Personality: {cost_info['personality_name'] or 'None'}")

# Create a numeric question
numeric_question = NumericQuestion(
    question_text="What will be the global average temperature increase in °C by 2050?",
    background_info="""
    According to the IPCC, human activities have already caused approximately 1.0°C of
    global warming above pre-industrial levels. The rate of temperature increase depends
    on future emissions scenarios.
    """,
    resolution_criteria="Based on official IPCC or NASA temperature data for 2050",
    fine_print="",
    min_value=0.0,
    max_value=5.0,
    unit="°C"
)

# Generate another forecast
print("\n" + "="*80 + "\n")
print(f"Forecasting: {numeric_question.question_text}")
numeric_forecast = bot.forecast_numeric(numeric_question)

# Display the forecast
print("\nNumeric Forecast:")
print(f"Mean: {numeric_forecast.mean:.2f} °C")
print(f"Range: {numeric_forecast.low:.2f} - {numeric_forecast.high:.2f} °C")
print("Reasoning:")
print(numeric_forecast.reasoning)

# Display cost information
if hasattr(numeric_forecast, 'metadata') and 'cost_info' in numeric_forecast.metadata:
    cost_info = numeric_forecast.metadata['cost_info']
    print("\nCost Information:")
    print(f"Tokens used: {cost_info['tokens_used']:,}")
    print(f"Cost: ${cost_info['cost_usd']:.4f}")
    print(f"Model: {cost_info['model_name']}")
    print(f"Personality: {cost_info['personality_name'] or 'None'}")

# Display cost statistics
print("\n" + "="*80)
print("\nCost Statistics:")
total_cost = tracker.get_total_cost()
print(f"Total cost: ${total_cost:.4f}")

# Get cost history
history = tracker.get_cost_history()
print(f"Number of forecasts: {len(history)}")

# Get costs by model
model_costs = tracker.get_cost_by_model()
print("\nCosts by model:")
for model, cost in model_costs.items():
    print(f"  {model}: ${cost:.4f}")

# Get costs by personality
personality_costs = tracker.get_cost_by_personality()
print("\nCosts by personality:")
for personality, cost in personality_costs.items():
    print(f"  {personality}: ${cost:.4f}")

# Export cost history to CSV
export_path = str(data_dir / "cost_history_export.csv")
tracker.export_to_csv(export_path)
print(f"\nExported cost history to {export_path}")

print("\nDone!") 