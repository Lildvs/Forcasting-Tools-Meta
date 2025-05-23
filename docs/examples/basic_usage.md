# Basic Usage Examples

This document provides examples of common usage patterns for the Forecasting Tools library.

## Installation

```bash
pip install forecasting-tools
```

## Basic Forecast Creation

The simplest way to create a forecast is through the ForecastingAPI:

```python
from forecasting_tools.api import ForecastingAPI

# Initialize the API client
api = ForecastingAPI()

# Submit a binary forecast question
job_id = api.submit_forecast(
    question="Will SpaceX successfully land humans on Mars before 2030?",
    forecast_type="binary",
    search_queries=[
        "SpaceX Mars mission timeline",
        "Mars colonization plans",
        "SpaceX Starship development"
    ],
    user_id="user123"
)

# Check the status of the forecast
status = api.get_forecast_status(job_id)
print(f"Forecast status: {status['status']}")

# Once completed, retrieve the forecast
if status["status"] == "completed":
    forecast = api.get_forecast(job_id)
    print(f"Question: {forecast['question']}")
    print(f"Probability: {forecast['probability']}")
    print(f"Reasoning: {forecast['reasoning']}")
```

## Working with Different Forecast Types

The library supports multiple forecast types:

### Binary Forecasts

```python
# Binary forecast (yes/no question with probability)
job_id = api.submit_forecast(
    question="Will Bitcoin exceed $100,000 by the end of 2024?",
    forecast_type="binary",
    search_queries=["Bitcoin price predictions", "cryptocurrency market analysis"],
    user_id="user123"
)
```

### Numeric Forecasts

```python
# Numeric forecast (with uncertainty ranges)
job_id = api.submit_forecast(
    question="What will the S&P 500 close at on December 31, 2023?",
    forecast_type="numeric",
    search_queries=["S&P 500 predictions 2023", "stock market outlook"],
    user_id="user123"
)

# Retrieve a numeric forecast
forecast = api.get_forecast(job_id)
print(f"Question: {forecast['question']}")
print(f"Median estimate: {forecast['distribution']['median']}")
print(f"10th percentile: {forecast['distribution']['p10']}")
print(f"90th percentile: {forecast['distribution']['p90']}")
print(f"Reasoning: {forecast['reasoning']}")
```

### Multiple Choice Forecasts

```python
# Multiple choice forecast
job_id = api.submit_forecast(
    question="Which country will win the most gold medals at the 2024 Olympics?",
    forecast_type="multiple_choice",
    options=["USA", "China", "Russia", "Japan", "France"],
    search_queries=["Olympics medal predictions", "2024 Olympics favorite countries"],
    user_id="user123"
)

# Retrieve a multiple choice forecast
forecast = api.get_forecast(job_id)
print(f"Question: {forecast['question']}")
for option in forecast['options']:
    print(f"{option['option']}: {option['probability']}")
print(f"Reasoning: {forecast['reasoning']}")
```

## Custom Storage Configuration

You can configure custom storage options:

```python
from forecasting_tools.api import ForecastingAPI
from forecasting_tools.data.storage import SqliteStorage

# Create a storage adapter
storage = SqliteStorage("my_forecasts.db")

# Initialize the API with custom storage
api = ForecastingAPI(storage=storage)

# The API will now use your custom storage
```

## Working with the Queue Manager Directly

For more control over the forecasting process, you can use the Queue Manager directly:

```python
from forecasting_tools.util.queue_manager import QueueManager, JobType, JobPriority

# Initialize the queue manager
queue_manager = QueueManager(max_workers=4)

# Define a handler for forecast jobs
def handle_forecast_job(job_data):
    question = job_data["question"]
    # Process the forecast...
    return {
        "question": question,
        "probability": 0.75,
        "reasoning": "Based on the evidence..."
    }

# Register the handler
queue_manager.register_handler(JobType.FORECAST, handle_forecast_job)

# Enqueue a high-priority forecast job
job_id = queue_manager.enqueue(
    job_type=JobType.FORECAST,
    job_data={
        "question": "Will quantum computing achieve practical commercial applications by 2025?",
        "search_queries": ["quantum computing timeline", "quantum supremacy"]
    },
    priority=JobPriority.HIGH
)

# Check job status
status = queue_manager.get_job_status(job_id)
print(f"Job status: {status['status']}")

# Get job result when completed
if status["status"] == "completed":
    result = queue_manager.get_job_result(job_id)
    print(f"Result: {result}")
```

## Batch Processing of Multiple Forecasts

For efficiently processing multiple forecasts:

```python
from forecasting_tools.api import ForecastingAPI
import asyncio

# Initialize the API client
api = ForecastingAPI()

# Define a list of forecast questions
forecast_questions = [
    {
        "question": "Will AI systems be capable of writing novel-length fiction by 2025?",
        "forecast_type": "binary",
        "search_queries": ["AI fiction writing", "large language model capabilities"]
    },
    {
        "question": "Will global average temperatures increase by more than 1.5°C by 2030?",
        "forecast_type": "binary",
        "search_queries": ["climate change projections", "global warming trends"]
    },
    {
        "question": "What will Tesla's stock price be on December 31, 2023?",
        "forecast_type": "numeric",
        "search_queries": ["Tesla stock predictions", "EV market analysis"]
    }
]

# Submit all forecasts
job_ids = []
for question_data in forecast_questions:
    job_id = api.submit_forecast(
        question=question_data["question"],
        forecast_type=question_data["forecast_type"],
        search_queries=question_data["search_queries"],
        user_id="user123"
    )
    job_ids.append(job_id)

# Wait for all forecasts to complete
all_completed = False
while not all_completed:
    statuses = [api.get_forecast_status(job_id)["status"] for job_id in job_ids]
    all_completed = all("completed" == status for status in statuses)
    if not all_completed:
        print("Waiting for forecasts to complete...")
        time.sleep(5)

# Retrieve all results
forecasts = [api.get_forecast(job_id) for job_id in job_ids]
for forecast in forecasts:
    print(f"Question: {forecast['question']}")
    if forecast.get("probability"):
        print(f"Probability: {forecast['probability']}")
    elif "distribution" in forecast:
        print(f"Median: {forecast['distribution']['median']}")
    print(f"Reasoning: {forecast['reasoning']}")
    print("-" * 50)
```

## Working with Forecast Evaluation

To evaluate forecasts against actual outcomes:

```python
from forecasting_tools.evaluation.scoring import ForecastEvaluator, ReasoningEvaluator
import json

# Load historical forecasts with outcomes
with open("historical_predictions.json", "r") as f:
    historical_data = json.load(f)

# Create an evaluator
evaluator = ForecastEvaluator()
evaluator.forecasts = historical_data["predictions"]

# Evaluate binary forecasts
binary_results = evaluator.evaluate_binary_forecasts()
print(f"Number of binary forecasts: {binary_results['count']}")
print(f"Brier score: {binary_results['brier_score']}")
print(f"Accuracy: {binary_results['accuracy']}")
print(f"Calibration reliability: {binary_results['calibration']['reliability']}")

# Evaluate by category
category_results = evaluator.evaluate_by_category()
for category, results in category_results.items():
    print(f"\nCategory: {category}")
    binary = results.get("binary", {})
    if "count" in binary:
        print(f"  Binary forecasts: {binary['count']}")
        print(f"  Binary accuracy: {binary['accuracy']}")

# Evaluate reasoning quality
reasoning_evaluator = ReasoningEvaluator()
reasoning_results = reasoning_evaluator.batch_evaluate_reasoning(evaluator.forecasts)
print(f"\nReasoning Quality:")
print(f"Average quality score: {reasoning_results['average_quality_score']}")
print(f"Quantitative reasoning: {reasoning_results['quantitative_reasoning_percentage']}")
print(f"Causal reasoning: {reasoning_results['causal_reasoning_percentage']}")
```

## Using the Monitoring System

To track system performance and usage:

```python
from forecasting_tools.utils.monitoring import get_monitoring_manager
import time

# Get the monitoring manager
monitor = get_monitoring_manager()

# Track API usage
with monitor.api_call_timer("OpenAI", "completion"):
    # Simulate an API call
    time.sleep(1.0)

# Track custom metrics
monitor.increment_counter("forecasts_created")
monitor.record_gauge("active_users", 42)

# Track operation duration
timer_id = monitor.start_timer("forecast_generation")
# Perform operation
time.sleep(0.5)
monitor.stop_timer(timer_id)

# Get current metrics
metrics = monitor.get_current_metrics()
print(f"API calls: {metrics['api_calls']}")
print(f"Operation durations: {metrics['operation_durations']}")
print(f"Custom metrics: {metrics['custom_metrics']}")
```

## Advanced Configuration

For advanced configuration options:

```python
from forecasting_tools.config import Config

# Get configuration instance
config = Config.get_instance()

# Set configuration values
config.set("api.openai.max_tokens", 2000)
config.set("storage.cache_expiry", 3600)  # 1 hour
config.set("queue.max_workers", 8)

# Load configuration from a file
config.load_from_file("config.yaml")

# Access configuration values
api_key = config.get("api.openai.key")
db_url = config.get("database.url")

# Environment-specific configuration
if config.is_production():
    # Use production settings
    log_level = "WARNING"
else:
    # Use development settings
    log_level = "DEBUG"
```

## Error Handling

Proper error handling:

```python
from forecasting_tools.api import ForecastingAPI
from forecasting_tools.exceptions import (
    ForecastingError, APIError, StorageError, QueueError, TimeoutError
)

# Initialize the API client
api = ForecastingAPI()

try:
    # Submit a forecast
    job_id = api.submit_forecast(
        question="Will quantum computing break current encryption by 2030?",
        forecast_type="binary",
        search_queries=["quantum computing encryption", "post-quantum cryptography"],
        user_id="user123"
    )
    
    # Check status and get result
    status = api.get_forecast_status(job_id)
    if status["status"] == "completed":
        forecast = api.get_forecast(job_id)
        print(f"Probability: {forecast['probability']}")
    
except APIError as e:
    print(f"API Error: {e}")
except StorageError as e:
    print(f"Storage Error: {e}")
except QueueError as e:
    print(f"Queue Error: {e}")
except TimeoutError as e:
    print(f"Timeout Error: {e}")
except ForecastingError as e:
    # Base class for all forecasting errors
    print(f"Forecasting Error: {e}")
except Exception as e:
    # Unexpected error
    print(f"Unexpected error: {e}")
```

## Conclusion

These examples demonstrate the basic usage patterns for the Forecasting Tools library. For more detailed information, refer to the API reference documentation and the system architecture documentation. 