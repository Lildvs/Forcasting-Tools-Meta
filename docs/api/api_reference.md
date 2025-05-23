# Forecasting Tools API Reference

This document provides detailed information about the available API endpoints, classes, and methods in the Forecasting Tools library.

## Core Components

### Forecasting API

The Forecasting API provides endpoints for submitting forecast questions, retrieving results, and managing forecasts.

#### Forecast Submission

```python
from forecasting_tools.api import ForecastingAPI

# Initialize the API client
api = ForecastingAPI()

# Submit a forecast question
job_id = api.submit_forecast(
    question="Will AI replace programmers by 2030?",
    forecast_type="binary",
    search_queries=["AI impact on programming jobs", "software engineering future"],
    user_id="user123"
)

# Check forecast status
status = api.get_forecast_status(job_id)

# Retrieve completed forecast
if status["status"] == "completed":
    forecast = api.get_forecast(job_id)
    print(f"Probability: {forecast['probability']}")
    print(f"Reasoning: {forecast['reasoning']}")
```

#### Available Methods

| Method | Description | Parameters | Return Type |
|--------|-------------|------------|-------------|
| `submit_forecast` | Submit a new forecast question | `question` (str), `forecast_type` (str), `search_queries` (List[str]), `user_id` (str), `priority` (JobPriority, optional) | `job_id` (str) |
| `get_forecast_status` | Check the status of a forecast job | `job_id` (str) | Dict[str, Any] |
| `get_forecast` | Retrieve a completed forecast | `job_id` (str), `include_research` (bool, optional) | Dict[str, Any] |
| `list_forecasts` | List all forecasts for a user | `user_id` (str), `limit` (int, optional), `offset` (int, optional) | List[Dict[str, Any]] |
| `update_forecast` | Update an existing forecast | `forecast_id` (str), `updates` (Dict[str, Any]) | bool |
| `delete_forecast` | Delete a forecast | `forecast_id` (str) | bool |

#### Forecast Types

The system supports the following forecast types:

| Type | Description | Example |
|------|-------------|---------|
| `binary` | Yes/No questions with probability | "Will Bitcoin exceed $100,000 by the end of 2024?" |
| `numeric` | Forecasts of numeric values with uncertainty ranges | "What will the S&P 500 close at on December 31, 2023?" |
| `multiple_choice` | Selection among defined options with probabilities | "Which country will win the most gold medals at the 2024 Olympics?" |
| `date` | Predicting when an event will occur | "When will human-level AGI be developed?" |

### Storage Manager

The Storage Manager provides access to stored forecasts and research data.

```python
from forecasting_tools.data.storage import StorageManager, SqliteStorage

# Initialize with a storage adapter
storage = StorageManager(SqliteStorage("forecasts.db"))

# Save forecast data
forecast_id = storage.save_forecast({
    "question": "Will quantum computing be commercially viable by 2030?",
    "forecast_type": "binary",
    "probability": 0.65,
    "reasoning": "Based on current research pace...",
    "user_id": "user123",
    "metadata": {"sources": [...]}
})

# Retrieve forecast
forecast = storage.get_forecast(forecast_id)

# Save research data linked to a forecast
research_id = storage.save_research_data(
    forecast_id=forecast_id,
    data={
        "search_queries": ["quantum computing timeline", "quantum supremacy"],
        "results": [...]
    }
)

# Retrieve research data
research = storage.get_research_data(research_id)
```

#### Available Methods

| Method | Description | Parameters | Return Type |
|--------|-------------|------------|-------------|
| `save_forecast` | Save a new forecast | `forecast_data` (Dict[str, Any]) | `forecast_id` (str) |
| `get_forecast` | Retrieve a forecast | `forecast_id` (str) | Dict[str, Any] |
| `update_forecast` | Update an existing forecast | `forecast_id` (str), `updates` (Dict[str, Any]) | bool |
| `delete_forecast` | Delete a forecast | `forecast_id` (str) | bool |
| `save_research_data` | Save research data for a forecast | `forecast_id` (str), `data` (Dict[str, Any]) | `research_id` (str) |
| `get_research_data` | Retrieve research data | `research_id` (str) | Dict[str, Any] |
| `get_research_for_forecast` | Get all research for a forecast | `forecast_id` (str) | List[Dict[str, Any]] |
| `list_forecasts` | List forecasts with optional filtering | `filters` (Dict[str, Any], optional), `limit` (int, optional), `offset` (int, optional) | List[Dict[str, Any]] |

### Queue Manager

The Queue Manager handles asynchronous processing of forecast jobs.

```python
from forecasting_tools.util.queue_manager import QueueManager, JobType, JobPriority

# Initialize the queue manager
queue = QueueManager(max_workers=4)

# Define a handler function
def handle_forecast_job(job_data):
    question = job_data["question"]
    # Process the forecast
    return {"status": "success", "result": {...}}

# Register handlers
queue.register_handler(JobType.FORECAST, handle_forecast_job)

# Enqueue a job
job_id = queue.enqueue(
    job_type=JobType.FORECAST,
    job_data={
        "question": "Will SpaceX land humans on Mars by 2030?",
        "user_id": "user123"
    },
    priority=JobPriority.HIGH
)

# Check job status
status = queue.get_job_status(job_id)

# Get job result when completed
if status["status"] == "completed":
    result = queue.get_job_result(job_id)
```

#### Available Methods

| Method | Description | Parameters | Return Type |
|--------|-------------|------------|-------------|
| `enqueue` | Add a job to the queue | `job_type` (JobType), `job_data` (Dict[str, Any]), `priority` (JobPriority, optional), `max_retries` (int, optional) | `job_id` (str) |
| `register_handler` | Register a function to handle a job type | `job_type` (JobType), `handler` (Callable) | None |
| `get_job_status` | Check job status | `job_id` (str) | Dict[str, Any] |
| `get_job_result` | Get the result of a completed job | `job_id` (str) | Dict[str, Any] |
| `cancel_job` | Cancel a job | `job_id` (str) | bool |
| `pause_queue` | Pause processing new jobs | None | None |
| `resume_queue` | Resume processing jobs | None | None |
| `shutdown` | Shutdown the queue manager | `wait` (bool, optional) | None |

### Monitoring System

The Monitoring System tracks the system's performance and usage.

```python
from forecasting_tools.utils.monitoring import get_monitoring_manager

# Get the monitoring manager
monitor = get_monitoring_manager()

# Record API usage
with monitor.api_call_timer("OpenAI", "completion"):
    # Make API call here
    pass

# Track custom metrics
monitor.increment_counter("forecasts_created")

# Track error rates
try:
    # Some operation
    pass
except Exception as e:
    monitor.record_error("forecast_generation", str(e))

# Get current metrics
metrics = monitor.get_current_metrics()
```

#### Available Methods

| Method | Description | Parameters | Return Type |
|--------|-------------|------------|-------------|
| `api_call_timer` | Context manager to time API calls | `provider` (str), `endpoint` (str) | Context manager |
| `increment_counter` | Increment a counter metric | `name` (str), `value` (int, optional) | None |
| `record_gauge` | Record a gauge value | `name` (str), `value` (float) | None |
| `record_error` | Record an error occurrence | `category` (str), `message` (str) | None |
| `start_timer` | Start a timer for operation duration | `operation` (str) | Timer ID |
| `stop_timer` | Stop a timer and record duration | `timer_id` (str) | None |
| `get_current_metrics` | Get current metrics snapshot | None | Dict[str, Any] |
| `add_custom_metric` | Add a custom metric | `name` (str), `type` (MetricType), `description` (str) | None |

## Evaluation Components

### Forecast Evaluator

The Forecast Evaluator measures the accuracy and quality of forecasts.

```python
from forecasting_tools.evaluation.scoring import ForecastEvaluator

# Initialize with a forecast file containing outcomes
evaluator = ForecastEvaluator("historical_predictions.json")

# Evaluate binary forecasts
binary_results = evaluator.evaluate_binary_forecasts()
print(f"Brier score: {binary_results['brier_score']}")
print(f"Accuracy: {binary_results['accuracy']}")

# Evaluate all forecast types
all_results = evaluator.evaluate_all()

# Generate a comprehensive report
report = evaluator.generate_report()
```

#### Available Methods

| Method | Description | Parameters | Return Type |
|--------|-------------|------------|-------------|
| `load_forecasts` | Load forecasts from a file | `file_path` (str) | None |
| `evaluate_binary_forecasts` | Evaluate binary forecasts | None | Dict[str, Any] |
| `evaluate_numeric_forecasts` | Evaluate numeric forecasts | None | Dict[str, Any] |
| `evaluate_multiple_choice_forecasts` | Evaluate multiple choice forecasts | None | Dict[str, Any] |
| `evaluate_all` | Evaluate all forecast types | None | Dict[str, Any] |
| `evaluate_by_category` | Evaluate forecasts grouped by category | None | Dict[str, Dict[str, Any]] |
| `evaluate_by_difficulty` | Evaluate forecasts grouped by difficulty | None | Dict[str, Dict[str, Any]] |
| `generate_calibration_plot_data` | Generate data for calibration plots | None | Dict[str, List[Dict[str, Any]]] |
| `generate_report` | Generate a comprehensive evaluation report | None | Dict[str, Any] |

### Reasoning Evaluator

The Reasoning Evaluator assesses the quality of forecast reasoning.

```python
from forecasting_tools.evaluation.scoring import ReasoningEvaluator

# Evaluate a single reasoning text
reasoning = "The probability of this event is high because..."
eval_result = ReasoningEvaluator.evaluate_reasoning(reasoning)
print(f"Quality score: {eval_result['quality_score']}")

# Batch evaluate multiple forecasts
with open("forecasts.json", "r") as f:
    forecasts = json.load(f)

batch_results = ReasoningEvaluator.batch_evaluate_reasoning(forecasts)
print(f"Average quality: {batch_results['average_quality_score']}")
```

#### Available Methods

| Method | Description | Parameters | Return Type |
|--------|-------------|------------|-------------|
| `evaluate_reasoning` | Evaluate a single reasoning text | `reasoning` (str) | Dict[str, Any] |
| `batch_evaluate_reasoning` | Evaluate reasoning for multiple forecasts | `forecasts` (List[Dict[str, Any]]) | Dict[str, Any] |

## Utility Components

### Async Helpers

The Async Helpers module provides utilities for asynchronous operations.

```python
from forecasting_tools.util.async_helpers import BatchProcessor, RateLimiter, APIClient, timeout

# Process items in batches
processor = BatchProcessor(
    batch_size=10,
    process_fn=lambda batch: [item.upper() for item in batch]
)
results = await processor.process_items(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"])

# Rate-limited API calls
rate_limiter = RateLimiter(
    max_calls=60,
    time_period=60  # 60 calls per minute
)

async with rate_limiter:
    # Make API call here
    pass

# API client with retries
client = APIClient(
    base_url="https://api.example.com",
    retry_attempts=3,
    timeout=10
)

response = await client.get("/users/123")

# Timeout decorator for async functions
@timeout(5.0)
async def long_running_operation():
    # Operation that might take too long
    pass
```

#### Available Classes and Functions

| Name | Type | Description |
|------|------|-------------|
| `BatchProcessor` | Class | Process items in batches asynchronously |
| `RateLimiter` | Class | Limit the rate of operations |
| `APIClient` | Class | Client for making API requests with retries |
| `timeout` | Decorator | Apply a timeout to an async function |
| `run_async` | Function | Run an async function from synchronous code |
| `gather_with_concurrency` | Function | Run async tasks with concurrency limit |

### Configuration System

The Configuration System manages application settings.

```python
from forecasting_tools.config import Config

# Get configuration for the current environment
config = Config.get_instance()

# Access configuration values
db_url = config.get("database.url")
api_key = config.get("api.openai.key")

# Check environment
if config.is_production():
    # Production-specific logic
    pass
elif config.is_development():
    # Development-specific logic
    pass
```

#### Available Methods

| Method | Description | Parameters | Return Type |
|--------|-------------|------------|-------------|
| `get_instance` | Get the singleton instance | None | Config |
| `get` | Get a configuration value | `key` (str), `default` (Any, optional) | Any |
| `set` | Set a configuration value | `key` (str), `value` (Any) | None |
| `is_production` | Check if running in production | None | bool |
| `is_development` | Check if running in development | None | bool |
| `is_testing` | Check if running in test mode | None | bool |
| `get_environment` | Get the current environment | None | str |
| `load_from_file` | Load configuration from a file | `file_path` (str) | None |

## Data Models

### ORM Models

The ORM Models provide a structured representation of system data.

```python
from forecasting_tools.data.models import DatabaseManager, User, Forecast, ResearchData

# Initialize the database manager
db_manager = DatabaseManager("sqlite:///forecasts.db")
db_manager.create_tables()

# Use the session scope for database operations
with db_manager.session_scope() as session:
    # Create a new user
    user = User(
        username="testuser",
        email="test@example.com",
        password_hash="hashed_password"
    )
    session.add(user)
    session.flush()  # Flush to get the ID
    
    # Create a forecast
    forecast = Forecast(
        question_text="Will quantum computing be commercially viable by 2030?",
        forecast_type="binary",
        probability=0.65,
        reasoning="Based on current development pace...",
        user_id=user.id
    )
    session.add(forecast)
    session.flush()
    
    # Add research data
    research = ResearchData(
        forecast_id=forecast.id,
        source_url="https://example.com/quantum-research",
        content="Research about quantum computing timeline...",
        relevance_score=0.85
    )
    session.add(research)
```

#### Available Models

| Model | Description | Key Fields |
|-------|-------------|------------|
| `User` | User information | `id`, `username`, `email`, `password_hash`, `created_at` |
| `Forecast` | Forecast data | `id`, `question_text`, `forecast_type`, `probability`, `reasoning`, `user_id`, `created_at`, `updated_at` |
| `ForecastHistory` | Historical changes to forecasts | `id`, `forecast_id`, `probability`, `reasoning`, `timestamp` |
| `ResearchData` | Research information for forecasts | `id`, `forecast_id`, `source_url`, `content`, `relevance_score` |
| `Tag` | Tags for organizing forecasts | `id`, `name` |
| `UserInteraction` | User interactions with the system | `id`, `user_id`, `interaction_type`, `timestamp`, `metadata` |
| `JobQueue` | Queue entries for background jobs | `id`, `job_type`, `status`, `data`, `priority`, `created_at`, `updated_at` |

## Error Handling

### Exception Types

The library uses custom exceptions for different error scenarios:

| Exception | Description |
|-----------|-------------|
| `ForecastingError` | Base class for all forecasting errors |
| `APIError` | Errors related to external API calls |
| `StorageError` | Errors related to data storage |
| `PersistenceError` | Errors related to data persistence |
| `ConfigurationError` | Errors related to system configuration |
| `QueueError` | Errors related to the job queue |
| `TimeoutError` | Errors related to operation timeouts |
| `ValidationError` | Errors related to data validation |

### Error Response Format

API errors are returned in a consistent format:

```json
{
  "error": {
    "code": "storage_error",
    "message": "Failed to save forecast data",
    "details": {
      "reason": "Database connection failed",
      "request_id": "req-123456"
    }
  }
}
```

## Conclusion

This API reference covers the main components, classes, and methods of the Forecasting Tools library. For more detailed information, refer to the specific module documentation or the usage examples in the example directory. 