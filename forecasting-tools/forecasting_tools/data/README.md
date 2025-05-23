# Forecasting Tools Backend Optimization

This directory contains the optimized backend components for the Forecasting Tools application. These components focus on performance, scalability, and reliability.

## Key Components

### 1. Asynchronous Processing (`util/async_helpers.py`)

The asynchronous processing module provides utilities for handling multiple API calls simultaneously:

- `RateLimiter`: Prevents exceeding API rate limits
- `APIClient`: Asynchronous API client with retries and timeout handling
- `BatchProcessor`: Process items in batches with concurrent execution
- Utilities for timeouts and sync/async conversion

### 2. Data Persistence (`data/storage.py`)

The data persistence layer provides storage for forecasts, research results, and user interactions:

- Abstract `StorageAdapter` interface for different backends
- `SqliteStorage`: Optimized SQLite storage with connection pooling
- `MemoryStorage`: In-memory storage for testing and caching
- `StorageManager`: High-level interface for storage operations

### 3. Database Models (`data/models.py`)

The database models provide structured storage with SQLAlchemy ORM:

- `User`, `Forecast`, `ResearchData`, and other core models
- Optimized indexes for query performance
- Connection pooling and session management
- Serialization methods for API responses

### 4. Job Queue System (`util/queue_manager.py`)

The job queue system handles forecast generation requests and background tasks:

- Priority-based job queuing
- Concurrent execution with worker pools
- Task retry mechanism with exponential backoff
- Timeout monitoring and graceful handling

### 5. Configuration (`config.py`)

The enhanced configuration system supports different deployment environments:

- Environment-specific settings (development, staging, production)
- Database connection management
- Caching strategies with tiered options
- Logging and monitoring configuration

## Usage Examples

### Asynchronous API Calls

```python
from forecasting_tools.util.async_helpers import APIClient

async def fetch_data():
    async with APIClient("https://api.example.com") as client:
        result = await client.request("GET", "endpoint")
    return result
```

### Data Storage

```python
from forecasting_tools.data.storage import SqliteStorage, StorageManager

# Initialize storage
storage = SqliteStorage("forecasts.db")
manager = StorageManager(storage)

# Save and retrieve data
forecast_id = manager.save_forecast(forecast_data)
forecast = manager.get_forecast(forecast_id)
```

### Database Operations

```python
from forecasting_tools.data.models import DatabaseManager, Forecast

# Initialize database
db_manager = DatabaseManager("sqlite:///forecasts.db")
db_manager.create_tables()

# Use session for database operations
with db_manager.session_scope() as session:
    forecast = Forecast(question_text="Will AI impact jobs?", forecast_type="binary")
    session.add(forecast)
```

### Job Queue

```python
from forecasting_tools.util.queue_manager import queue_manager_context, JobType

# Process a forecast in the background
def process_forecast(data, job):
    # Process the forecast
    return result

# Use the queue manager
with queue_manager_context() as manager:
    manager.register_handler(JobType.FORECAST, process_forecast)
    job_id = manager.enqueue(JobType.FORECAST, forecast_data)
```

## Performance Considerations

- **Connection Pooling**: Database connections are managed in pools to reduce overhead
- **Batch Processing**: Tasks can be processed in batches for improved throughput
- **Caching Strategies**: Multiple caching levels reduce API calls and database load
- **Concurrency Control**: Rate limits prevent overwhelming external services
- **Error Handling**: Comprehensive error handling with retries improves reliability 