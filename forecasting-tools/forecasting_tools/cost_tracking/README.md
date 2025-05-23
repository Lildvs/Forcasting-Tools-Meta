# Cost Tracking System

This module provides functionality to track and analyze token usage and costs for forecasts made with different models and personalities.

## Features

- **Accurate Cost Calculation:** Calculate costs based on model-specific token pricing
- **Persistent Storage:** Store cost history in SQLite database
- **Thread Safety:** Thread-safe operations for multi-user environments
- **Rich Analytics:** Aggregate costs by date range, model, personality, etc.
- **Integration Options:** Multiple ways to integrate with ForecastingBot

## Getting Started

### Basic Usage

```python
from forecasting_tools import ForecastingBot
from forecasting_tools.cost_tracking import CostTrackingBot
from forecasting_tools.data_models.questions import BinaryQuestion

# Create a tracking bot
bot = CostTrackingBot(
    personality_name="analytical",
    model_name="gpt-4"
)

# Create a question
question = BinaryQuestion(
    question_text="Will GDP growth exceed 2% next year?",
    background_info="Current growth trends are mixed.",
    resolution_criteria="Official government statistics",
    fine_print=""
)

# Generate forecast
forecast = bot.forecast_binary(question)

# Access cost information from the result
cost_info = forecast.metadata.get('cost_info', {})
print(f"Cost: ${cost_info.get('cost_usd', 0):.4f}")
print(f"Tokens used: {cost_info.get('tokens_used', 0)}")

# Get cost history
history = bot.get_cost_history(limit=10)
print(f"Total cost: ${bot.get_total_cost():.2f}")
```

### Using the Mixin

```python
from forecasting_tools import ForecastingBot
from forecasting_tools.cost_tracking import CostTrackingMixin
from forecasting_tools.data_models.questions import BinaryQuestion

# Create a custom bot class with cost tracking
class MyTrackingBot(CostTrackingMixin, ForecastingBot):
    pass

# Instantiate the bot
bot = MyTrackingBot(personality_name="analytical")

# Use as normal
question = BinaryQuestion(...)
forecast = bot.forecast_binary(question)
```

### Using the Decorator

```python
from forecasting_tools import ForecastingBot
from forecasting_tools.cost_tracking import with_cost_tracking
from forecasting_tools.data_models.questions import BinaryQuestion

# Apply the decorator to any bot class
@with_cost_tracking
class MyCustomBot(ForecastingBot):
    pass

# Instantiate and use
bot = MyCustomBot()
```

## Configuration

The cost tracker can be configured with custom database paths:

```python
from forecasting_tools.cost_tracking import CostTracker, CostTrackingBot

# Create a custom tracker
tracker = CostTracker(db_path="/path/to/custom/database.db")

# Use the custom tracker with a bot
bot = CostTrackingBot(cost_tracker=tracker)
```

## Cost Analysis

The system provides various methods for cost analysis:

```python
from forecasting_tools.cost_tracking import CostTracker

tracker = CostTracker()

# Get total cost
total = tracker.get_total_cost()

# Get cost by date range
from datetime import datetime, timedelta
start_date = datetime.now() - timedelta(days=30)
end_date = datetime.now()
monthly_cost = tracker.get_cost_by_date_range(start_date, end_date)

# Get costs by model
model_costs = tracker.get_cost_by_model()

# Get costs by personality
personality_costs = tracker.get_cost_by_personality()

# Get daily costs
daily_costs = tracker.get_daily_costs(days=30)

# Get comprehensive statistics
stats = tracker.get_cost_statistics()
```

## Data Export

The cost history can be exported to CSV for further analysis:

```python
tracker = CostTracker()
tracker.export_to_csv("cost_history.csv")
```

## Model Pricing

The system includes default pricing for common models, which can be customized if needed:

```python
tracker = CostTracker()

# View current pricing
print(tracker.model_rates)

# Update pricing
tracker.model_rates["gpt-4"] = 0.04  # $0.04 per 1K tokens
```

## Streamlit Integration

The cost tracking system works seamlessly with Streamlit for creating interactive dashboards:

```python
import streamlit as st
from forecasting_tools.cost_tracking import CostTracker

# Initialize in session state
if 'cost_tracker' not in st.session_state:
    st.session_state.cost_tracker = CostTracker()

# Display total cost
total_cost = st.session_state.cost_tracker.get_total_cost()
st.metric("Total Cost", f"${total_cost:.2f}")

# Display cost history
history = st.session_state.cost_tracker.get_cost_history()
# ... convert to dataframe and display ...
```

## Thread Safety

All database operations are thread-safe, making the system suitable for multi-user environments:

```python
import threading
from forecasting_tools.cost_tracking import CostTracker

tracker = CostTracker()

def worker_function(question_id):
    tracker.track_forecast(
        question_id=question_id,
        question_text=f"Question {question_id}",
        tokens_used=1000,
        model_name="gpt-4"
    )

# Create multiple threads
threads = []
for i in range(10):
    thread = threading.Thread(target=worker_function, args=(f"q{i}",))
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()
``` 