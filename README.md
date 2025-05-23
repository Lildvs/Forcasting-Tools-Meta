# Forecasting Tools

A comprehensive toolkit for generating, evaluating, and managing forecasts with a focus on accuracy, scalability, and reliability.

## Overview

Forecasting Tools provides a powerful platform for creating and managing probabilistic forecasts. The system is designed to help analysts, researchers, and decision-makers generate well-calibrated predictions supported by structured reasoning and evidence.

Key features include:
- Asynchronous processing of forecast requests
- Persistent storage of forecasts and associated data
- Rate limiting and batching for external API calls
- Comprehensive logging and monitoring
- Evaluation tools for forecast quality assessment
- A/B testing capabilities to compare different forecasting approaches

## Installation

### Prerequisites
- Python 3.9+
- SQLite or other database for persistent storage

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/forecasting-tools.git
cd forecasting-tools

# Install the package and dependencies
pip install -e ".[dev]"
```

## Quick Start

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

## Architecture

The system is built with a modular, component-based architecture:

### Core Components

- **Forecasting Pipeline**: Central component for processing forecast requests
- **Storage Layer**: Unified interface for data persistence 
- **Queue System**: Priority-based job queue for asynchronous processing
- **Monitoring System**: Tracks system performance, API usage, and error rates

### Supporting Components

- **Async Helpers**: Utilities for asynchronous operations
- **Configuration System**: Environment-specific settings
- **Evaluation Framework**: Tools for assessing forecast quality

For more details, see the [Architecture Documentation](docs/architecture/system_overview.md).

## Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=forecasting_tools
```

### Project Structure

```
forecasting-tools/
├── forecasting_tools/       # Main package
│   ├── api/                 # Public API interface
│   ├── data/                # Storage and database models
│   ├── evaluation/          # Forecast evaluation tools
│   └── utils/               # Utility modules
├── tests/                   # Test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── fixtures/            # Test data
├── docs/                    # Documentation
└── .github/                 # CI/CD configuration
```

## Testing

The project includes comprehensive test suites:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# Run with coverage report
pytest --cov=forecasting_tools --cov-report=html
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Documentation

Full documentation is available in the `docs/` directory:

- [System Architecture](docs/architecture/system_overview.md)
- [API Reference](docs/api/api_reference.md)
- [Usage Examples](docs/examples/basic_usage.md)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project builds on research in forecasting, decision science, and large language models
- Inspired by [Forecasting Research Institute](https://forecastingresearch.org/) and [Good Judgment Project](https://goodjudgment.com/) 