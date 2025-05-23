# Personality Management System

The Personality Management System provides a structured way to customize and control how forecasting bots approach questions, express uncertainty, and reason about complex problems. This guide provides a comprehensive overview of the system's architecture, components, and best practices for usage.

## Table of Contents

- [Introduction](#introduction)
- [Core Concepts](#core-concepts)
- [Installation and Setup](#installation-and-setup)
- [Basic Usage](#basic-usage)
- [Advanced Configuration](#advanced-configuration)
- [Architecture](#architecture)
- [Integration Points](#integration-points)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Introduction

The Personality Management System allows you to define, customize, and apply different "personalities" to forecasting bots. Each personality influences how a bot approaches forecasting tasks, including its:

- Thinking style (analytical, creative, Bayesian, or balanced)
- Approach to uncertainty (cautious, balanced, or bold)
- Depth of reasoning (shallow, moderate, deep, or exhaustive)
- Custom traits and characteristics

By using personalities, you can:

1. **Improve forecast quality** by matching personality characteristics to question types
2. **Increase diversity** of forecasts through multiple different perspectives
3. **Customize forecasting approach** for different domains or users
4. **Optimize resource usage** by selecting appropriate reasoning depths

## Core Concepts

The personality system is built around these key concepts:

### Personality Configuration

A `PersonalityConfig` defines the characteristics of a forecasting personality, including:

- `thinking_style`: How the forecaster approaches reasoning (analytical, creative, etc.)
- `uncertainty_approach`: How the forecaster handles uncertainty (cautious, bold, etc.)
- `reasoning_depth`: How thorough the reasoning process should be (shallow to exhaustive)
- `traits`: Custom attributes that further refine the personality
- `template_variables`: Specific variables used in prompt templates

### Templates

Templates define the structure of prompts used to instruct language models. The template system is:

- **Flexible**: Supports variables, conditional sections, and composition
- **Efficient**: Implements lazy loading and caching for performance
- **Extensible**: Allows for custom templates and inheritance

### Prompt Generation Pipeline

The prompt generation pipeline optimizes the creation of prompts by:

- Efficiently combining personalities with templates
- Applying caching to reduce redundant operations
- Supporting compression and token optimization
- Providing monitoring of performance characteristics

## Installation and Setup

### Prerequisites

- Python 3.10 or higher
- forecasting-tools package

### Installation

If you're using the full forecasting-tools package, the personality system is included. Otherwise, install it separately:

```bash
pip install forecasting-tools[personality]
```

### Configuration

1. Configure the system by setting environment variables:

```bash
# Enable/disable the entire system
export FORECASTING_PERSONALITY_PERSONALITY_SYSTEM_ENABLED=true

# Control caching behavior
export FORECASTING_PERSONALITY_USE_CACHING=true

# Set rollout percentage for gradual deployment
export FORECASTING_PERSONALITY_ROLLOUT_PERCENTAGE=50
```

2. Or create a configuration file at `~/.forecasting-tools/personality_flags.json`:

```json
{
  "feature_flags": {
    "personality_system_enabled": true,
    "use_caching": true,
    "rollout_percentage": 100
  }
}
```

## Basic Usage

### Loading a Personality

```python
from forecasting_tools.personality_management import PersonalityManager

# Initialize the manager
manager = PersonalityManager()

# List available personalities
personalities = manager.list_available_personalities()
print(f"Available personalities: {personalities}")

# Load a specific personality
analytical_personality = manager.load_personality("analytical")
```

### Creating a Forecast with a Personality

```python
from forecasting_tools import ForecastingBot
from forecasting_tools.data_models.questions import BinaryQuestion

# Create a question
question = BinaryQuestion(
    question_text="Will GDP growth exceed 2% next year?",
    background_info="Current growth is 1.5% with mixed economic indicators.",
    resolution_criteria="Based on official government statistics.",
    fine_print=""
)

# Create a forecasting bot with a personality
bot = ForecastingBot(personality_name="analytical")

# Generate forecast
forecast = bot.forecast_binary(question)

# Access the forecast
print(f"Probability: {forecast.binary_prob}")
print(f"Reasoning:\n{forecast.reasoning}")
```

### Using Multiple Personalities

```python
# Create an ensemble of forecasts with different personalities
personalities = ["analytical", "creative", "bayesian", "cautious"]
forecasts = {}

for personality_name in personalities:
    bot = ForecastingBot(personality_name=personality_name)
    forecasts[personality_name] = bot.forecast_binary(question)

# Compare forecasts
for name, forecast in forecasts.items():
    print(f"{name}: {forecast.binary_prob}")
```

## Advanced Configuration

### Creating Custom Personalities

1. Create a JSON configuration file:

```json
{
  "name": "finance_expert",
  "description": "Expert in financial forecasting with strong quantitative skills",
  "thinking_style": "analytical",
  "uncertainty_approach": "balanced",
  "reasoning_depth": "deep",
  "temperature": 0.4,
  "traits": {
    "domain_expertise": {
      "name": "domain_expertise",
      "description": "Level of expertise in finance",
      "value": 0.9
    },
    "quantitative_focus": {
      "name": "quantitative_focus",
      "description": "Focus on quantitative analysis",
      "value": 0.8
    }
  },
  "template_variables": {
    "extra_instructions": "Pay special attention to financial indicators and economic trends.",
    "special_knowledge": "You have extensive knowledge of financial markets and economic theory."
  }
}
```

2. Save this file in one of the personality directories:
   - System-wide: `/etc/forecasting-tools/personalities/`
   - User-specific: `~/.forecasting-tools/personalities/`
   - Project-specific: `./personalities/`

3. Use your custom personality:

```python
bot = ForecastingBot(personality_name="finance_expert")
```

### Creating Custom Templates

1. Create a template file:

```json
{
  "content": "You are a forecaster with a {{thinking_style}} thinking style and a {{uncertainty_approach}} approach to uncertainty. {{special_knowledge}}\n\nQuestion: {{question}}\n\n{{extra_instructions}}",
  "variables": {
    "thinking_style": "balanced",
    "uncertainty_approach": "balanced",
    "special_knowledge": "",
    "extra_instructions": ""
  }
}
```

2. Save this file in a template directory:
   - System templates: `/etc/forecasting-tools/templates/`
   - User templates: `~/.forecasting-tools/templates/`
   - Project templates: `./templates/`

3. Reference the template in your code:

```python
from forecasting_tools.personality_management.prompt_optimizer import PromptOptimizer

optimizer = PromptOptimizer()
prompt, metadata = optimizer.optimize_prompt_pipeline(
    personality_name="finance_expert",
    template_name="your_custom_template",
    variables={"question": "Will inflation exceed 3% next year?"}
)
```

## Architecture

The personality system consists of these key components:

### Core Components

- **PersonalityManager**: Central manager that loads, validates, and provides personalities
- **PersonalityConfig**: Data structure defining personality characteristics
- **TemplateManager**: Handles the loading and rendering of templates
- **PromptOptimizer**: Optimizes prompt generation for performance

### Performance Optimization

- **PersonalityCache**: Caches personality configurations to avoid repeated loading
- **Lazy Loading**: Templates are loaded only when needed
- **Prompt Cache**: Generated prompts are cached based on inputs

### Monitoring and Diagnostics

- **Telemetry**: Collects usage patterns and performance metrics
- **Validators**: Ensures personality configurations are valid and compatible
- **Debugger**: Provides tools for diagnosing and resolving issues

### Feature Flags

- Enables gradual rollout and A/B testing
- Controls specific features of the personality system
- Allows for user- or question-specific configurations

## Integration Points

The personality system integrates with other components in various ways:

### ForecastingBot Integration

The `ForecastingBot` class accepts a `personality_name` parameter that loads and applies the specified personality.

```python
bot = ForecastingBot(personality_name="analytical")
```

### LLM Provider Integration

The personality system works with different LLM providers by:

1. Generating appropriate prompts based on personality
2. Setting temperature and other parameters
3. Optimizing context usage for different models

### Custom Integrations

You can integrate the personality system with custom code:

```python
from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.prompt_optimizer import PromptOptimizer

# Load a personality
manager = PersonalityManager()
personality = manager.load_personality("analytical")

# Generate a prompt
optimizer = PromptOptimizer()
prompt, metadata = optimizer.generate_prompt(
    prompt_template="Your template content with {{variables}}",
    personality_name="analytical",
    variables={"key": "value"}
)

# Use the generated prompt with your own LLM call
# ...your custom LLM call code...
```

## Performance Considerations

The personality system is optimized for performance, but consider these guidelines:

### Caching Strategy

- Use the built-in caching for repeated personality usage
- For server environments, consider preloading common personalities

```python
# Preload common personalities at application startup
from forecasting_tools.personality_management import PersonalityManager
manager = PersonalityManager()
for name in ["analytical", "creative", "balanced", "cautious"]:
    manager.load_personality(name)
```

### Template Optimization

- Keep templates as small as possible
- Minimize the number of variables and conditional sections
- Use template combining for complex scenarios

### Memory Management

- Set appropriate cache sizes for your environment
- For long-running services, periodically clear caches

```python
from forecasting_tools.personality_management.cache import PersonalityCache
from forecasting_tools.personality_management.prompt_optimizer import PromptOptimizer

# Clear caches periodically
PersonalityCache().invalidate_all()
PromptOptimizer().clear_cache()
```

### Reasoning Depth Selection

Match reasoning depth to question complexity:
- Use `shallow` for simple, time-sensitive forecasts
- Use `moderate` for general-purpose forecasting
- Use `deep` for complex or critical forecasts
- Use `exhaustive` only when thorough analysis is essential

## Troubleshooting

### Common Issues

1. **Personality Not Found**

```
Error: Personality "xyz" not found
```

**Solution**: Check that the personality file exists in one of the personality directories and has the correct name.

2. **Template Compatibility Issues**

```
Warning: Personality missing template variables: special_instruction
```

**Solution**: Ensure your personality provides all variables required by the template.

3. **Performance Issues**

**Solution**: Enable caching, optimize templates, and select appropriate reasoning depth.

### Diagnostic Tools

Use the built-in debugging tools to diagnose issues:

```python
from forecasting_tools.personality_management.debugger import get_debugger

debugger = get_debugger()

# Diagnose a personality
diagnosis = debugger.diagnose_personality("problematic_personality")
print(diagnosis["issues"])

# Test template rendering
result = debugger.test_template_rendering(
    personality_name="analytical",
    template_name="forecast_template",
    variables={"question": "Will GDP grow?"}
)

# Generate a complete debug report
report = debugger.generate_debug_report("debug_report.json")
```

### Logging

Enable verbose logging for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("forecasting_tools.personality_management").setLevel(logging.DEBUG)
```

Or use the feature flag:

```python
from forecasting_tools.personality_management.feature_flags import get_feature_flags

flags = get_feature_flags()
flags.update_flag("verbose_logging", True)
```

## API Reference

For detailed API documentation, see the [Personality API Reference](api/personality_api.md).

## Further Resources

- [Performance Characteristics Documentation](personality_management/performance.md)
- [Testing Custom Personalities](personality_management/testing.md)
- [Personality Customization Tutorials](tutorials/personality_customization.md)
- [Example Notebooks](examples/personality_examples.ipynb) 