# Personality Management System

This module provides a centralized system for managing different AI "personalities" in the forecasting tools. This allows for controlling the reasoning approach, thought process, and prompt engineering across the application from a single location.

## Overview

The personality management system consists of:

1. **Personality Configurations**: YAML files that define different forecasting personalities with traits, thinking configurations, and prompt-specific values
2. **Prompt Templates**: Template files that contain the basic structure of prompts used throughout the application
3. **Personality Manager**: A class that loads personalities and templates, and provides methods to retrieve and apply prompts

## Usage

### Basic Usage

```python
from forecasting_tools.personality_management import PersonalityManager

# Create a personality manager with a specific personality
personality = PersonalityManager(personality_name="creative")

# Get a prompt with personality-specific values injected
research_prompt = personality.get_prompt("research_prompt")

# Get a prompt with additional parameters
forecast_prompt = personality.get_prompt(
    "binary_forecast_prompt",
    question_text="Will X happen by Y date?",
    current_date="2025-01-01"
)

# Get thinking configuration for the LLM
thinking_config = personality.get_thinking_config()
```

### Available Personalities

- **balanced**: A balanced personality that prioritizes objectivity and careful consideration of evidence
- **cautious**: A cautious personality that prioritizes thorough analysis and avoids overconfidence
- **creative**: A creative personality that thinks outside the box and considers unconventional scenarios

### Command Line Usage

When running the bot via the command line, you can specify a personality:

```bash
python run_bot.py --personality creative
```

## Extending the System

### Adding New Personalities

1. Create a new YAML file in `forecasting_tools/personality_management/personalities/`
2. Follow the structure of existing personality files
3. Customize traits, thinking configuration, and prompt-specific values

### Adding New Prompt Templates

1. Create a new text file in `forecasting_tools/personality_management/templates/`
2. Use placeholder variables with the format `{variable_name}`
3. Update personality YAML files to include values for these variables

## Personality Configuration Structure

```yaml
# Basic information
name: PersonalityName
description: Description of the personality

# Personality traits
traits:
  risk_tolerance: 0.5  # 0-1 scale
  creativity: 0.5      # 0-1 scale
  thoroughness: 0.7    # 0-1 scale
  # Add more traits as needed

# Thinking configuration
thinking:
  type: "enabled"
  budget_tokens: 32000
  reasoning_framework: "balanced"

# Prompt-specific values
prompts:
  research_prompt:
    perspective: "objective and analytical"
    focus_areas: "key facts and context"
    # More prompt-specific values
``` 