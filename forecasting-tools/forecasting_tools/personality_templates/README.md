# Personality Templates System

This system provides a flexible way to configure AI personalities and prompts for forecasting tasks. It allows for consistent prompt engineering across different forecasting bots while enabling easy customization of the AI's reasoning approach.

## Overview

The system consists of:

1. **Research Templates**: Different templates for research tasks with specialized versions for different research methods.
2. **Forecast Templates**: Templates for different question types (binary, multiple choice, numeric) and bot versions (Q1, Q2, Q3, Q4).
3. **Personality Traits**: Configurable aspects of personality like reasoning depth, uncertainty approach, expert persona, and thinking style.
4. **Personality Configurations**: Predefined combinations of traits for different forecasting personalities.

## Directory Structure

```
forecasting_tools/personality_templates/
├── research/
│   ├── default_research.txt
│   ├── perplexity_research.txt
│   └── smart_searcher_research.txt
├── forecasts/
│   ├── binary/
│   │   ├── q1_binary.txt
│   │   ├── q2_binary.txt 
│   │   ├── q3_binary.txt
│   │   └── q4_veritas_binary.txt
│   ├── multiple_choice/
│   │   ├── q1_multiple_choice.txt
│   │   └── q2_multiple_choice.txt
│   └── numeric/
│       ├── q1_numeric.txt
│       └── q2_numeric.txt
├── personality_traits/
│   ├── reasoning_depth.json
│   ├── uncertainty_approach.json
│   ├── expert_persona.json
│   └── thinking_style.json
└── personalities/
    ├── balanced.json
    ├── cautious.json
    ├── creative.json
    ├── economist.json
    └── bayesian.json
```

## Usage

### Basic Usage

```python
from forecasting_tools.personality_templates import TemplateManager, PersonalityConfig

# Initialize with desired bot version
template_manager = TemplateManager(bot_version="q2")

# Initialize with desired personality
personality_config = PersonalityConfig(personality_name="balanced")

# Get traits configuration
traits_config = personality_config.get_traits_config()

# Get research template
research_template = template_manager.get_research_template("perplexity_research")

# Apply personality traits
research_template = template_manager.apply_personality_traits(research_template, traits_config)

# Format with question-specific values
research_prompt = research_template.format(question="Will AI safety become a major political issue by 2025?")

# Get forecast template for a specific question type
forecast_template = template_manager.get_forecast_template(BinaryQuestion)

# Apply personality traits
forecast_template = template_manager.apply_personality_traits(forecast_template, traits_config)

# Format with question-specific values
forecast_prompt = forecast_template.format(
    question_text="Will AI safety become a major political issue by 2025?",
    background_info="...",
    resolution_criteria="...",
    fine_print="...",
    research="...",
    current_date="2023-06-15"
)

# Get thinking parameters
thinking_params = personality_config.get_thinking_parameters()
```

### Supported Personalities

- **balanced**: A balanced personality that prioritizes objectivity and careful consideration of evidence
- **cautious**: A cautious personality that prioritizes thorough analysis and avoids overconfidence
- **creative**: A creative personality that thinks outside the box and considers unconventional scenarios
- **economist**: An economist personality that specializes in economic analysis and market forecasting
- **bayesian**: A Bayesian personality that excels at updating beliefs based on new evidence

### Personality Traits

1. **Reasoning Depth**: Controls how deep the analysis goes
   - shallow: Quick, focused analysis
   - medium: Balanced depth
   - deep: Thorough, detailed analysis

2. **Uncertainty Approach**: Controls how uncertainty is handled
   - cautious: Conservative estimates, wider confidence intervals
   - balanced: Well-calibrated probabilities
   - confident: More precise estimates

3. **Expert Persona**: Defines the expert perspective
   - forecaster: Professional forecaster
   - statistician: Statistical analysis focus
   - economist: Economic analysis focus
   - political_analyst: Political analysis focus
   - scientist: Scientific method focus
   - security_analyst: Risk assessment focus

4. **Thinking Style**: Defines the reasoning approach
   - analytical: Structured, logical approach
   - creative: Lateral thinking, unusual connections
   - systematic: Methodical, comprehensive approach
   - bayesian: Bayesian updating focus
   - contrarian: Challenges conventional wisdom
   - intuitive: Balances intuition and analysis

## Extending the System

### Adding New Personality Traits

1. Create a new JSON file in `personality_traits/` directory
2. Define different levels of the trait and the corresponding instructions
3. Update templates to include the new trait placeholder

### Adding New Personalities

1. Create a new JSON file in `personalities/` directory
2. Define the combination of traits and thinking parameters
3. Use existing personalities as a reference

### Adding New Templates

1. Create a new text file in the appropriate directory
2. Include placeholders for personality traits with `{trait_name_instructions}`
3. Include placeholders for question-specific values with `{value_name}` 