# Personality Customization Tutorial

This tutorial guides you through the process of creating and customizing personalities for the forecasting tools system. You'll learn how to create personalities optimized for different forecasting scenarios, domains, and use cases.

## Table of Contents

- [Introduction](#introduction)
- [Basic Personality Creation](#basic-personality-creation)
- [Domain-Specific Personalities](#domain-specific-personalities)
- [Advanced Traits and Customization](#advanced-traits-and-customization)
- [Personality Ensembles](#personality-ensembles)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Introduction

Personalities in the forecasting tools system control how forecasting bots approach questions, handle uncertainty, and reason about complex problems. By customizing personalities, you can:

- Create forecasters specialized for different domains (finance, science, politics, etc.)
- Optimize for different forecasting tasks (short-term predictions, long-range forecasts, etc.)
- Generate diverse perspectives on the same question
- Balance between speed, thoroughness, and resource usage

## Basic Personality Creation

Let's start by creating a simple custom personality.

### Step 1: Create a Basic Configuration File

Create a file named `my_first_personality.json` with this basic structure:

```json
{
  "name": "my_first_personality",
  "description": "My first custom forecasting personality",
  "thinking_style": "analytical",
  "uncertainty_approach": "balanced",
  "reasoning_depth": "moderate",
  "temperature": 0.6
}
```

### Step 2: Choose Appropriate Traits

Select the appropriate values for the core personality traits:

- **thinking_style**: How the forecaster approaches reasoning
  - `analytical`: Systematic, logical approach with structured analysis
  - `creative`: Explores unusual possibilities and novel perspectives
  - `bayesian`: Focuses on probability updates based on evidence
  - `balanced`: Equal mix of analytical and creative approaches

- **uncertainty_approach**: How the forecaster handles uncertainty
  - `cautious`: More hedging, wider confidence intervals
  - `balanced`: Moderate approach to uncertainty
  - `bold`: Narrower confidence intervals, more decisive

- **reasoning_depth**: How thorough the reasoning process should be
  - `shallow`: Quick, high-level analysis
  - `moderate`: Standard level of detail
  - `deep`: Thorough, detailed analysis
  - `exhaustive`: Extremely comprehensive analysis

- **temperature**: Controls randomness in language generation (0.0 to 1.0)
  - Lower values (0.1-0.4): More deterministic, consistent outputs
  - Medium values (0.5-0.7): Balanced creativity and consistency
  - Higher values (0.8-1.0): More creative, diverse outputs

### Step 3: Save and Test Your Personality

Save your file in one of these directories:
- System-wide: `/etc/forecasting-tools/personalities/`
- User-specific: `~/.forecasting-tools/personalities/`
- Project-specific: `./personalities/`

Test your personality with this code:

```python
from forecasting_tools import ForecastingBot
from forecasting_tools.data_models.questions import BinaryQuestion

# Create a test question
question = BinaryQuestion(
    question_text="Will the S&P 500 index be higher one month from now?",
    background_info="Current market conditions are mixed, with inflation concerns but strong earnings.",
    resolution_criteria="Based on closing price comparison.",
    fine_print=""
)

# Create a bot with your custom personality
bot = ForecastingBot(personality_name="my_first_personality")

# Generate forecast
forecast = bot.forecast_binary(question)

# Print results
print(f"Probability: {forecast.binary_prob}")
print(f"Reasoning:\n{forecast.reasoning}")
```

## Domain-Specific Personalities

Now, let's create a personality specialized for a specific domain.

### Financial Expert Personality

Create a file named `finance_expert.json`:

```json
{
  "name": "finance_expert",
  "description": "Expert in financial forecasting with quantitative focus",
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
    "extra_instructions": "Pay special attention to financial indicators, market trends, and economic data. Consider both technical and fundamental analysis when relevant.",
    "special_knowledge": "You have extensive knowledge of financial markets, economic theory, and investment principles."
  }
}
```

### Political Analyst Personality

Create a file named `political_analyst.json`:

```json
{
  "name": "political_analyst",
  "description": "Expert in political analysis and geopolitical forecasting",
  "thinking_style": "bayesian",
  "uncertainty_approach": "cautious",
  "reasoning_depth": "deep",
  "temperature": 0.5,
  "traits": {
    "domain_expertise": {
      "name": "domain_expertise",
      "description": "Level of expertise in politics",
      "value": 0.9
    },
    "historical_awareness": {
      "name": "historical_awareness",
      "description": "Awareness of historical precedents",
      "value": 0.8
    }
  },
  "template_variables": {
    "extra_instructions": "Consider historical precedents, institutional constraints, and stakeholder incentives. Pay attention to both public statements and revealed preferences of key actors.",
    "special_knowledge": "You have extensive knowledge of political systems, international relations, and historical political events."
  }
}
```

## Advanced Traits and Customization

Let's create a more complex personality with custom traits.

### Step 1: Define Custom Traits

Custom traits are additional characteristics that further refine a personality. 

```json
{
  "name": "advanced_personality",
  "description": "Advanced personality with custom traits",
  "thinking_style": "creative",
  "uncertainty_approach": "bold",
  "reasoning_depth": "moderate",
  "temperature": 0.7,
  "traits": {
    "creativity": {
      "name": "creativity",
      "description": "Level of creative thinking",
      "value": 0.9
    },
    "contrarian_tendency": {
      "name": "contrarian_tendency",
      "description": "Tendency to challenge mainstream views",
      "value": 0.7
    },
    "strategic_depth": {
      "name": "strategic_depth",
      "description": "Focus on long-term strategic implications",
      "value": 0.8
    },
    "multimodal_reasoning": {
      "name": "multimodal_reasoning",
      "description": "Ability to integrate different types of evidence",
      "value": 0.6
    }
  },
  "template_variables": {
    "extra_instructions": "Look for non-obvious connections and challenge conventional wisdom when appropriate. Consider strategic implications and second-order effects.",
    "reasoning_approach": "You should consider multiple perspectives and models, even unconventional ones, when analyzing this question."
  }
}
```

### Step 2: Use Template Variables

Template variables allow you to inject personality-specific content into prompt templates. These can be used to:

- Add domain knowledge
- Specify reasoning approaches
- Provide extra instructions
- Control output format

### Step 3: Validate Your Personality

Use the validation tools to check your personality:

```python
from forecasting_tools.personality_management.validators import PersonalityValidator

validator = PersonalityValidator()
is_valid, errors = validator.check_file_integrity("./personalities/advanced_personality.json")

if not is_valid:
    print("Validation errors:")
    for error in errors:
        print(f"- {error}")
else:
    print("Personality is valid!")
```

## Personality Ensembles

Combining multiple personalities can provide more robust forecasts. Let's create a script to generate an ensemble forecast:

```python
from forecasting_tools import ForecastingBot
from forecasting_tools.data_models.questions import BinaryQuestion
import numpy as np

# Create a test question
question = BinaryQuestion(
    question_text="Will the EU implement new carbon regulations by the end of next year?",
    background_info="The EU has been discussing carbon policy reform for several years.",
    resolution_criteria="Official announcement of new carbon regulations.",
    fine_print=""
)

# Define an ensemble of personalities
personalities = [
    "analytical",        # Systematic approach
    "creative",          # Novel perspectives
    "bayesian",          # Probability-focused
    "finance_expert",    # Domain expert (economic impact)
    "political_analyst"  # Domain expert (political process)
]

# Generate forecasts from each personality
forecasts = {}
reasonings = {}

for personality_name in personalities:
    bot = ForecastingBot(personality_name=personality_name)
    forecast = bot.forecast_binary(question)
    
    forecasts[personality_name] = forecast.binary_prob
    reasonings[personality_name] = forecast.reasoning

# Aggregate the forecasts (simple average)
average_prob = np.mean(list(forecasts.values()))

# Print results
print(f"Ensemble forecast probability: {average_prob:.2f}")
print("\nIndividual forecasts:")
for name, prob in forecasts.items():
    print(f"- {name}: {prob:.2f}")

# Create a meta-analysis using the bayesian personality
meta_analysis_bot = ForecastingBot(personality_name="bayesian")
meta_analysis = meta_analysis_bot.analyze_forecasts(forecasts, reasonings, question)
print(f"\nMeta-analysis:\n{meta_analysis}")
```

## Best Practices

Follow these best practices when creating custom personalities:

### 1. Match Personality to Question Type

- **Analytical** personalities work best for:
  - Financial/economic questions
  - Questions with quantifiable metrics
  - Technical or scientific forecasts

- **Creative** personalities work best for:
  - Novel scenarios without precedent
  - Questions requiring outside-the-box thinking
  - Technology adoption and innovation questions

- **Bayesian** personalities work best for:
  - Questions with established prior probabilities
  - Iterative forecasting situations
  - Questions with new information updating previous beliefs

### 2. Adjust Reasoning Depth Based on Complexity

- Use **shallow** reasoning for:
  - Simple, straightforward questions
  - Time-sensitive forecasts
  - Initial screening of many questions

- Use **deep** or **exhaustive** reasoning for:
  - Complex, multi-faceted questions
  - Critical decisions with high stakes
  - Questions requiring thorough analysis of many factors

### 3. Balance Trait Values

Avoid extreme combinations that create internal inconsistencies:

- Don't combine `analytical` thinking with very high creativity traits
- Don't set `cautious` uncertainty approach with bold traits
- Don't use `shallow` reasoning depth with high thoroughness traits

### 4. Test and Iterate

Always test your personalities on multiple questions and refine based on results:

```python
from forecasting_tools.personality_management.debugger import get_debugger

debugger = get_debugger()

# Test your personality on different templates
results = debugger.test_template_rendering(
    personality_name="finance_expert",
    template_name="forecast_template",
    variables={"question": "Will inflation exceed 3% next year?"}
)

# Check if successful
if results["success"]:
    print("Template rendered successfully!")
    print(f"Content snippet: {results['rendered_content'][:200]}...")
else:
    print(f"Error: {results['error']}")
```

## Troubleshooting

### Common Issues and Solutions

1. **Personality Not Found**

   **Error**: `Personality "my_personality" not found`
   
   **Solution**: Verify the file is in the correct directory and named correctly. The filename (without `.json`) should match the `name` field in the configuration.

2. **Invalid Configuration**

   **Error**: `Invalid thinking_style: creative_analytical`
   
   **Solution**: Use only the approved values for each field:
   - `thinking_style`: analytical, creative, balanced, bayesian
   - `uncertainty_approach`: cautious, balanced, bold
   - `reasoning_depth`: shallow, moderate, deep, exhaustive

3. **Template Compatibility Issues**

   **Error**: `Personality missing template variables: reasoning_style`
   
   **Solution**: Check which variables your template requires and provide them in your personality's `template_variables` section.

4. **Performance Issues**

   **Problem**: Generating forecasts is too slow
   
   **Solution**: Use a shallower reasoning depth or adjust traits to be less exhaustive. Enable caching for repeated usage.

### Diagnostic Tools

Use these diagnostic tools to troubleshoot issues:

```python
from forecasting_tools.personality_management.debugger import get_debugger

debugger = get_debugger()

# Diagnose issues with a personality
diagnosis = debugger.diagnose_personality("problematic_personality")
print("Issues found:")
for issue in diagnosis["issues"]:
    print(f"- {issue}")

# Check template compatibility
for template_name, result in diagnosis["template_compatibility"].items():
    if not result["compatible"]:
        print(f"Incompatible with template {template_name}:")
        for issue in result["issues"]:
            print(f"  - {issue}")
```

## Conclusion

You now have the knowledge to create and customize personalities for different forecasting scenarios. Experiment with different combinations of traits to find what works best for your specific needs.

For more advanced functionality, review the [API Reference](../api/personality_api.md) and explore the [Example Notebooks](../examples/personality_examples.ipynb). 