# Personality Selection Guide by Question Type

This guide helps you select the most appropriate personality for different types of forecasting questions. Choosing the right personality can significantly improve forecast quality and relevance.

## Table of Contents

- [Understanding Personality Traits](#understanding-personality-traits)
- [Question Categorization](#question-categorization)
- [Selection Matrix](#selection-matrix)
- [Domain-Specific Recommendations](#domain-specific-recommendations)
- [Multi-Personality Approaches](#multi-personality-approaches)
- [Customization Guidelines](#customization-guidelines)
- [Performance Testing](#performance-testing)

## Understanding Personality Traits

Before selecting a personality, it's important to understand the key traits that define them:

### Thinking Styles

| Style | Description | Best For |
|-------|-------------|----------|
| **Analytical** | Systematic, logical approach with structured analysis | Technical questions, data-driven forecasts |
| **Creative** | Explores unusual possibilities and novel perspectives | Emerging trends, paradigm shifts |
| **Balanced** | Equal mix of analytical and creative approaches | General-purpose forecasting |
| **Bayesian** | Focuses on probability updates based on evidence | Questions with prior data, iterative forecasting |

### Uncertainty Approaches

| Approach | Description | Best For |
|----------|-------------|----------|
| **Cautious** | More hedging, wider confidence intervals | High-stakes decisions, asymmetric risks |
| **Balanced** | Moderate approach to uncertainty | General-purpose forecasting |
| **Bold** | Narrower confidence intervals, more decisive | Short-term forecasts, competitive scenarios |

### Reasoning Depths

| Depth | Description | Best For |
|-------|-------------|----------|
| **Shallow** | Quick, high-level analysis | Time-sensitive forecasts, simple questions |
| **Moderate** | Standard level of detail | General-purpose forecasting |
| **Deep** | Thorough, detailed analysis | Complex questions, important decisions |
| **Exhaustive** | Extremely comprehensive analysis | Critical forecasts, multi-faceted problems |

## Question Categorization

Forecasting questions can be categorized along several dimensions:

### Time Horizon

- **Short-term** (days to weeks)
- **Medium-term** (months to a year)
- **Long-term** (years to decades)

### Uncertainty Level

- **Low uncertainty** (established domains, clear rules)
- **Medium uncertainty** (some unknowns, but precedents exist)
- **High uncertainty** (novel situations, many unknowns)

### Complexity

- **Simple** (few variables, clear mechanisms)
- **Moderate** (several variables, some interactions)
- **Complex** (many variables, complex interactions)
- **Wicked** (complex with changing conditions and requirements)

### Question Type

- **Binary** (yes/no questions)
- **Numeric** (point estimates or ranges)
- **Distribution** (full probability distribution)
- **Ranking** (relative ordering)

## Selection Matrix

Use this matrix to help select the appropriate personality traits for different question categories:

| Question Category | Thinking Style | Uncertainty Approach | Reasoning Depth |
|-------------------|----------------|----------------------|-----------------|
| **Short-term, Low uncertainty** | Analytical | Bold | Shallow to Moderate |
| **Short-term, High uncertainty** | Creative | Cautious | Moderate |
| **Long-term, Low uncertainty** | Analytical | Balanced | Deep |
| **Long-term, High uncertainty** | Creative or Bayesian | Cautious | Deep to Exhaustive |
| **Simple, Binary** | Analytical | Bold | Shallow to Moderate |
| **Complex, Binary** | Balanced | Balanced | Deep |
| **Numeric Estimation** | Analytical | Balanced | Moderate to Deep |
| **Novel Domains** | Creative | Cautious | Moderate to Deep |
| **Rapidly Changing** | Creative | Balanced | Moderate |
| **Data-Rich** | Analytical | Bold | Deep |
| **Data-Poor** | Bayesian | Cautious | Moderate |
| **High Stakes** | Balanced | Cautious | Exhaustive |
| **Competitive** | Creative | Bold | Deep |

## Domain-Specific Recommendations

Different domains often benefit from specific personality configurations:

### Financial Markets

**Short-term market movements:**
- Thinking Style: Analytical
- Uncertainty Approach: Bold
- Reasoning Depth: Moderate
- Custom Traits: High quantitative_focus, moderate contrarian_tendency

**Long-term market trends:**
- Thinking Style: Balanced
- Uncertainty Approach: Balanced
- Reasoning Depth: Deep
- Custom Traits: High historical_awareness, moderate domain_expertise

### Technology Forecasting

**Near-term product adoption:**
- Thinking Style: Analytical
- Uncertainty Approach: Balanced
- Reasoning Depth: Moderate
- Custom Traits: High market_awareness, high user_empathy

**Long-term technology evolution:**
- Thinking Style: Creative
- Uncertainty Approach: Cautious
- Reasoning Depth: Deep
- Custom Traits: High innovation_awareness, high strategic_vision

### Political Forecasting

**Election outcomes:**
- Thinking Style: Bayesian
- Uncertainty Approach: Balanced
- Reasoning Depth: Deep
- Custom Traits: High polling_literacy, high historical_awareness

**Geopolitical developments:**
- Thinking Style: Balanced
- Uncertainty Approach: Cautious
- Reasoning Depth: Deep
- Custom Traits: High strategic_thinking, high historical_awareness

### Scientific Progress

**Research breakthroughs:**
- Thinking Style: Creative
- Uncertainty Approach: Cautious
- Reasoning Depth: Deep
- Custom Traits: High domain_expertise, high innovation_awareness

**Technology timelines:**
- Thinking Style: Bayesian
- Uncertainty Approach: Cautious
- Reasoning Depth: Deep
- Custom Traits: High scaling_awareness, high development_process_knowledge

## Multi-Personality Approaches

For important questions, consider using multiple personalities and aggregating their forecasts:

### Ensemble Types

1. **Diverse Panel**: Use personalities with different thinking styles to generate diverse perspectives
   - Example: analytical + creative + bayesian
   - Best for: Novel questions without clear precedents

2. **Depth Variation**: Use same personality with different reasoning depths
   - Example: analytical-shallow + analytical-deep
   - Best for: Time-sensitive questions where both quick takes and deep analysis are valuable

3. **Uncertainty Spectrum**: Use same personality with different uncertainty approaches
   - Example: balanced-cautious + balanced-bold
   - Best for: High-stakes questions where understanding the range of confidence is important

4. **Domain Expert Panel**: Use domain-specific personalities relevant to different aspects of the question
   - Example: finance_expert + political_analyst + tech_expert
   - Best for: Cross-domain questions that require multiple types of expertise

### Aggregation Methods

- **Simple Average**: Equal weighting of all forecasts
- **Weighted Average**: Weight personalities based on past performance in similar questions
- **Bayesian Aggregation**: Use a Bayesian personality to integrate multiple forecasts
- **Extremization**: Adjust the aggregate toward the most confident predictions
- **Trimmed Mean**: Remove outliers before averaging

## Customization Guidelines

When customizing personalities for specific use cases:

### 1. Start with a Base Personality

Choose the closest built-in personality to your needs:
- `analytical` - For data-driven, methodical forecasting
- `creative` - For innovation, emerging trends, and novel scenarios
- `balanced` - For general-purpose forecasting
- `bayesian` - For evidence-based, probabilistic reasoning

### 2. Adjust Core Traits

Modify the core traits based on your specific forecasting needs:
- Increase `reasoning_depth` for higher stakes questions
- Adjust `uncertainty_approach` based on the risk profile
- Change `temperature` to control randomness (lower for more consistency)

### 3. Add Domain-Specific Traits

Add custom traits that are relevant to your domain:
- `domain_expertise`: Level of knowledge in the specific field
- `quantitative_focus`: Emphasis on numerical analysis
- `historical_awareness`: Consideration of historical precedents
- `contrarian_tendency`: Willingness to challenge common views
- `innovation_awareness`: Sensitivity to emerging trends

### 4. Customize Template Variables

Add domain-specific instructions via template variables:
```json
"template_variables": {
  "extra_instructions": "Consider regulatory constraints and market sentiment in financial forecasts.",
  "relevant_data_sources": ["economic indicators", "central bank statements", "market technicals"]
}
```

## Performance Testing

To identify the best personality for a specific type of question:

### 1. Benchmark Testing

Test multiple personalities on questions with known outcomes:

```python
from forecasting_tools import ForecastingBot
from forecasting_tools.data_models.questions import BinaryQuestion

# Define test questions with known outcomes
test_questions = [
    {
        "question": "Will unemployment rise in Q3 2022?",
        "background": "...",
        "actual_outcome": False
    },
    # Add more test questions...
]

# Personalities to test
personalities = ["analytical", "creative", "bayesian", "my_custom_personality"]

# Run benchmark
results = {}
for personality in personalities:
    bot = ForecastingBot(personality_name=personality)
    correct_count = 0
    
    for q in test_questions:
        question = BinaryQuestion(
            question_text=q["question"],
            background_info=q["background"],
            resolution_criteria="",
            fine_print=""
        )
        
        forecast = bot.forecast_binary(question)
        prediction = forecast.binary_prob > 0.5
        
        if prediction == q["actual_outcome"]:
            correct_count += 1
    
    accuracy = correct_count / len(test_questions)
    results[personality] = accuracy

print(results)
```

### 2. Calibration Analysis

Analyze how well calibrated different personalities are:

```python
import numpy as np
import matplotlib.pyplot as plt

# Collect predictions and outcomes
predictions = []  # Probability forecasts
outcomes = []     # Actual boolean outcomes (True/False)

# Calculate calibration curve
def calibration_curve(predictions, outcomes, bins=10):
    bin_edges = np.linspace(0, 1, bins+1)
    bin_indices = np.digitize(predictions, bin_edges[:-1])
    
    bin_sums = np.zeros(bins)
    bin_counts = np.zeros(bins)
    
    for i, outcome in enumerate(outcomes):
        bin_idx = min(bin_indices[i] - 1, bins - 1)
        bin_counts[bin_idx] += 1
        if outcome:
            bin_sums[bin_idx] += 1
    
    bin_props = np.zeros(bins)
    for i in range(bins):
        if bin_counts[i] > 0:
            bin_props[i] = bin_sums[i] / bin_counts[i]
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, bin_props

# Plot calibration curves for different personalities
plt.figure(figsize=(10, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

for personality in personalities:
    # Get predictions and outcomes for this personality
    # ...
    
    bin_centers, bin_props = calibration_curve(predictions, outcomes)
    plt.plot(bin_centers, bin_props, 'o-', label=personality)

plt.xlabel('Predicted probability')
plt.ylabel('Actual frequency')
plt.title('Calibration Curve by Personality')
plt.legend()
plt.grid(True)
plt.show()
```

### 3. Comparative Analysis

Compare different aspects of forecast quality:

- **Accuracy**: Percentage of correct binary predictions
- **Calibration**: Alignment between confidence and outcomes
- **Brier Score**: Squared error of probabilistic forecasts
- **Log Loss**: Logarithmic scoring rule for probabilistic accuracy
- **Resolution**: Ability to make distinct predictions for different questions
- **Sharpness**: Tendency to make confident predictions

## Conclusion

Selecting the right personality for a particular forecasting question is both an art and a science. Start with the recommendations in this guide, but don't hesitate to experiment with different configurations. Over time, you'll develop an intuition for which personalities work best for your specific forecasting needs.

Remember that different personalities often provide complementary insights. For critical forecasts, consider using multiple personalities to get a more robust perspective on the question. 