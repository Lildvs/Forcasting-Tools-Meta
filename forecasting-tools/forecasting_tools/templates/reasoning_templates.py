"""
Reasoning Templates

This module provides templates for different reasoning approaches, including
evidence evaluation, probability calibration, and Bayesian updating.
"""

# Binary question templates
BINARY_QUESTION_BAYESIAN_TEMPLATE = """
# Bayesian Reasoning Process for Binary Question

## Step 1: Framing the Question
{question_text}

### Background Information
{background_info}

### Resolution Criteria
{resolution_criteria}

## Step 2: Establish Prior Probability
Based on the background information and general knowledge about this topic, I'll establish a prior probability:

1. First, I'll identify the base rate by considering:
   - Historical frequency of similar events
   - Default outcomes in comparable situations
   - General context and constraints

2. Initial assessment of base rate: {base_rate_probability}

## Step 3: Evidence Evaluation
I'll now evaluate the evidence from my research and update my probability:

{evidence_evaluation}

## Step 4: Bayesian Updates
Starting with my prior probability of {prior_probability}, I'll update based on each piece of evidence:

{bayesian_updates}

## Step 5: Cognitive Bias Check
I'll check for potential cognitive biases that might affect my judgment:

{bias_check}

## Step 6: Final Probability Assessment
After considering all evidence and adjusting for potential biases:

- Final probability: {final_probability}
- Confidence interval: {confidence_interval}
- Confidence level: {confidence_level}

## Step 7: Key Uncertainties
The main uncertainties affecting this forecast are:

{uncertainties}
"""

# Numeric question templates
NUMERIC_QUESTION_FERMI_TEMPLATE = """
# Fermi Estimation Process for Numeric Question

## Step 1: Framing the Question
{question_text}

### Background Information
{background_info}

### Resolution Criteria
{resolution_criteria}

## Step 2: Decomposition into Components
I'll break this estimation into component factors:

{decomposition}

## Step 3: Estimation of Components
For each component, I'll provide an estimate with uncertainty:

{component_estimates}

## Step 4: Combining Estimates
Combining these component estimates:

{calculation}

## Step 5: Cognitive Bias Check
I'll check for potential cognitive biases that might affect my judgment:

{bias_check}

## Step 6: Calibration
Adjusting for historical calibration and uncertainty:

{calibration}

## Step 7: Final Estimate
After considering all factors and adjusting for potential biases:

- Point estimate: {point_estimate} {units}
- 90% confidence interval: {confidence_interval} {units}
- Confidence level: {confidence_level}

## Step 8: Key Uncertainties
The main uncertainties affecting this forecast are:

{uncertainties}
"""

# Multiple choice question templates
MULTIPLE_CHOICE_TEMPLATE = """
# Comparative Reasoning Process for Multiple Choice Question

## Step 1: Framing the Question
{question_text}

### Background Information
{background_info}

### Resolution Criteria
{resolution_criteria}

## Step 2: Option Analysis
I'll analyze each option individually:

{option_analysis}

## Step 3: Comparative Assessment
Comparing the options against each other:

{comparative_assessment}

## Step 4: Evidence Evaluation
Evaluating the evidence for each option:

{evidence_evaluation}

## Step 5: Cognitive Bias Check
I'll check for potential cognitive biases that might affect my judgment:

{bias_check}

## Step 6: Probability Distribution
After considering all evidence and adjusting for potential biases:

{probability_distribution}

## Step 7: Consistency Check
Ensuring probabilities sum to 100% and checking for internal consistency:

{consistency_check}

## Step 8: Key Uncertainties
The main uncertainties affecting this forecast are:

{uncertainties}
"""

# Evidence evaluation templates
EVIDENCE_EVALUATION_TEMPLATE = """
### Evidence: {evidence_content}
- **Type**: {evidence_type}
- **Source**: {source}
- **Relevance**: {relevance_score}/10
- **Reliability**: {reliability_score}/10
- **Direction**: {impact_direction}
- **Confidence**: {confidence_level}

#### Analysis:
{evidence_analysis}
"""

# Bayesian update template
BAYESIAN_UPDATE_TEMPLATE = """
### Update based on: {evidence_summary}
- Prior probability: {prior_probability}
- Likelihood ratio: {likelihood_ratio}
  - P(Evidence|Hypothesis): {p_evidence_given_hypothesis}
  - P(Evidence|~Hypothesis): {p_evidence_given_not_hypothesis}
- Posterior probability: {posterior_probability}
"""

# Fermi decomposition component template
FERMI_COMPONENT_TEMPLATE = """
### Component: {component_name}
- **Estimate**: {estimate} {units}
- **Uncertainty**: ±{uncertainty_percent}% ({lower_bound} to {upper_bound})
- **Reasoning**: {reasoning}
- **Sources**: {sources}
"""

# Uncertainty template
UNCERTAINTY_TEMPLATE = """
### Uncertainty: {uncertainty_name}
- **Impact**: {impact_level}
- **Direction**: {direction}
- **Description**: {description}
"""

# Cognitive bias template
COGNITIVE_BIAS_TEMPLATE = """
### Potential Bias: {bias_name}
- **Description**: {bias_description}
- **Potential Impact**: {potential_impact}
- **Mitigation Strategy**: {mitigation_strategy}
"""

# Final forecast template for documentation
FINAL_FORECAST_TEMPLATE = """
# Final Forecast: {question_text}

## Summary
{forecast_summary}

## Quantitative Assessment
{quantitative_assessment}

## Evidence and Reasoning
{evidence_and_reasoning}

## Uncertainties and Caveats
{uncertainties_and_caveats}

## Confidence Assessment
{confidence_assessment}
"""

# Template for reasoning steps
REASONING_STEP_TEMPLATE = """
## {step_number}: {step_title}
{step_content}

{evidence_list}

{intermediate_conclusion}
"""

# Template mapping reasoning approaches to templates
APPROACH_TEMPLATES = {
    "bayesian": BINARY_QUESTION_BAYESIAN_TEMPLATE,
    "fermi": NUMERIC_QUESTION_FERMI_TEMPLATE,
    "analog_comparison": MULTIPLE_CHOICE_TEMPLATE,
    "outside_view": """
# Outside View Reasoning Process

## Step 1: Framing the Question
{question_text}

## Step 2: Reference Class Selection
I'll identify an appropriate reference class for this question:

{reference_class_selection}

## Step 3: Reference Class Analysis
Analyzing the reference class data:

{reference_class_analysis}

## Step 4: Initial Forecast Based on Reference Class
Based solely on the reference class:

{initial_forecast}

## Step 5: Adjustments for Specific Case
Adjusting for specific factors in this case:

{specific_adjustments}

## Step 6: Final Forecast
After reference class forecasting and specific adjustments:

{final_forecast}

## Step 7: Key Uncertainties
The main uncertainties affecting this forecast are:

{uncertainties}
""",
    "inside_view": """
# Inside View Reasoning Process

## Step 1: Framing the Question
{question_text}

## Step 2: Causal Model Development
I'll develop a causal model for this question:

{causal_model}

## Step 3: Key Factors Analysis
Analyzing the key factors in the causal model:

{key_factors}

## Step 4: Scenario Development
Based on the causal model, I'll develop scenarios:

{scenarios}

## Step 5: Probability Assignment
Assigning probabilities to scenarios:

{probability_assignment}

## Step 6: Final Forecast
After scenario analysis:

{final_forecast}

## Step 7: Key Uncertainties
The main uncertainties affecting this forecast are:

{uncertainties}
""",
    "trend_extrapolation": """
# Trend Extrapolation Reasoning Process

## Step 1: Framing the Question
{question_text}

## Step 2: Historical Trend Identification
I'll identify relevant historical trends:

{historical_trends}

## Step 3: Trend Analysis
Analyzing the trends:

{trend_analysis}

## Step 4: Extrapolation Methods
Methods for extrapolating the trends:

{extrapolation_methods}

## Step 5: Adjustments for Context
Adjusting for specific contextual factors:

{context_adjustments}

## Step 6: Final Forecast
After trend extrapolation and adjustments:

{final_forecast}

## Step 7: Key Uncertainties
The main uncertainties affecting this forecast are:

{uncertainties}
""",
    "scout_mindset": """
# Scout Mindset Reasoning Process

## Step 1: Framing the Question
{question_text}

## Step 2: Initial Impressions
My initial impressions about this question:

{initial_impressions}

## Step 3: Alternative Perspectives
I'll consider alternative perspectives:

{alternative_perspectives}

## Step 4: Evidential Reasoning
Analyzing the evidence for different perspectives:

{evidential_reasoning}

## Step 5: Probability Calibration
Calibrating probabilities based on evidence:

{probability_calibration}

## Step 6: Final Forecast
After considering multiple perspectives:

{final_forecast}

## Step 7: Key Uncertainties
The main uncertainties affecting this forecast are:

{uncertainties}
""",
    "counterfactual": """
# Counterfactual Reasoning Process

## Step 1: Framing the Question
{question_text}

## Step 2: Current State Assessment
Assessing the current state:

{current_state}

## Step 3: Counterfactual Scenarios
Developing counterfactual scenarios:

{counterfactual_scenarios}

## Step 4: Causal Analysis
Analyzing causal factors across scenarios:

{causal_analysis}

## Step 5: Probability Assessment
Assessing probabilities for different scenarios:

{probability_assessment}

## Step 6: Final Forecast
After counterfactual analysis:

{final_forecast}

## Step 7: Key Uncertainties
The main uncertainties affecting this forecast are:

{uncertainties}
""",
    "decomposition": """
# Decomposition Reasoning Process

## Step 1: Framing the Question
{question_text}

## Step 2: Question Decomposition
Breaking down the question into components:

{question_decomposition}

## Step 3: Component Analysis
Analyzing each component:

{component_analysis}

## Step 4: Component Forecasts
Forecast for each component:

{component_forecasts}

## Step 5: Recombination
Recombining component forecasts:

{recombination}

## Step 6: Final Forecast
After decomposition and recombination:

{final_forecast}

## Step 7: Key Uncertainties
The main uncertainties affecting this forecast are:

{uncertainties}
""",
} 