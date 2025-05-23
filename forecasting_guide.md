# Forecasting Project Guide

## 1. Project Purpose

The primary purpose of this project is to generate the most accurate forecasts possible on user queries. We aim to:
- Provide precise probability estimates for binary questions
- Deliver well-calibrated confidence intervals for numeric predictions
- Base all forecasts on thorough research and rigorous reasoning
- Continuously improve prediction accuracy through systematic evaluation
- Outperform individual human forecasters and competing AI systems

## 2. Development Approach

When programming and implementing this forecasting system, Cursor AI should:
- Leverage the Metaculus forecasting-tools package for core functionality
- Implement custom research methods that combine multiple information sources
- Use ensemble methods to aggregate predictions from different models
- Apply Bayesian updating when new information becomes available
- Properly decompose complex questions into more tractable sub-questions
- Maintain clear explanation chains for all predictions
- Test predictions against historical data whenever possible
- Store forecasts and track performance over time

## 3. External Platforms

This project integrates with several external platforms:

### OpenAI
- Primary source for general reasoning and decomposition
- Used for generating structured analysis of questions
- Models include GPT-4o for high-quality reasoning

### Perplexity
- Specialized for real-time web search and information synthesis
- Provides up-to-date information with source attribution
- Helps identify recent developments not in training data

### Crawl4ai
- Used for targeted web crawling on specific domains
- Enables deeper analysis of specialized sources
- Helps gather detailed information for niche questions

### Metaculus
- Platform for formal forecasting questions and competitions
- Source of well-defined questions with clear resolution criteria
- Provides historical community predictions for calibration

### Exa
- Semantic search engine for finding relevant information
- Supports structured data extraction from web content
- Enables citation of specific paragraphs from sources

## 4. Metaculus Quarterly Competition

Our participation in the Metaculus quarterly competition represents a concrete benchmark for our forecasting capabilities:

### Goals
- Primary goal: Achieve 1st place in the competition
- Minimum acceptable performance: Place in the top 3
- Surpass both individual superforecasters and other AI systems

### Strategy
- Carefully analyze competition questions to understand exact resolution criteria
- Allocate more research time to high-weight questions
- Update forecasts regularly as new information becomes available
- Avoid overconfidence by properly quantifying uncertainty
- Learn from performance on earlier questions to improve later forecasts

## 5. Additional Requirements

This section is reserved for future requirements and adjustments as the project evolves.

Potential areas for future enhancement:
- Do not create a app.py file for the GUI, we are using front_end/Home.py as the path for our GUI
- We are using Streamlit cloud for our deployment of this project. Do not attempt to create a local instance of this project, unless for testing purposes
- Implementation of more sophisticated aggregation methods
- Development of domain-specific reasoning modules
- Incorporation of automated evaluation metrics
- Creation of visualization tools for forecast analysis 