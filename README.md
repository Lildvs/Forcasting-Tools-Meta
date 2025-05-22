# Forecasting Tools Demo

This project demonstrates how to use the [Metaculus forecasting-tools](https://github.com/Metaculus/forecasting-tools) package for creating forecasting bots that can help predict the outcomes of various questions.

## Setup

### Prerequisites

- Python 3.9 or higher
- OpenAI API key
- Exa API key (for web search capabilities)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/Lildvs/Forcasting-Tools-Meta
   cd forecasting-tools-demo
   ```

2. Install dependencies using one of these methods:

   **Using pip with requirements.txt:**
   ```
   pip install -r requirements.txt
   ```

   **Using setup.py:**
   ```
   pip install -e .
   ```

   **Installing directly:**
   ```
   pip install forecasting-tools python-dotenv
   ```

3. Create a `.env` file with your API keys:
   ```
   PYTHONPATH=.
   
   # The most important API keys (required for basic functionality)
   OPENAI_API_KEY=your_openai_api_key_here
   EXA_API_KEY=your_exa_api_key_here
   
   # Optional API keys
   PERPLEXITY_API_KEY=
   ASKNEWS_CLIENT_ID=
   ASKNEWS_SECRET=
   OPENROUTER_API_KEY=
   ANTHROPIC_API_KEY=
   HUGGINGFACE_API_KEY=
   
   # For Metaculus API integration
   METACULUS_TOKEN=
   
   # Enable file writing (needed for logs and saving reports)
   FILE_WRITING_ALLOWED=TRUE
   ```

   You can get these API keys from:
   - OpenAI API Key: [OpenAI Platform](https://platform.openai.com/api-keys)
   - Exa API Key: [Exa.ai](https://www.exa.ai/)
   - Metaculus Token: [Metaculus](https://www.metaculus.com/)

## Running the Examples

### Basic Forecast Bot

Run the basic forecast bot example:

```
python my_forecast_demo.py
```

This script demonstrates how to:
1. Create a simple binary question
2. Use the TemplateBot to forecast the outcome
3. Display the prediction and explanation

### Custom Forecast Bot

Run the custom forecast bot example:

```
python my_custom_bot.py
```

This script shows how to:
1. Create a custom forecasting bot by inheriting from TemplateBot
2. Override research and forecasting methods
3. Forecast multiple questions with custom parameters

## Important Notes

- **API Usage**: Running these examples will use your API keys and incur charges based on usage. Monitor your usage to avoid unexpected costs.
- **Folders**: The scripts will create directories for saving forecast reports. These will be automatically created when running the examples.
- **API Keys**: Never share your API keys. The `.env` file is added to `.gitignore` to prevent accidental commits.

## Key Components

### Question Types
- **BinaryQuestion**: Questions with yes/no answers (probabilities)
- **NumericQuestion**: Questions with numeric answers (distributions)
- **MultipleChoiceQuestion**: Questions with multiple choice answers

### Forecasting Bot Features
- Research using AI and web searches
- Multiple predictions per question
- Aggregation of predictions
- Saving detailed reports
- Integration with Metaculus

## Resources

- [Forecasting Tools GitHub Repository](https://github.com/Metaculus/forecasting-tools)
- [Forecasting Tools Demo Website](https://forecasting-tools.streamlit.app/)
- [Metaculus Website](https://www.metaculus.com/)
- [Metaculus Discord](https://discord.gg/Dtq4JNdXnw) 