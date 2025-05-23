# Forecasting Tools Streamlit App

This Streamlit application provides a user-friendly interface for generating forecasts while tracking token usage and costs.

## Features

- **Forecast Generation**: Create binary and numeric forecasts with customizable parameters
- **Cost Tracking**: Monitor token usage and costs for each forecast in real-time
- **Cost Analytics**: Visualize and analyze cost data with interactive charts
- **Cost Export**: Export cost history to CSV for further analysis

## Installation

1. Make sure you have Python 3.10+ installed
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the App

### Method 1: Using the Python script

```bash
python run_streamlit.py
```

### Method 2: Using the shell script

```bash
./run_app.sh
```

### Method 3: Using Streamlit directly

```bash
streamlit run streamlit_app.py
```

## Usage

1. Navigate to the "Forecasting" tab to create forecasts
2. Fill in the forecast form with your question details
3. Select the model and personality to use
4. Click "Generate Forecast" to create a forecast
5. View the forecast results and cost information
6. Switch to the "Cost History" tab to view detailed cost analytics

## Cost Information

The app displays cost information in several places:

- **Post-Query Feedback**: After each forecast, the app shows how many tokens were used and the cost of the forecast
- **Grand Total**: The total cost of all forecasts is displayed in the top-right corner
- **Cost History Tab**: This tab provides detailed cost analytics, including:
  - Total cost, average cost, and total forecasts
  - Cost breakdowns by model and personality
  - Daily cost trends
  - Detailed history table with all forecasts

## Customization

You can customize the app by modifying the following files:

- `streamlit_app.py`: Main Streamlit application code
- `forecasting_tools/cost_tracking/cost_tracker.py`: Configure model pricing and database settings
- `run_app.sh` or `run_streamlit.py`: Modify startup settings

## Deployment

The app can be deployed to Streamlit Cloud by connecting your GitHub repository and selecting the `streamlit_app.py` file. Make sure to include all the required dependencies in your `requirements.txt` file.

## Troubleshooting

- **Database Errors**: If you encounter database errors, check that the data directory exists and is writable.
- **Missing Dependencies**: Make sure all dependencies are installed with `pip install -r requirements.txt`.
- **Token Estimation Issues**: If token usage seems inaccurate, check the model configuration and ensure proper metadata is being returned.

## License

This project is licensed under the terms of the MIT license. 