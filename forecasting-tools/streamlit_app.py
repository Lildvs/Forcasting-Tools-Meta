import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import uuid
from pathlib import Path
import os

from forecasting_tools.data_models.questions import BinaryQuestion, NumericQuestion
from forecasting_tools.cost_tracking import CostTrackingBot, CostTracker

# Configure the page
st.set_page_config(
    page_title="Forecasting Tools",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'cost_tracker' not in st.session_state:
    # Create data directory if it doesn't exist
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # Initialize cost tracker with data file in the app directory
    st.session_state.cost_tracker = CostTracker(db_path=str(data_dir / "forecast_costs.db"))

if 'forecast_history' not in st.session_state:
    st.session_state.forecast_history = []

# Function to format cost as currency
def format_cost(cost):
    return f"${cost:.4f}"

# CSS for the total cost badge in the top right
# This uses custom CSS to position the element
st.markdown("""
<style>
.total-cost-badge {
    position: absolute;
    top: 0.5rem;
    right: 1.5rem;
    padding: 0.5rem 1rem;
    background-color: #f0f2f6;
    border-radius: 0.5rem;
    font-weight: bold;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    z-index: 1000;
}
.total-cost-badge .cost {
    color: #ff4b4b;
    font-size: 1.2rem;
}
.total-cost-badge .label {
    font-size: 0.8rem;
    color: #262730;
}
</style>
""", unsafe_allow_html=True)

# Display total cost in top right
total_cost = st.session_state.cost_tracker.get_total_cost()
st.markdown(
    f"""
    <div class="total-cost-badge">
        <div class="label">Total Cost</div>
        <div class="cost">{format_cost(total_cost)}</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Create tabs
tab1, tab2 = st.tabs(["Forecasting", "Cost History"])

with tab1:
    st.title("Forecasting Tool")
    
    # Question form
    with st.form("forecast_form"):
        question_type = st.selectbox("Question Type", ["Binary", "Numeric"])
        question_text = st.text_area("Question", "Will AI significantly impact employment in the next 5 years?")
        background_info = st.text_area("Background Information (optional)", "")
        
        # Model and personality selection
        col1, col2 = st.columns(2)
        with col1:
            model_name = st.selectbox(
                "Model", 
                ["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
            )
        with col2:
            personality_name = st.selectbox(
                "Personality", 
                ["None", "analytical", "creative", "balanced", "bayesian"]
            )
        
        # Numeric question specific inputs (shown conditionally)
        if question_type == "Numeric":
            col1, col2, col3 = st.columns(3)
            with col1:
                min_value = st.number_input("Minimum Value", value=0.0)
            with col2:
                max_value = st.number_input("Maximum Value", value=100.0)
            with col3:
                unit = st.text_input("Unit", "")
        
        submitted = st.form_submit_button("Generate Forecast")
    
    # Process form submission
    if submitted:
        # Show a spinner while forecasting
        with st.spinner("Generating forecast..."):
            # Initialize the bot
            bot = CostTrackingBot(
                personality_name=None if personality_name == "None" else personality_name,
                model_name=model_name
            )
            
            # Create question object
            if question_type == "Binary":
                question = BinaryQuestion(
                    question_text=question_text,
                    background_info=background_info,
                    resolution_criteria="",
                    fine_print=""
                )
                # Generate forecast
                forecast = bot.forecast_binary(question)
                
                # Display the result
                st.subheader("Forecast Result")
                st.write(f"Probability: {forecast.binary_prob:.2%}")
                st.write("Reasoning:")
                st.write(forecast.reasoning)
                
            else:  # Numeric
                question = NumericQuestion(
                    question_text=question_text,
                    background_info=background_info,
                    resolution_criteria="",
                    fine_print="",
                    min_value=min_value,
                    max_value=max_value,
                    unit=unit
                )
                # Generate forecast
                forecast = bot.forecast_numeric(question)
                
                # Display the result
                st.subheader("Forecast Result")
                st.write(f"Mean: {forecast.mean} {unit}")
                st.write(f"Range: {forecast.low} - {forecast.high} {unit}")
                st.write("Reasoning:")
                st.write(forecast.reasoning)
            
            # Display cost information
            if hasattr(forecast, 'metadata') and 'cost_info' in forecast.metadata:
                cost_info = forecast.metadata['cost_info']
                with st.container():
                    st.info(
                        f"📊 This forecast used {cost_info['tokens_used']} tokens and cost "
                        f"{format_cost(cost_info['cost_usd'])}"
                    )

with tab2:
    st.title("Cost History")
    
    # Get cost history data
    cost_history = st.session_state.cost_tracker.get_cost_history(limit=1000)
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    with col1:
        time_filter = st.selectbox(
            "Time Range", 
            ["All Time", "Today", "Last 7 Days", "Last 30 Days", "This Month"]
        )
    with col2:
        sort_by = st.selectbox(
            "Sort By", 
            ["Most Recent", "Most Expensive", "Least Expensive"]
        )
    with col3:
        grouping = st.selectbox(
            "Group By", 
            ["None", "Model", "Personality", "Day"]
        )
    
    # Export button
    export_col1, export_col2 = st.columns([3, 1])
    with export_col2:
        if st.button("Export to CSV"):
            export_path = "./data/cost_history_export.csv"
            if st.session_state.cost_tracker.export_to_csv(export_path):
                st.success(f"Exported to {export_path}")
                
                # Create a download link
                with open(export_path, "r") as f:
                    st.download_button(
                        label="Download CSV",
                        data=f,
                        file_name="cost_history_export.csv",
                        mime="text/csv"
                    )
            else:
                st.error("Failed to export data")
    
    # Convert to pandas DataFrame for easier manipulation
    if cost_history:
        df = pd.DataFrame([{
            "Date": cost.timestamp,
            "Question": cost.question_text,
            "Tokens": cost.tokens_used,
            "Cost": cost.cost_usd,
            "Model": cost.model_name,
            "Personality": cost.personality_name or "None"
        } for cost in cost_history])
        
        # Apply time filter
        now = datetime.now()
        if time_filter == "Today":
            df = df[df["Date"].dt.date == now.date()]
        elif time_filter == "Last 7 Days":
            df = df[df["Date"] >= (now - timedelta(days=7))]
        elif time_filter == "Last 30 Days":
            df = df[df["Date"] >= (now - timedelta(days=30))]
        elif time_filter == "This Month":
            df = df[df["Date"].dt.month == now.month]
            df = df[df["Date"].dt.year == now.year]
        
        # Apply sorting
        if sort_by == "Most Recent":
            df = df.sort_values("Date", ascending=False)
        elif sort_by == "Most Expensive":
            df = df.sort_values("Cost", ascending=False)
        elif sort_by == "Least Expensive":
            df = df.sort_values("Cost", ascending=True)
        
        # Show summary statistics
        st.subheader("Summary")
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cost", format_cost(df["Cost"].sum()))
        with col2:
            st.metric("Average Cost", format_cost(df["Cost"].mean()))
        with col3:
            st.metric("Total Forecasts", f"{len(df):,}")
        with col4:
            st.metric("Total Tokens", f"{int(df['Tokens'].sum()):,}")
            
        # Create visualizations
        st.subheader("Visualizations")
        
        # Cost over time chart
        df_daily = df.copy()
        df_daily["Day"] = df_daily["Date"].dt.date
        daily_costs = df_daily.groupby("Day")["Cost"].sum().reset_index()
        
        if not daily_costs.empty:
            fig1 = px.line(
                daily_costs, 
                x="Day", 
                y="Cost", 
                title="Daily Forecast Costs",
                labels={"Cost": "Cost (USD)", "Day": "Date"}
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        # Cost breakdown by model/personality
        if grouping == "Model" and not df.empty:
            model_costs = df.groupby("Model")["Cost"].sum().reset_index()
            fig2 = px.pie(
                model_costs, 
                values="Cost", 
                names="Model", 
                title="Cost by Model",
                hole=0.4
            )
            st.plotly_chart(fig2, use_container_width=True)
        elif grouping == "Personality" and not df.empty:
            personality_costs = df.groupby("Personality")["Cost"].sum().reset_index()
            fig2 = px.pie(
                personality_costs, 
                values="Cost", 
                names="Personality", 
                title="Cost by Personality",
                hole=0.4
            )
            st.plotly_chart(fig2, use_container_width=True)
        elif grouping == "Day" and not df.empty:
            fig2 = px.bar(
                daily_costs, 
                x="Day", 
                y="Cost",
                title="Cost by Day"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Display full history as a table
        st.subheader("Detailed History")
        
        # Format the date and cost columns
        df["Formatted Date"] = df["Date"].dt.strftime("%Y-%m-%d %H:%M")
        df["Formatted Cost"] = df["Cost"].apply(format_cost)
        
        # Display the table with formatted columns
        st.dataframe(
            df[["Formatted Date", "Question", "Tokens", "Formatted Cost", "Model", "Personality"]]
            .rename(columns={
                "Formatted Date": "Date",
                "Formatted Cost": "Cost"
            }),
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No forecast history yet. Start making some forecasts!")

# Add app info in sidebar
with st.sidebar:
    st.title("Forecasting Tools")
    st.info(
        "This app allows you to generate forecasts using various models and personalities, "
        "while tracking token usage and costs."
    )
    
    st.subheader("Cost Statistics")
    cost_stats = st.session_state.cost_tracker.get_cost_statistics()
    
    st.write(f"Total forecasts: {cost_stats['total_forecasts']:,}")
    st.write(f"Total cost: {format_cost(cost_stats['total_cost'])}")
    st.write(f"Average cost per forecast: {format_cost(cost_stats['average_cost'])}")
    st.write(f"Recent forecasts (7 days): {cost_stats['recent_forecasts']:,}")
    st.write(f"Recent cost (7 days): {format_cost(cost_stats['recent_cost'])}")
    
    # Add disclaimer
    st.caption(
        "Note: Costs are calculated based on estimated token usage and may vary slightly "
        "from actual charges by API providers."
    )

# Run the app with: streamlit run streamlit_app.py 