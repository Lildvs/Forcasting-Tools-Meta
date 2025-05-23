import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import uuid
from pathlib import Path
import os
import inspect
from typing import Optional, Dict, Any, List, Tuple, Union

# Import data models
from forecasting_tools.data_models.questions import BinaryQuestion, NumericQuestion
from forecasting_tools.data_models.forecast_report import ReasonedPrediction
from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.numeric_report import NumericReport

# Import cost tracking
from forecasting_tools.cost_tracking import CostTrackingBot, CostTracker
from forecasting_tools.ai_models.general_llm import GeneralLlm

# Import LLM configuration
from forecasting_tools.llm_config import LLMConfigManager

# Import reasoning modules
from forecasting_tools.forecast_bots.reasoning import (
    ConfidenceLevel,
    Evidence, 
    EvidenceType,
    StructuredReasoning, 
    create_reasoning_for_question
)

# Import our UI components
from forecasting_tools.front_end.components.probability_visualization import (
    display_binary_forecast,
    display_numeric_forecast,
    create_binary_probability_chart,
    create_numeric_distribution_chart
)
from forecasting_tools.front_end.components.research_sources import (
    display_research_sources,
    display_reasoning_steps,
    display_biases_and_uncertainties
)
from forecasting_tools.front_end.components.confidence_display import (
    display_confidence_breakdown,
    create_confidence_gauge
)
from forecasting_tools.front_end.components.ui_utils import (
    load_css,
    display_info_box,
    create_card
)

# Configure the page
st.set_page_config(
    page_title="Forecasting Tools",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
css_path = Path(__file__).parent / "forecasting_tools" / "front_end" / "components" / "styles.css"
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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

# Function to create a CostTrackingBot with proper parameter handling
def create_cost_tracking_bot(model_name=None, personality_name=None):
    """
    Create a CostTrackingBot instance with proper parameter handling.
    
    Args:
        model_name: The name of the model to use
        personality_name: The name of the personality to use
        
    Returns:
        A CostTrackingBot instance
    """
    # Create a default LLM config with the specified model
    llms = None
    if model_name:
        # Use the new LLMConfigManager for default settings
        config = LLMConfigManager.get_default_config()
        llms = {
            "default": GeneralLlm(model=model_name, temperature=0.3),
            "summarizer": GeneralLlm(model=config["summarizer"]["model"], temperature=0.1),
            "researcher": GeneralLlm(model=config["researcher"]["model"], temperature=0.1),
        }
    
    # Create the bot with the proper parameters
    return CostTrackingBot(
        llms=llms,
        personality_name=personality_name,
        cost_tracker=st.session_state.cost_tracker
    )

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
tab1, tab2, tab3 = st.tabs(["Forecasting", "Research Explorer", "Cost History"])

with tab1:
    st.title("Forecasting Tool")
    
    # Question form
    with st.form("forecast_form"):
        question_type = st.selectbox("Question Type", ["Binary", "Numeric"])
        question_text = st.text_area("Question", "Will AI significantly impact employment in the next 5 years?")
        background_info = st.text_area("Background Information (optional)", "")
        
        # Model and personality selection
        col1, col2, col3 = st.columns(3)
        with col1:
            # Get models from LLMConfigManager
            config = LLMConfigManager.get_default_config()
            default_models = [
                "gpt-4.1",  # Latest default from config
                "gpt-4o",
                "gpt-4o-mini",
                "claude-3-opus",
                "claude-3-sonnet",
                "claude-3-haiku"
            ]
            model_name = st.selectbox(
                "Model", 
                default_models
            )
        with col2:
            personality_name = st.selectbox(
                "Personality", 
                ["None", "analytical", "creative", "balanced", "bayesian"]
            )
        with col3:
            confidence_display = st.checkbox("Show Confidence Analysis", value=True)
        
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
            processed_personality = None if personality_name == "None" else personality_name
            bot = create_cost_tracking_bot(model_name=model_name, personality_name=processed_personality)
            
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
                
                # Display the result using our components
                st.subheader("Forecast Result")
                
                # Extract research from reasoning if available
                research = ""
                reasoning = forecast.reasoning
                if "## Research" in reasoning:
                    parts = reasoning.split("## Research", 1)
                    if len(parts) > 1:
                        research_section = "## Research" + parts[1].split("##", 1)[0]
                        research = research_section
                
                # Create a structured reasoning object for the forecast
                structured_reasoning = create_reasoning_for_question(question)
                
                # Create mock reasoning steps from the forecast reasoning
                reasoning_steps = []
                if "##" in reasoning:
                    sections = reasoning.split("##")
                    for i, section in enumerate(sections[1:], 1):  # Skip the first empty part
                        section_title = section.split("\n", 1)[0].strip()
                        section_content = section.split("\n", 1)[1].strip() if "\n" in section else ""
                        reasoning_steps.append({
                            "title": f"Step {i}",
                            "type": section_title,
                            "content": section_content
                        })
                
                # Display using our visualization component
                display_binary_forecast(
                    probability=forecast.prediction,
                    reasoning=reasoning,
                    community_prediction=question.community_prediction_at_access_time
                )
                
                # Display confidence information if requested
                if confidence_display:
                    st.markdown("### Confidence Analysis")
                    
                    # Create mock confidence scores
                    component_scores = {
                        "evidence_quality": 0.75,
                        "reasoning_process": 0.8,
                        "relevant_expertise": 0.7,
                        "information_recency": 0.6,
                        "bias_consideration": 0.65
                    }
                    
                    # Display confidence breakdown
                    display_confidence_breakdown(
                        component_scores=component_scores,
                        overall_confidence=0.7,
                        strengths=["Strong evidence quality", "Thorough reasoning process"],
                        weaknesses=["Limited information recency"]
                    )
                
                # Display research sources if available
                if research:
                    display_research_sources(
                        research=research,
                        allow_filtering=True
                    )
                
                # Display reasoning steps
                if reasoning_steps:
                    display_reasoning_steps(
                        steps=reasoning_steps,
                        expand_all=False
                    )
                    
                    # Display mock biases and uncertainties
                    display_biases_and_uncertainties(
                        biases=["Recency bias", "Confirmation bias", "Availability bias"],
                        uncertainties=["Future technological advancements", "Economic policy changes", "COVID-19 long-term impact"]
                    )
            
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
                
                # Extract research from reasoning if available
                research = ""
                reasoning = forecast.reasoning
                if "## Research" in reasoning:
                    parts = reasoning.split("## Research", 1)
                    if len(parts) > 1:
                        research_section = "## Research" + parts[1].split("##", 1)[0]
                        research = research_section
                
                # Create mock reasoning steps from the forecast reasoning
                reasoning_steps = []
                if "##" in reasoning:
                    sections = reasoning.split("##")
                    for i, section in enumerate(sections[1:], 1):  # Skip the first empty part
                        section_title = section.split("\n", 1)[0].strip()
                        section_content = section.split("\n", 1)[1].strip() if "\n" in section else ""
                        reasoning_steps.append({
                            "title": f"Step {i}",
                            "type": section_title,
                            "content": section_content
                        })
                
                # Display using our visualization component
                display_numeric_forecast(
                    mean=forecast.mean,
                    low=forecast.low,
                    high=forecast.high,
                    reasoning=reasoning,
                    unit=unit
                )
                
                # Display confidence information if requested
                if confidence_display:
                    st.markdown("### Confidence Analysis")
                    
                    # Create mock confidence scores
                    component_scores = {
                        "evidence_quality": 0.7,
                        "reasoning_process": 0.75,
                        "relevant_expertise": 0.65,
                        "information_recency": 0.6,
                        "bias_consideration": 0.7
                    }
                    
                    # Display confidence breakdown
                    display_confidence_breakdown(
                        component_scores=component_scores,
                        overall_confidence=0.68,
                        strengths=["Strong reasoning process", "Good bias consideration"],
                        weaknesses=["Limited relevant expertise"]
                    )
                
                # Display research sources if available
                if research:
                    display_research_sources(
                        research=research,
                        allow_filtering=True
                    )
                
                # Display reasoning steps
                if reasoning_steps:
                    display_reasoning_steps(
                        steps=reasoning_steps,
                        expand_all=False
                    )
                    
                    # Display mock biases and uncertainties
                    display_biases_and_uncertainties(
                        biases=["Planning fallacy", "Anchoring bias", "Overconfidence bias"],
                        uncertainties=["Data quality issues", "Model parameterization", "External unpredictable factors"]
                    )
            
            # Display cost information
            if hasattr(forecast, 'metadata') and 'cost_info' in forecast.metadata:
                cost_info = forecast.metadata['cost_info']
                display_info_box(
                    message=f"This forecast used {cost_info['tokens_used']} tokens and cost {format_cost(cost_info['cost_usd'])}",
                    type="info"
                )
            
            # Add to forecast history for the research explorer
            st.session_state.forecast_history.append({
                "question": question,
                "forecast": forecast,
                "timestamp": datetime.now(),
                "model": model_name,
                "personality": personality_name
            })

with tab2:
    st.title("Research Explorer")
    
    # Show forecast history if available
    if not st.session_state.forecast_history:
        st.markdown("No forecasts generated yet. Generate a forecast to explore research sources.")
    else:
        # Create a selection for previously generated forecasts
        forecast_options = [
            f"{idx+1}. {hist['question'].question_text[:50]}... ({hist['timestamp'].strftime('%Y-%m-%d %H:%M')})"
            for idx, hist in enumerate(st.session_state.forecast_history)
        ]
        
        selected_forecast_idx = st.selectbox(
            "Select a forecast to explore:",
            range(len(forecast_options)),
            format_func=lambda i: forecast_options[i]
        )
        
        # Get selected forecast
        selected_forecast_data = st.session_state.forecast_history[selected_forecast_idx]
        forecast = selected_forecast_data["forecast"]
        question = selected_forecast_data["question"]
        
        # Display forecast details
        st.subheader(f"Question: {question.question_text}")
        
        if isinstance(forecast, BinaryReport):
            # Display binary forecast visualization
            col1, col2 = st.columns([1, 2])
            with col1:
                fig = create_binary_probability_chart(
                    probability=forecast.prediction,
                    community_prediction=question.community_prediction_at_access_time
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown(f"**Probability:** {forecast.prediction:.2%}")
                if question.community_prediction_at_access_time:
                    st.markdown(f"**Community Prediction:** {question.community_prediction_at_access_time:.2%}")
                st.markdown(f"**Model:** {selected_forecast_data['model']}")
                st.markdown(f"**Personality:** {selected_forecast_data['personality']}")
                
        elif hasattr(forecast, 'mean'):  # Numeric forecast
            # Display numeric forecast visualization
            col1, col2 = st.columns([1, 2])
            with col1:
                fig = create_numeric_distribution_chart(
                    mean=forecast.mean,
                    low=forecast.low,
                    high=forecast.high,
                    unit=question.unit if hasattr(question, 'unit') else ""
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                unit = question.unit if hasattr(question, 'unit') else ""
                st.markdown(f"**Mean Estimate:** {forecast.mean} {unit}")
                st.markdown(f"**Confidence Interval:** {forecast.low} - {forecast.high} {unit}")
                st.markdown(f"**Model:** {selected_forecast_data['model']}")
                st.markdown(f"**Personality:** {selected_forecast_data['personality']}")
        
        # Extract research from reasoning
        research = ""
        reasoning = forecast.reasoning
        if "## Research" in reasoning:
            parts = reasoning.split("## Research", 1)
            if len(parts) > 1:
                research_section = "## Research" + parts[1].split("##", 1)[0]
                research = research_section
        
        # Display research sources and allow exploration
        st.markdown("### Research Sources and Evidence")
        if research:
            display_research_sources(
                research=research,
                allow_filtering=True,
                max_height=500
            )
        else:
            st.markdown("No structured research data available for this forecast.")
        
        # Display reasoning structure
        st.markdown("### Reasoning Structure")
        
        # Create mock reasoning steps from the forecast reasoning
        reasoning_steps = []
        if "##" in reasoning:
            sections = reasoning.split("##")
            for i, section in enumerate(sections[1:], 1):  # Skip the first empty part
                section_title = section.split("\n", 1)[0].strip()
                section_content = section.split("\n", 1)[1].strip() if "\n" in section else ""
                reasoning_steps.append({
                    "title": f"Step {i}",
                    "type": section_title,
                    "content": section_content
                })
        
        if reasoning_steps:
            display_reasoning_steps(
                steps=reasoning_steps,
                expand_all=False
            )

with tab3:
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
            df = df[df["Date"] >= now - timedelta(days=7)]
        elif time_filter == "Last 30 Days":
            df = df[df["Date"] >= now - timedelta(days=30)]
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
        
        # Display grouped data if requested
        if grouping != "None":
            if grouping == "Day":
                grouped = df.groupby(df["Date"].dt.date).agg({
                    "Tokens": "sum",
                    "Cost": "sum",
                    "Question": "count"
                }).reset_index()
                grouped = grouped.rename(columns={"Question": "Count"})
                grouped = grouped.sort_values("Date", ascending=False)
            else:
                grouped = df.groupby(grouping).agg({
                    "Tokens": "sum",
                    "Cost": "sum",
                    "Question": "count"
                }).reset_index()
                grouped = grouped.rename(columns={"Question": "Count"})
                grouped = grouped.sort_values("Cost", ascending=False)
            
            # Display the grouped data
            st.subheader(f"Costs Grouped by {grouping}")
            st.dataframe(grouped, use_container_width=True)
            
            # Create a visualization
            if grouping == "Day":
                chart_df = grouped.copy()
                chart_df["Date"] = chart_df["Date"].astype(str)
                fig = px.bar(
                    chart_df, 
                    x="Date", 
                    y="Cost",
                    title=f"Cost by {grouping}",
                    labels={"Cost": "Cost (USD)", "Date": "Date"},
                    text_auto=".2f"
                )
            else:
                fig = px.bar(
                    grouped, 
                    x=grouping, 
                    y="Cost",
                    title=f"Cost by {grouping}",
                    labels={"Cost": "Cost (USD)"},
                    text_auto=".2f"
                )
            st.plotly_chart(fig, use_container_width=True)
        
        # Show the individual entries
        st.subheader("Individual Costs")
        
        # Format the cost as currency
        df["Cost"] = df["Cost"].apply(lambda x: f"${x:.4f}")
        
        # Format the date
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Display the data
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No cost history available yet.")

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