import logging
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel

from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.config import PersonalityConfig
from forecasting_tools.personality_management.diversity import PersonalityDiversityScorer
from forecasting_tools.forecast_helpers.competition import CompetitionTracker, CompetitionMetric
from forecasting_tools.front_end.helpers.tool_page import ToolPage
from forecasting_tools.front_end.helpers.personality_details import PersonalityDetails
from forecasting_tools.util.jsonable import Jsonable

logger = logging.getLogger(__name__)


class AnalyticsInput(Jsonable, BaseModel):
    personality_names: List[str]
    domains: List[str]
    metrics: List[str]
    time_period: str


class AnalyticsOutput(Jsonable, BaseModel):
    results: Dict[str, Any]
    timestamp: str


class PersonalityAnalyticsPage(ToolPage):
    PAGE_DISPLAY_NAME: str = "📊 Personality Analytics"
    URL_PATH: str = "/personality-analytics"
    INPUT_TYPE = AnalyticsInput
    OUTPUT_TYPE = AnalyticsOutput
    EXAMPLES_FILE_PATH = None

    @classmethod
    async def _display_intro_text(cls) -> None:
        st.markdown("""
        # Personality Analytics Dashboard
        
        This dashboard provides comprehensive analytics on personality performance across different domains,
        question types, and time periods. Use these insights to optimize your forecasting strategy.
        """)
        
        # Create tabs for different analytics views
        tab1, tab2, tab3, tab4 = st.tabs([
            "Performance Overview", 
            "Domain Analysis", 
            "Personality Comparison",
            "Forecast History"
        ])
        
        with tab1:
            await cls._display_performance_overview()
        
        with tab2:
            await cls._display_domain_analysis()
        
        with tab3:
            await cls._display_personality_comparison()
        
        with tab4:
            await cls._display_forecast_history()
    
    @classmethod
    async def _display_performance_overview(cls) -> None:
        """Display an overview of personality performance."""
        st.subheader("Performance Overview")
        
        # Get personality manager
        personality_manager = PersonalityManager()
        available_personalities = personality_manager.list_available_personalities()
        
        # Time period selector
        time_periods = ["Last 7 days", "Last 30 days", "Last 3 months", "All time"]
        selected_period = st.selectbox("Time Period", options=time_periods, index=2)
        
        # Create metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Forecasts",
                value="387",
                delta="28",
                delta_color="normal",
                help="Total number of forecasts made by all personalities"
            )
        
        with col2:
            st.metric(
                label="Average Accuracy",
                value="76%",
                delta="3%",
                delta_color="normal",
                help="Average accuracy across all personalities"
            )
        
        with col3:
            st.metric(
                label="Best Performer",
                value="Bayesian",
                help="Personality with the highest overall performance"
            )
        
        with col4:
            st.metric(
                label="Most Improved",
                value="Creative",
                delta="12%",
                delta_color="normal",
                help="Personality with the most improvement in performance"
            )
        
        # Create performance chart
        st.subheader("Overall Performance by Personality")
        
        # Generate mock data
        personalities = available_personalities[:5] if len(available_personalities) >= 5 else available_personalities
        metrics = ["Accuracy", "Calibration", "Information Score", "Expected Score"]
        
        data = []
        for personality in personalities:
            row = {"Personality": personality}
            
            # Generate different base values based on personality type
            if "bayesian" in personality.lower():
                base_values = [0.78, 0.85, 0.65, 0.76]
            elif "economist" in personality.lower():
                base_values = [0.75, 0.80, 0.70, 0.75]
            elif "creative" in personality.lower():
                base_values = [0.70, 0.72, 0.80, 0.74]
            elif "cautious" in personality.lower():
                base_values = [0.76, 0.88, 0.60, 0.75]
            else:
                base_values = [0.74, 0.78, 0.70, 0.74]
                
            # Add some randomness
            values = [min(0.95, max(0.5, v + np.random.normal(0, 0.03))) for v in base_values]
            
            for i, metric in enumerate(metrics):
                row[metric] = values[i]
            
            data.append(row)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create plotly bar chart
        fig = px.bar(
            df,
            x="Personality",
            y=metrics,
            barmode="group",
            title="Performance Metrics by Personality",
            labels={"value": "Score", "variable": "Metric"},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        fig.update_layout(
            xaxis_title="Personality",
            yaxis_title="Score",
            legend_title="Metric",
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add performance trend over time
        st.subheader("Performance Trend Over Time")
        
        # Generate mock time series data
        num_days = 90
        dates = [datetime.now() - timedelta(days=i) for i in range(num_days)]
        dates.reverse()
        
        time_data = []
        for personality in personalities:
            # Generate base trend
            if "bayesian" in personality.lower():
                base = 0.75
                slope = 0.0005
            elif "economist" in personality.lower():
                base = 0.72
                slope = 0.0003
            elif "creative" in personality.lower():
                base = 0.65
                slope = 0.001
            elif "cautious" in personality.lower():
                base = 0.73
                slope = 0.0002
            else:
                base = 0.70
                slope = 0.0004
            
            # Generate scores with trend and noise
            for i, date in enumerate(dates):
                score = base + (slope * i) + np.random.normal(0, 0.03)
                score = min(0.95, max(0.5, score))
                
                time_data.append({
                    "Date": date,
                    "Personality": personality,
                    "Score": score
                })
        
        # Convert to DataFrame
        time_df = pd.DataFrame(time_data)
        
        # Create plotly line chart
        fig = px.line(
            time_df,
            x="Date",
            y="Score",
            color="Personality",
            title="Performance Trend Over Time",
            labels={"Score": "Accuracy Score"},
            line_shape="spline"
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Accuracy Score",
            legend_title="Personality",
            yaxis=dict(range=[0.5, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show notes
        st.info("Note: The data shown is simulated for demonstration purposes.")
    
    @classmethod
    async def _display_domain_analysis(cls) -> None:
        """Display domain-specific performance analysis."""
        st.subheader("Domain Analysis")
        
        # Get personality manager
        personality_manager = PersonalityManager()
        available_personalities = personality_manager.list_available_personalities()
        
        # Domain selector
        domains = [
            "Economics", "Finance", "Politics", "Technology", 
            "Science", "Health", "Sports", "Entertainment",
            "Geopolitics", "Environment", "Energy", "Social"
        ]
        
        selected_domains = st.multiselect(
            "Select Domains to Analyze",
            options=domains,
            default=domains[:4]
        )
        
        if not selected_domains:
            st.warning("Please select at least one domain to analyze.")
            return
        
        # Create heatmap of domain performance
        st.subheader("Domain-Specific Performance")
        
        # Generate mock data
        personalities = available_personalities[:5] if len(available_personalities) >= 5 else available_personalities
        
        # Initialize data with domains
        domain_data = []
        
        for personality in personalities:
            # Generate scores for each domain
            row = {"Personality": personality}
            
            for domain in selected_domains:
                # Generate domain-specific scores based on personality-domain affinities
                if domain == "Economics" or domain == "Finance":
                    if "economist" in personality.lower():
                        base = 0.85
                    elif "bayesian" in personality.lower():
                        base = 0.80
                    else:
                        base = 0.70
                elif domain == "Politics" or domain == "Geopolitics":
                    if "bayesian" in personality.lower():
                        base = 0.82
                    elif "creative" in personality.lower():
                        base = 0.78
                    else:
                        base = 0.72
                elif domain == "Technology" or domain == "Science":
                    if "bayesian" in personality.lower():
                        base = 0.83
                    elif "creative" in personality.lower():
                        base = 0.80
                    else:
                        base = 0.75
                else:
                    base = 0.75
                
                # Add some randomness
                score = min(0.95, max(0.5, base + np.random.normal(0, 0.05)))
                row[domain] = score
            
            domain_data.append(row)
        
        # Convert to DataFrame
        domain_df = pd.DataFrame(domain_data)
        
        # Create heatmap
        heatmap_data = domain_df.set_index("Personality")
        
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Domain", y="Personality", color="Score"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale="YlGnBu",
            title="Performance by Domain and Personality"
        )
        
        fig.update_layout(
            xaxis_title="Domain",
            yaxis_title="Personality",
            coloraxis_colorbar=dict(title="Score")
        )
        
        # Add text annotations
        for i, personality in enumerate(heatmap_data.index):
            for j, domain in enumerate(heatmap_data.columns):
                fig.add_annotation(
                    x=domain,
                    y=personality,
                    text=f"{heatmap_data.loc[personality, domain]:.2f}",
                    showarrow=False,
                    font=dict(color="black")
                )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Domain recommendations
        st.subheader("Domain-Specific Recommendations")
        
        for domain in selected_domains:
            # Find best personalities for this domain
            domain_scores = [(p, domain_df.loc[domain_df["Personality"] == p, domain].values[0]) 
                             for p in personalities]
            domain_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get top recommendations
            top_recommendations = domain_scores[:2]
            
            st.markdown(f"**{domain}:** Best personalities are {top_recommendations[0][0]} ({top_recommendations[0][1]:.2f}) and {top_recommendations[1][0]} ({top_recommendations[1][1]:.2f})")
        
        # Personality strengths
        st.subheader("Personality Domain Strengths")
        
        for personality in personalities:
            # Find best domains for this personality
            row = domain_df.loc[domain_df["Personality"] == personality].iloc[0]
            domain_scores = [(domain, row[domain]) for domain in selected_domains]
            domain_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get top domains
            top_domains = domain_scores[:3]
            
            st.markdown(f"**{personality}** excels in: {', '.join([f'{d[0]} ({d[1]:.2f})' for d in top_domains])}")
        
        # Show notes
        st.info("Note: The data shown is simulated for demonstration purposes.")
    
    @classmethod
    async def _display_personality_comparison(cls) -> None:
        """Display detailed personality comparison."""
        st.subheader("Personality Comparison")
        
        # Get personality manager
        personality_manager = PersonalityManager()
        available_personalities = personality_manager.list_available_personalities()
        
        # Personality selector
        selected_personalities = st.multiselect(
            "Select Personalities to Compare",
            options=available_personalities,
            default=available_personalities[:3] if len(available_personalities) >= 3 else available_personalities
        )
        
        if not selected_personalities:
            st.warning("Please select at least one personality to analyze.")
            return
        
        # Metric selector
        metrics = ["Accuracy", "Calibration", "Information Score", "Expected Score"]
        selected_metric = st.selectbox("Select Performance Metric", options=metrics)
        
        # Create radar chart for question types
        st.subheader("Performance by Question Type")
        
        # Question types
        question_types = ["Binary", "Numeric", "Multiple Choice", "Date"]
        
        # Generate mock data
        question_type_data = []
        
        for personality in selected_personalities:
            # Generate scores for each question type
            values = []
            
            for question_type in question_types:
                # Generate question type specific scores based on personality
                if question_type == "Binary":
                    if "bayesian" in personality.lower():
                        base = 0.83
                    elif "economist" in personality.lower():
                        base = 0.78
                    else:
                        base = 0.75
                elif question_type == "Numeric":
                    if "economist" in personality.lower():
                        base = 0.82
                    elif "bayesian" in personality.lower():
                        base = 0.85
                    else:
                        base = 0.73
                elif question_type == "Multiple Choice":
                    if "creative" in personality.lower():
                        base = 0.80
                    else:
                        base = 0.75
                else:  # Date
                    if "bayesian" in personality.lower():
                        base = 0.78
                    else:
                        base = 0.72
                
                # Add some randomness
                score = min(0.95, max(0.5, base + np.random.normal(0, 0.03)))
                values.append(score)
            
            # Create radar chart data
            question_type_data.append({
                "Personality": personality,
                "Values": values
            })
        
        # Create plotly radar chart
        fig = go.Figure()
        
        for data in question_type_data:
            # Add trace for each personality
            fig.add_trace(go.Scatterpolar(
                r=data["Values"] + [data["Values"][0]],  # Close the polygon
                theta=question_types + [question_types[0]],  # Close the polygon
                fill="toself",
                name=data["Personality"]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0.5, 1]
                )
            ),
            showlegend=True,
            title="Performance by Question Type"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create comparison by time horizon
        st.subheader("Performance by Time Horizon")
        
        # Time horizons
        time_horizons = ["Short-term (< 1 month)", "Medium-term (1-6 months)", "Long-term (> 6 months)"]
        
        # Generate mock data
        horizon_data = []
        
        for personality in selected_personalities:
            row = {"Personality": personality}
            
            # Generate horizon-specific scores
            for horizon in time_horizons:
                if horizon == "Short-term (< 1 month)":
                    if "economist" in personality.lower():
                        base = 0.82
                    else:
                        base = 0.75
                elif horizon == "Medium-term (1-6 months)":
                    if "bayesian" in personality.lower():
                        base = 0.80
                    else:
                        base = 0.73
                else:  # Long-term
                    if "creative" in personality.lower():
                        base = 0.78
                    elif "bayesian" in personality.lower():
                        base = 0.75
                    else:
                        base = 0.70
                
                # Add some randomness
                score = min(0.95, max(0.5, base + np.random.normal(0, 0.03)))
                row[horizon] = score
            
            horizon_data.append(row)
        
        # Convert to DataFrame
        horizon_df = pd.DataFrame(horizon_data)
        
        # Create plotly bar chart
        fig = px.bar(
            horizon_df,
            x="Personality",
            y=time_horizons,
            barmode="group",
            title="Performance by Time Horizon",
            labels={"value": "Score", "variable": "Time Horizon"},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        fig.update_layout(
            xaxis_title="Personality",
            yaxis_title="Score",
            legend_title="Time Horizon",
            yaxis=dict(range=[0.5, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparative strengths and weaknesses
        st.subheader("Strengths and Weaknesses Analysis")
        
        for personality in selected_personalities:
            st.markdown(f"#### {personality}")
            
            # Generate mock strengths and weaknesses
            if "bayesian" in personality.lower():
                strengths = ["Strong calibration", "Excels at complex questions", "Good with numeric forecasts"]
                weaknesses = ["Slower for time-sensitive questions", "May be overly conservative in novel domains"]
            elif "economist" in personality.lower():
                strengths = ["Best in economics/finance domains", "Strong with quantitative analysis", "Good short-term forecasting"]
                weaknesses = ["Weaker in non-economic domains", "Less effective with qualitative questions"]
            elif "creative" in personality.lower():
                strengths = ["Identifies novel scenarios", "Strong in technology forecasting", "Good with multiple choice questions"]
                weaknesses = ["May be overconfident", "Less consistent calibration"]
            elif "cautious" in personality.lower():
                strengths = ["Well-calibrated", "Reliable in uncertain domains", "Consistent performance"]
                weaknesses = ["May miss tail risks", "Conservative predictions", "Slower improvement over time"]
            else:
                strengths = ["Well-rounded performance", "Adaptable across domains", "Consistent results"]
                weaknesses = ["No standout strengths", "Medium performance ceiling"]
            
            # Create columns for strengths and weaknesses
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Strengths:**")
                for strength in strengths:
                    st.markdown(f"- {strength}")
            
            with col2:
                st.markdown("**Weaknesses:**")
                for weakness in weaknesses:
                    st.markdown(f"- {weakness}")
        
        # Show notes
        st.info("Note: The data shown is simulated for demonstration purposes.")
    
    @classmethod
    async def _display_forecast_history(cls) -> None:
        """Display forecast history and trends."""
        st.subheader("Forecast History")
        
        # Get personality manager
        personality_manager = PersonalityManager()
        available_personalities = personality_manager.list_available_personalities()
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_personality = st.selectbox(
                "Filter by Personality",
                options=["All"] + available_personalities,
                index=0
            )
        
        with col2:
            domains = ["All", "Economics", "Finance", "Politics", "Technology", "Science", "Health"]
            selected_domain = st.selectbox("Filter by Domain", options=domains, index=0)
        
        with col3:
            time_periods = ["Last 7 days", "Last 30 days", "Last 3 months", "All time"]
            selected_period = st.selectbox("Time Period", options=time_periods, index=2)
        
        # Generate mock forecast history data
        num_forecasts = 50
        forecast_data = []
        
        for i in range(num_forecasts):
            # Random personality
            if selected_personality == "All":
                personality = np.random.choice(available_personalities)
            else:
                personality = selected_personality
            
            # Random domain
            if selected_domain == "All":
                domain = np.random.choice(domains[1:])
            else:
                domain = selected_domain
            
            # Random date within selected period
            if selected_period == "Last 7 days":
                days_ago = np.random.randint(0, 7)
            elif selected_period == "Last 30 days":
                days_ago = np.random.randint(0, 30)
            elif selected_period == "Last 3 months":
                days_ago = np.random.randint(0, 90)
            else:
                days_ago = np.random.randint(0, 180)
                
            date = datetime.now() - timedelta(days=days_ago)
            
            # Generate question and prediction
            questions = [
                "Will inflation exceed 3% in 2024?",
                "Will Party X win the next election?",
                "Will Company Y release Product Z before the end of the year?",
                "Will tensions between Countries A and B escalate to military conflict?",
                "Will the new AI regulation bill pass in 2024?",
                "Will unemployment rates drop below 4% in the next quarter?",
                "Will the central bank lower interest rates in the next meeting?",
                "Will Country X achieve its climate goals by 2025?"
            ]
            
            question = np.random.choice(questions)
            prediction = round(np.random.uniform(0.1, 0.9), 2)
            
            # Generate resolution if applicable
            if days_ago > 60:
                resolution = np.random.choice([True, False])
                # Calculate accuracy
                accuracy = 1 - abs(int(resolution) - prediction)
            else:
                resolution = None
                accuracy = None
            
            forecast_data.append({
                "Date": date,
                "Personality": personality,
                "Domain": domain,
                "Question": question,
                "Prediction": prediction,
                "Resolution": resolution,
                "Accuracy": accuracy
            })
        
        # Sort by date (newest first)
        forecast_data.sort(key=lambda x: x["Date"], reverse=True)
        
        # Convert to DataFrame
        history_df = pd.DataFrame(forecast_data)
        
        # Display data table
        st.dataframe(
            history_df,
            use_container_width=True,
            column_config={
                "Date": st.column_config.DatetimeColumn("Date", format="MMM DD, YYYY"),
                "Prediction": st.column_config.ProgressColumn("Prediction", format="%.0f%%", min_value=0, max_value=1),
                "Accuracy": st.column_config.ProgressColumn("Accuracy", format="%.0f%%", min_value=0, max_value=1)
            }
        )
        
        # Display accuracy trends
        resolved_forecasts = history_df[history_df["Resolution"].notnull()]
        
        if not resolved_forecasts.empty:
            st.subheader("Accuracy Trends")
            
            # Group by personality and calculate average accuracy
            if selected_personality == "All":
                personality_accuracy = resolved_forecasts.groupby("Personality")["Accuracy"].mean().reset_index()
                
                # Create bar chart
                fig = px.bar(
                    personality_accuracy,
                    x="Personality",
                    y="Accuracy",
                    title="Average Accuracy by Personality",
                    labels={"Accuracy": "Average Accuracy"},
                    color="Personality"
                )
                
                fig.update_layout(
                    xaxis_title="Personality",
                    yaxis_title="Accuracy",
                    yaxis=dict(range=[0, 1])
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Group by domain and calculate average accuracy
            if selected_domain == "All":
                domain_accuracy = resolved_forecasts.groupby("Domain")["Accuracy"].mean().reset_index()
                
                # Create bar chart
                fig = px.bar(
                    domain_accuracy,
                    x="Domain",
                    y="Accuracy",
                    title="Average Accuracy by Domain",
                    labels={"Accuracy": "Average Accuracy"},
                    color="Domain"
                )
                
                fig.update_layout(
                    xaxis_title="Domain",
                    yaxis_title="Accuracy",
                    yaxis=dict(range=[0, 1])
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Show notes
        st.info("Note: The forecast history data shown is simulated for demonstration purposes.")

    @classmethod
    async def _get_input(cls) -> AnalyticsInput | None:
        # Input is handled in the _display_intro_text method
        return None

    @classmethod
    async def _run_tool(cls, input: AnalyticsInput) -> AnalyticsOutput:
        # Tool functionality is handled in the _display_intro_text method
        return AnalyticsOutput(
            results={},
            timestamp=datetime.now().isoformat()
        )

    @classmethod
    async def _save_run_to_coda(
        cls,
        input_to_tool: AnalyticsInput,
        output: AnalyticsOutput,
        is_premade: bool,
    ) -> None:
        # No need to save to database for this tool
        pass

    @classmethod
    async def _display_outputs(cls, outputs: list[AnalyticsOutput]) -> None:
        # Output is handled in the _display_intro_text method
        pass


if __name__ == "__main__":
    PersonalityAnalyticsPage.main() 