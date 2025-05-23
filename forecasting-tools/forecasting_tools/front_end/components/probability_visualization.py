import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
import pandas as pd

def create_binary_probability_chart(
    probability: float,
    community_prediction: Optional[float] = None,
    historical_predictions: Optional[List[Tuple[str, float]]] = None,
    height: int = 300,
    width: int = None
) -> go.Figure:
    """
    Create a visualization for binary probability.
    
    Args:
        probability: The forecast probability (0-1)
        community_prediction: Optional community prediction for comparison
        historical_predictions: Optional list of (timestamp, probability) for showing changes
        height: Chart height
        width: Chart width
        
    Returns:
        Plotly figure object
    """
    # Create the base figure
    fig = go.Figure()
    
    # Add the main probability bar
    fig.add_trace(
        go.Bar(
            x=["Forecast"],
            y=[probability],
            text=[f"{probability:.1%}"],
            textposition="auto",
            marker_color="#0068c9",
            name="Forecast Probability",
            hoverinfo="text",
            hovertext=[f"Probability: {probability:.2%}"]
        )
    )
    
    # Add community prediction if available
    if community_prediction is not None:
        fig.add_trace(
            go.Bar(
                x=["Community"],
                y=[community_prediction],
                text=[f"{community_prediction:.1%}"],
                textposition="auto",
                marker_color="#83c9ff",
                name="Community Prediction",
                hoverinfo="text",
                hovertext=[f"Community: {community_prediction:.2%}"]
            )
        )
    
    # Configure layout
    fig.update_layout(
        height=height,
        width=width,
        margin=dict(l=40, r=40, t=40, b=40),
        yaxis=dict(
            title="Probability",
            range=[0, 1],
            tickformat=".0%",
            gridcolor="#e6e9ef"
        ),
        xaxis=dict(
            title="",
            gridcolor="#e6e9ef"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor="white",
    )
    
    return fig

def create_probability_timeline(
    historical_predictions: List[Tuple[str, float]],
    current_probability: float,
    height: int = 300,
    width: int = None
) -> go.Figure:
    """
    Create a visualization showing how probability changed over time.
    
    Args:
        historical_predictions: List of (timestamp, probability) tuples
        current_probability: Current forecast probability
        height: Chart height
        width: Chart width
        
    Returns:
        Plotly figure object
    """
    if not historical_predictions:
        return None
    
    # Ensure the current probability is included
    all_predictions = historical_predictions.copy()
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame(all_predictions, columns=["Timestamp", "Probability"])
    
    # Create the timeline
    fig = px.line(
        df, 
        x="Timestamp", 
        y="Probability",
        markers=True,
    )
    
    # Add the current probability point with different styling
    fig.add_trace(
        go.Scatter(
            x=[df["Timestamp"].iloc[-1] if not df.empty else "Current"],
            y=[current_probability],
            mode="markers",
            marker=dict(
                color="#09ab3b",
                size=12,
                line=dict(
                    color="#ffffff",
                    width=2
                )
            ),
            name="Current Forecast",
            hoverinfo="text",
            hovertext=[f"Current: {current_probability:.2%}"]
        )
    )
    
    # Configure layout
    fig.update_layout(
        height=height,
        width=width,
        margin=dict(l=40, r=40, t=40, b=40),
        yaxis=dict(
            title="Probability",
            range=[0, 1],
            tickformat=".0%",
            gridcolor="#e6e9ef"
        ),
        xaxis=dict(
            title="Time",
            gridcolor="#e6e9ef"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor="white",
    )
    
    return fig

def create_numeric_distribution_chart(
    mean: float,
    low: float,
    high: float,
    percentiles: Optional[List[Tuple[float, float]]] = None,
    community_mean: Optional[float] = None,
    unit: str = "",
    height: int = 300,
    width: int = None
) -> go.Figure:
    """
    Create a visualization for numeric distribution.
    
    Args:
        mean: Mean estimate
        low: Lower bound of confidence interval
        high: Upper bound of confidence interval
        percentiles: Optional list of (value, percentile) tuples for distribution
        community_mean: Optional community prediction mean for comparison
        unit: Unit of measurement
        height: Chart height
        width: Chart width
        
    Returns:
        Plotly figure object
    """
    # Create the base figure
    fig = go.Figure()
    
    # If we have percentiles, create a distribution
    if percentiles and len(percentiles) > 2:
        # Extract values and percentiles
        values = [p[0] for p in percentiles]
        percs = [p[1] for p in percentiles]
        
        # Generate a smooth curve for the distribution
        x_values = np.linspace(min(values), max(values), 100)
        
        # Add the distribution curve
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=[0.5] * len(x_values),  # Center the distribution
                mode="lines",
                line=dict(width=0),
                showlegend=False
            )
        )
        
        # Add the confidence interval as a shaded area
        fig.add_trace(
            go.Scatter(
                x=[low, high],
                y=[0.5, 0.5],
                mode="lines",
                line=dict(width=0),
                fill="toself",
                fillcolor="rgba(0, 104, 201, 0.2)",
                name=f"90% Confidence Interval: {low} - {high} {unit}"
            )
        )
    else:
        # If no percentiles, just show a simple confidence interval
        x_range = np.linspace(low, high, 100)
        y_values = np.zeros(100)
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_values,
                mode="lines",
                line=dict(width=2, color="#0068c9"),
                name=f"Confidence Interval: {low} - {high} {unit}"
            )
        )
    
    # Add the mean as a vertical line
    fig.add_vline(
        x=mean,
        line_width=2,
        line_dash="solid",
        line_color="#09ab3b",
        annotation_text=f"Mean: {mean} {unit}",
        annotation_position="top right"
    )
    
    # Add community mean if available
    if community_mean is not None:
        fig.add_vline(
            x=community_mean,
            line_width=2,
            line_dash="dash",
            line_color="#83c9ff",
            annotation_text=f"Community: {community_mean} {unit}",
            annotation_position="top left"
        )
    
    # Configure layout
    fig.update_layout(
        height=height,
        width=width,
        margin=dict(l=40, r=40, t=40, b=40),
        yaxis=dict(
            title="",
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        xaxis=dict(
            title=f"Value ({unit})" if unit else "Value",
            gridcolor="#e6e9ef"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor="white",
    )
    
    return fig

def display_binary_forecast(
    probability: float,
    reasoning: str,
    community_prediction: Optional[float] = None,
    historical_predictions: Optional[List[Tuple[str, float]]] = None,
    confidence_level: Optional[str] = None,
    show_timeline: bool = True
):
    """
    Display a binary forecast with visualization.
    
    Args:
        probability: The forecast probability (0-1)
        reasoning: Reasoning text
        community_prediction: Optional community prediction
        historical_predictions: Optional historical predictions
        confidence_level: Optional confidence level
        show_timeline: Whether to show timeline of changes
    """
    # Create columns for layout
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Display probability chart
        fig = create_binary_probability_chart(
            probability=probability,
            community_prediction=community_prediction
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display confidence level if available
        if confidence_level:
            confidence_class = f"confidence-{confidence_level.lower().replace(' ', '-')}"
            st.markdown(
                f"<div class='confidence-level {confidence_class}'>{confidence_level}</div>",
                unsafe_allow_html=True
            )
    
    with col2:
        # Display reasoning
        st.markdown("### Reasoning")
        
        # Only show first 3 paragraphs initially
        paragraphs = reasoning.split('\n\n')
        if len(paragraphs) > 3:
            st.markdown('\n\n'.join(paragraphs[:3]))
            with st.expander("Show full reasoning"):
                st.markdown('\n\n'.join(paragraphs[3:]))
        else:
            st.markdown(reasoning)
    
    # Show timeline if requested and available
    if show_timeline and historical_predictions and len(historical_predictions) > 1:
        st.markdown("### Forecast Changes Over Time")
        timeline_fig = create_probability_timeline(
            historical_predictions=historical_predictions,
            current_probability=probability
        )
        st.plotly_chart(timeline_fig, use_container_width=True)

def display_numeric_forecast(
    mean: float,
    low: float,
    high: float,
    reasoning: str,
    unit: str = "",
    percentiles: Optional[List[Tuple[float, float]]] = None,
    community_mean: Optional[float] = None,
    confidence_level: Optional[str] = None
):
    """
    Display a numeric forecast with visualization.
    
    Args:
        mean: Mean estimate
        low: Lower bound of confidence interval
        high: Upper bound of confidence interval
        reasoning: Reasoning text
        unit: Unit of measurement
        percentiles: Optional percentiles for distribution
        community_mean: Optional community mean prediction
        confidence_level: Optional confidence level
    """
    # Create columns for layout
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Display numeric distribution
        fig = create_numeric_distribution_chart(
            mean=mean,
            low=low,
            high=high,
            percentiles=percentiles,
            community_mean=community_mean,
            unit=unit
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the numeric values
        st.markdown(f"**Mean estimate:** {mean} {unit}")
        st.markdown(f"**90% confidence interval:** {low} - {high} {unit}")
        
        # Display confidence level if available
        if confidence_level:
            confidence_class = f"confidence-{confidence_level.lower().replace(' ', '-')}"
            st.markdown(
                f"<div class='confidence-level {confidence_class}'>{confidence_level}</div>",
                unsafe_allow_html=True
            )
    
    with col2:
        # Display reasoning
        st.markdown("### Reasoning")
        
        # Only show first 3 paragraphs initially
        paragraphs = reasoning.split('\n\n')
        if len(paragraphs) > 3:
            st.markdown('\n\n'.join(paragraphs[:3]))
            with st.expander("Show full reasoning"):
                st.markdown('\n\n'.join(paragraphs[3:]))
        else:
            st.markdown(reasoning) 