import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Tuple, Union

def create_confidence_gauge(
    confidence_score: float,
    min_score: float = 0.0,
    max_score: float = 1.0,
    height: int = 150,
    width: Optional[int] = None,
    thresholds: Optional[List[Tuple[float, str, str]]] = None
) -> go.Figure:
    """
    Create a gauge chart for confidence visualization.
    
    Args:
        confidence_score: Confidence score (typically 0-1)
        min_score: Minimum value for the gauge
        max_score: Maximum value for the gauge
        height: Chart height
        width: Chart width
        thresholds: Optional list of (threshold, label, color) tuples
        
    Returns:
        Plotly figure object
    """
    # Default thresholds if not provided
    if thresholds is None:
        thresholds = [
            (0.2, "Very Low", "#f8d7da"),
            (0.4, "Low", "#fff3cd"),
            (0.6, "Medium", "#cce5ff"),
            (0.8, "High", "#d4edda"),
            (1.0, "Very High", "#c3e6cb")
        ]
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Confidence Level"},
        gauge={
            "axis": {"range": [min_score, max_score]},
            "bar": {"color": "rgba(0, 104, 201, 0.7)"},
            "steps": [
                {"range": [threshold_min, threshold_max], "color": color}
                for i, (threshold_max, label, color) in enumerate(thresholds)
                for threshold_min in [thresholds[i-1][0] if i > 0 else min_score]
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": confidence_score
            }
        },
        number={
            "suffix": "",
            "valueformat": ".2f"
        }
    ))
    
    # Add annotations for threshold labels
    for threshold, label, color in thresholds:
        # Calculate position for label
        position = (threshold + (
            0 if threshold == thresholds[0][0] else
            thresholds[thresholds.index((threshold, label, color)) - 1][0]
        )) / 2
        
        # Add annotation
        fig.add_annotation(
            x=position,
            y=0.2,
            text=label,
            showarrow=False,
            font={"size": 10},
            xref="x",
            yref="paper"
        )
    
    # Update layout
    fig.update_layout(
        height=height,
        width=width,
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    
    return fig

def display_confidence_breakdown(
    component_scores: Dict[str, float],
    overall_confidence: float,
    strengths: Optional[List[str]] = None,
    weaknesses: Optional[List[str]] = None,
    height: int = 300,
    width: Optional[int] = None
):
    """
    Display a breakdown of confidence components.
    
    Args:
        component_scores: Dictionary of component name to score
        overall_confidence: Overall confidence score
        strengths: Optional list of strength points
        weaknesses: Optional list of weakness points
        height: Chart height
        width: Chart width
    """
    # Create columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display gauge
        fig = create_confidence_gauge(
            confidence_score=overall_confidence,
            height=200
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Display component scores
        st.markdown("##### Confidence Components")
        
        for component, score in component_scores.items():
            # Format component name
            component_name = component.replace("_", " ").title()
            
            # Determine color based on score
            if score >= 0.7:
                color = "green"
            elif score >= 0.4:
                color = "orange"
            else:
                color = "red"
            
            # Calculate width percentage
            width_pct = int(score * 100)
            
            # Create progress bar with label
            st.markdown(
                f"""
                <div style="margin-bottom: 8px;">
                    <div style="font-size: 0.8rem; margin-bottom: 2px;">{component_name}: {score:.2f}</div>
                    <div style="background-color: #e6e9ef; border-radius: 3px; height: 8px; width: 100%;">
                        <div style="background-color: {color}; width: {width_pct}%; height: 100%; border-radius: 3px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Display strengths and weaknesses
    if strengths or weaknesses:
        cols = st.columns(2)
        
        if strengths:
            with cols[0]:
                st.markdown("##### Strengths")
                for strength in strengths:
                    st.markdown(f"✅ {strength}")
        
        if weaknesses:
            with cols[1]:
                st.markdown("##### Weaknesses")
                for weakness in weaknesses:
                    st.markdown(f"⚠️ {weakness}")

def display_calibration_info(
    original_value: float,
    calibrated_value: float,
    historical_calibration: Optional[float] = None,
    is_probability: bool = True
):
    """
    Display information about forecast calibration.
    
    Args:
        original_value: Original forecast value
        calibrated_value: Calibrated forecast value
        historical_calibration: Optional historical calibration factor
        is_probability: Whether the value is a probability (0-1)
    """
    st.markdown("### Forecast Calibration")
    
    format_func = lambda x: f"{x:.1%}" if is_probability else f"{x:.2f}"
    
    # Calculate adjustment percentage
    if original_value != 0:
        adjustment_pct = (calibrated_value - original_value) / abs(original_value) * 100
        adjustment_dir = "upward" if adjustment_pct > 0 else "downward"
    else:
        adjustment_pct = 0
        adjustment_dir = "none"
    
    # Display calibration info
    st.markdown(f"""
    **Original forecast:** {format_func(original_value)}  
    **Calibrated forecast:** {format_func(calibrated_value)} ({adjustment_dir} adjustment of {abs(adjustment_pct):.1f}%)
    """)
    
    if historical_calibration:
        # Explain calibration factor
        st.markdown(f"""
        **Historical calibration factor:** {historical_calibration:.2f}  
        *A calibration factor below 1.0 indicates historical overconfidence.*
        """)
    
    with st.expander("About Calibration", expanded=False):
        st.markdown("""
        Calibration adjusts forecasts to account for common cognitive biases and historical forecast accuracy. 
        Research shows forecasters tend to be overconfident in their predictions. Calibration helps correct 
        for this by widening confidence intervals or moving probability estimates toward 50% in a way that 
        is proportional to the evidence quality and the forecaster's historical accuracy.
        """)

def create_confidence_interval_display(
    mean: float,
    low: float,
    high: float,
    unit: str = "",
    additional_markers: Optional[List[Tuple[str, float, str]]] = None
) -> str:
    """
    Create an HTML representation of a confidence interval.
    
    Args:
        mean: Mean value
        low: Lower bound
        high: Upper bound
        unit: Unit of measurement
        additional_markers: Optional list of (label, value, color) tuples
        
    Returns:
        HTML string for confidence interval
    """
    # Calculate positions
    range_size = high - low
    if range_size <= 0:
        range_size = high  # Fallback if range is invalid
    
    mean_pos = ((mean - low) / range_size) * 100 if range_size > 0 else 50
    mean_pos = max(0, min(100, mean_pos))  # Constrain to 0-100%
    
    # Generate HTML
    html = f"""
    <div class="confidence-interval">
        <div class="confidence-interval-inner" style="left: {0}%; width: {100}%;"></div>
        <div class="confidence-interval-marker" style="left: {mean_pos}%;"></div>
        <div class="confidence-interval-label" style="left: {0}%; bottom: -25px;">{low} {unit}</div>
        <div class="confidence-interval-label" style="left: {100}%; bottom: -25px;">{high} {unit}</div>
        <div class="confidence-interval-label" style="left: {mean_pos}%; top: -25px;">{mean} {unit}</div>
    """
    
    # Add any additional markers
    if additional_markers:
        for label, value, color in additional_markers:
            marker_pos = ((value - low) / range_size) * 100 if range_size > 0 else 50
            marker_pos = max(0, min(100, marker_pos))  # Constrain to 0-100%
            
            html += f"""
            <div class="confidence-interval-marker" style="left: {marker_pos}%; background-color: {color};"></div>
            <div class="confidence-interval-label" style="left: {marker_pos}%; top: -45px; color: {color};">{label}: {value} {unit}</div>
            """
    
    html += "</div>"
    return html 