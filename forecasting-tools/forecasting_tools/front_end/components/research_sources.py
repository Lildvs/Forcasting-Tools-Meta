import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
import re

from forecasting_tools.forecast_bots.reasoning import (
    Evidence, 
    EvidenceType,
    ConfidenceLevel
)

def display_evidence_item(evidence: Evidence, index: int = 0):
    """
    Display a single evidence item.
    
    Args:
        evidence: The evidence object to display
        index: Optional index number for the evidence
    """
    # Determine impact direction icon
    if evidence.impact_direction > 0:
        impact_icon = "✅"
        impact_text = "Supporting"
    elif evidence.impact_direction < 0:
        impact_icon = "❌"
        impact_text = "Contradicting"
    else:
        impact_icon = "⚖️"
        impact_text = "Neutral"
    
    # Determine confidence level styling
    confidence_class = f"confidence-{evidence.confidence.value.replace('_', '-')}"
    
    # Display the evidence in a card-like format
    st.markdown(
        f"""
        <div class="research-source">
            <div class="source-title">{impact_icon} Evidence {index+1}: {evidence.evidence_type.value.replace('_', ' ').title()}</div>
            <p>{evidence.content}</p>
            <div class="source-relevance">
                <span class="confidence-level {confidence_class}">{evidence.confidence.value.replace('_', ' ').title()}</span>
                <span style="margin-left: 10px;">{impact_text} evidence</span>
                {f'<span style="margin-left: 10px;">Source: {evidence.source}</span>' if evidence.source else ''}
                <span style="margin-left: 10px;">Relevance: {int(evidence.relevance_score * 10)}/10</span>
                <span style="margin-left: 10px;">Reliability: {int(evidence.reliability_score * 10)}/10</span>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

def create_evidence_from_text(
    content: str,
    evidence_type: EvidenceType = EvidenceType.FACTUAL,
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
    source: Optional[str] = None,
    relevance_score: float = 0.7,
    reliability_score: float = 0.7,
    impact_direction: int = 0
) -> Evidence:
    """
    Create an Evidence object from text content.
    
    Args:
        content: The content of the evidence
        evidence_type: Type of evidence
        confidence: Confidence level
        source: Source of evidence
        relevance_score: Relevance score (0-1)
        reliability_score: Reliability score (0-1)
        impact_direction: Impact direction (-1, 0, 1)
        
    Returns:
        An Evidence object
    """
    return Evidence(
        content=content,
        evidence_type=evidence_type,
        confidence=confidence,
        source=source,
        relevance_score=relevance_score,
        reliability_score=reliability_score,
        impact_direction=impact_direction
    )

def extract_evidence_from_markdown(markdown_text: str) -> List[Evidence]:
    """
    Extract evidence items from a markdown text.
    
    Args:
        markdown_text: Markdown text containing evidence
        
    Returns:
        List of Evidence objects
    """
    evidence_items = []
    
    # Pattern to detect evidence items
    evidence_pattern = r'(?:^|\n)(?:[-*]\s*|[0-9]+\.\s*)(\b(?:Evidence|Fact|Data\spoint|Statistics?|Source|Reference|Expert\sopinion|Study|Research)(?:\s*\d+)?:?\s*)(.*?)(?=\n(?:[-*]|[0-9]+\.)|$)'
    
    # Search for evidence items
    matches = re.finditer(evidence_pattern, markdown_text, re.IGNORECASE | re.DOTALL)
    
    for match in matches:
        content = match.group(2).strip()
        if content:
            # Simple heuristics for evidence type and confidence
            evidence_type = EvidenceType.FACTUAL  # Default
            if re.search(r'\bstatistic|percent|rate|survey|poll\b', content, re.IGNORECASE):
                evidence_type = EvidenceType.STATISTICAL
            elif re.search(r'\bexpert|opinion|analyst|researcher|scientist\b', content, re.IGNORECASE):
                evidence_type = EvidenceType.EXPERT_OPINION
            elif re.search(r'\bhistor|past|previous|precedent\b', content, re.IGNORECASE):
                evidence_type = EvidenceType.HISTORICAL
            elif re.search(r'\bsimilar|compari|analog\b', content, re.IGNORECASE):
                evidence_type = EvidenceType.ANALOGICAL
            elif re.search(r'\banecdot|example|instance|case\b', content, re.IGNORECASE):
                evidence_type = EvidenceType.ANECDOTAL
            elif re.search(r'\btheor|model|framework|concept\b', content, re.IGNORECASE):
                evidence_type = EvidenceType.THEORETICAL
            elif re.search(r'\babsence|lack|missing|not found\b', content, re.IGNORECASE):
                evidence_type = EvidenceType.ABSENCE_OF_EVIDENCE
            
            # Confidence heuristics
            confidence = ConfidenceLevel.MEDIUM  # Default
            if re.search(r'\bstrong|high|clear|definite|certain|conclusive\b', content, re.IGNORECASE):
                confidence = ConfidenceLevel.HIGH
            elif re.search(r'\bvery\s+strong|overwhelming|undeniable\b', content, re.IGNORECASE):
                confidence = ConfidenceLevel.VERY_HIGH
            elif re.search(r'\bweak|limited|questionable|some\b', content, re.IGNORECASE):
                confidence = ConfidenceLevel.LOW
            elif re.search(r'\bvery\s+weak|minimal|scarce|lack\b', content, re.IGNORECASE):
                confidence = ConfidenceLevel.VERY_LOW
            
            # Impact direction heuristics
            impact = 0  # Default neutral
            if re.search(r'\bsupport|confirm|positive|increase|favor|consistent with\b', content, re.IGNORECASE):
                impact = 1
            elif re.search(r'\bcontradicts?|against|negative|decrease|oppose|inconsistent\b', content, re.IGNORECASE):
                impact = -1
            
            # Source extraction (simple heuristic)
            source_match = re.search(r'\(([^)]+)\)|\[([^\]]+)\]', content)
            source = source_match.group(1) if source_match else None
            
            # Create evidence object
            evidence = create_evidence_from_text(
                content=content,
                evidence_type=evidence_type,
                confidence=confidence,
                source=source,
                impact_direction=impact
            )
            
            evidence_items.append(evidence)
    
    return evidence_items

def display_research_sources(
    research: str, 
    evidences: Optional[List[Evidence]] = None,
    allow_filtering: bool = True,
    max_height: Optional[int] = 500
):
    """
    Display research sources with filtering capabilities.
    
    Args:
        research: Research text
        evidences: Optional list of Evidence objects
        allow_filtering: Whether to allow filtering
        max_height: Maximum height for the research panel
    """
    # Extract evidences from research if not provided
    if evidences is None:
        evidences = extract_evidence_from_markdown(research)
    
    # If no evidences found or extracted, just show the raw research
    if not evidences:
        with st.expander("Research Sources", expanded=False):
            st.markdown(research)
        return
    
    # Display with filtering capabilities
    st.markdown("### Research Sources")
    
    # Evidence filtering capabilities
    if allow_filtering and evidences:
        # Get unique evidence types
        evidence_types = list(set(e.evidence_type for e in evidences))
        
        # Create filter columns
        filter_col1, filter_col2 = st.columns([3, 2])
        
        with filter_col1:
            # Filter by evidence type
            selected_types = st.multiselect(
                "Filter by evidence type",
                options=evidence_types,
                format_func=lambda x: x.value.replace('_', ' ').title(),
                default=evidence_types
            )
        
        with filter_col2:
            # Filter by impact direction
            impact_options = [("Supporting", 1), ("Neutral", 0), ("Contradicting", -1)]
            selected_impacts = st.multiselect(
                "Filter by impact",
                options=[i[1] for i in impact_options],
                format_func=lambda x: next(i[0] for i in impact_options if i[1] == x),
                default=[i[1] for i in impact_options]
            )
        
        # Apply filters
        filtered_evidences = [
            e for e in evidences 
            if e.evidence_type in selected_types and e.impact_direction in selected_impacts
        ]
    else:
        filtered_evidences = evidences
    
    # Display evidence items
    st.markdown(f"Showing {len(filtered_evidences)} of {len(evidences)} evidence items")
    
    # Create scrollable container with fixed height if specified
    if max_height:
        st.markdown(f"""
        <style>
        .evidence-container {{
            max-height: {max_height}px;
            overflow-y: auto;
            padding-right: 10px;
        }}
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="evidence-container">', unsafe_allow_html=True)
    
    # Display evidence items
    for i, evidence in enumerate(filtered_evidences):
        display_evidence_item(evidence, i)
    
    # Close container div if using height limit
    if max_height:
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show raw research
    with st.expander("View Raw Research", expanded=False):
        st.markdown(research)

def display_reasoning_steps(
    steps: List[Dict[str, Any]],
    expand_all: bool = False
):
    """
    Display reasoning steps with progressive disclosure.
    
    Args:
        steps: List of reasoning steps with content and other metadata
        expand_all: Whether to expand all steps by default
    """
    st.markdown("### Reasoning Process")
    
    # Create expandable sections for each step
    for i, step in enumerate(steps):
        # Extract step data
        step_title = step.get("title", f"Step {i+1}")
        step_content = step.get("content", "")
        step_type = step.get("type", "")
        
        # Add step type to title if available
        if step_type:
            title_display = f"{step_title}: {step_type.replace('_', ' ').title()}"
        else:
            title_display = step_title
        
        # Display step in expandable section
        with st.expander(title_display, expanded=expand_all or i < 2):  # Expand first 2 by default
            st.markdown(step_content)
            
            # Display evidences if available
            step_evidences = step.get("evidences", [])
            if step_evidences and isinstance(step_evidences, list):
                st.markdown("**Evidence for this step:**")
                for j, evidence in enumerate(step_evidences):
                    if isinstance(evidence, Evidence):
                        display_evidence_item(evidence, j)
                    else:
                        st.markdown(f"- {evidence}")
            
            # Display intermediate conclusion if available
            conclusion = step.get("conclusion", "")
            if conclusion:
                st.markdown(f"**Intermediate conclusion:** {conclusion}")

def display_biases_and_uncertainties(
    biases: List[str],
    uncertainties: List[str]
):
    """
    Display biases and uncertainties in a structured way.
    
    Args:
        biases: List of bias descriptions
        uncertainties: List of uncertainty descriptions
    """
    cols = st.columns(2)
    
    with cols[0]:
        st.markdown("### Cognitive Biases Considered")
        if biases:
            for bias in biases:
                st.markdown(f"- {bias}")
        else:
            st.markdown("No biases explicitly considered.")
    
    with cols[1]:
        st.markdown("### Key Uncertainties")
        if uncertainties:
            for uncertainty in uncertainties:
                st.markdown(f"- {uncertainty}")
        else:
            st.markdown("No explicit uncertainties documented.") 