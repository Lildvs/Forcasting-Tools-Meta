"""
Thinking Steps

This module provides a structured thinking process for forecasters, guiding them
through a sequence of thinking steps to improve forecast quality and transparency.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    DateQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.reasoning import (
    BayesianReasoning,
    CognitiveBiasMitigation,
    ConfidenceLevel,
    Evidence,
    EvidenceType,
    FermiEstimation,
    ReasoningApproach,
    ReasoningStep,
    StructuredReasoning,
    UncertaintyQuantification,
    create_reasoning_for_question,
)
from forecasting_tools.templates.reasoning_templates import (
    APPROACH_TEMPLATES,
    BAYESIAN_UPDATE_TEMPLATE,
    COGNITIVE_BIAS_TEMPLATE,
    EVIDENCE_EVALUATION_TEMPLATE,
    FERMI_COMPONENT_TEMPLATE,
    UNCERTAINTY_TEMPLATE,
)

logger = logging.getLogger(__name__)


class ThinkingStep(Enum):
    """Enumeration of thinking steps in the forecasting process."""
    
    INITIAL_FRAMING = "initial_framing"
    EVIDENCE_GATHERING = "evidence_gathering"
    EVIDENCE_EVALUATION = "evidence_evaluation"
    APPROACH_SELECTION = "approach_selection"
    BIAS_IDENTIFICATION = "bias_identification"
    STRUCTURED_REASONING = "structured_reasoning"
    UNCERTAINTY_QUANTIFICATION = "uncertainty_quantification"
    FINAL_PREDICTION = "final_prediction"
    REFLECTION = "reflection"


class ThinkingProcess:
    """
    Implements a structured thinking process for forecasting.
    
    This class guides the forecaster through a sequence of thinking steps
    to improve forecast quality and transparency.
    """
    
    def __init__(
        self,
        question: MetaculusQuestion,
        research: str,
        reasoning_approach: Optional[ReasoningApproach] = None,
    ):
        """
        Initialize the thinking process.
        
        Args:
            question: The question to think about
            research: Research material to consider
            reasoning_approach: Optional specific reasoning approach
        """
        self.question = question
        self.research = research
        self.current_step = ThinkingStep.INITIAL_FRAMING
        
        # Create structured reasoning framework
        self.reasoning = create_reasoning_for_question(
            question=question,
            approach=reasoning_approach,
        )
        
        # Initialize state for tracking thinking process
        self.thinking_state: Dict[str, Any] = {
            "evidences": [],
            "biases_considered": [],
            "uncertainties": [],
            "approaches_considered": [],
            "intermediate_conclusions": {},
        }
        
        # Track progress through steps
        self.completed_steps: List[ThinkingStep] = []
        
        logger.info(f"Initialized thinking process for question: {question.question_text}")
    
    def get_current_prompt(self) -> str:
        """
        Get the prompt for the current thinking step.
        
        Returns:
            A prompt to guide the current thinking step
        """
        step_prompts = {
            ThinkingStep.INITIAL_FRAMING: self._get_initial_framing_prompt,
            ThinkingStep.EVIDENCE_GATHERING: self._get_evidence_gathering_prompt,
            ThinkingStep.EVIDENCE_EVALUATION: self._get_evidence_evaluation_prompt,
            ThinkingStep.APPROACH_SELECTION: self._get_approach_selection_prompt,
            ThinkingStep.BIAS_IDENTIFICATION: self._get_bias_identification_prompt,
            ThinkingStep.STRUCTURED_REASONING: self._get_structured_reasoning_prompt,
            ThinkingStep.UNCERTAINTY_QUANTIFICATION: self._get_uncertainty_quantification_prompt,
            ThinkingStep.FINAL_PREDICTION: self._get_final_prediction_prompt,
            ThinkingStep.REFLECTION: self._get_reflection_prompt,
        }
        
        prompt_generator = step_prompts.get(self.current_step)
        if prompt_generator:
            return prompt_generator()
        
        return f"Error: Unknown thinking step {self.current_step}"
    
    def advance_to_next_step(self, step_output: str) -> None:
        """
        Process the current step output and advance to the next step.
        
        Args:
            step_output: Output from the current thinking step
        """
        # Process the current step output
        self._process_step_output(step_output)
        
        # Add current step to completed steps
        self.completed_steps.append(self.current_step)
        
        # Determine next step
        step_order = [
            ThinkingStep.INITIAL_FRAMING,
            ThinkingStep.EVIDENCE_GATHERING,
            ThinkingStep.EVIDENCE_EVALUATION,
            ThinkingStep.APPROACH_SELECTION,
            ThinkingStep.BIAS_IDENTIFICATION,
            ThinkingStep.STRUCTURED_REASONING,
            ThinkingStep.UNCERTAINTY_QUANTIFICATION,
            ThinkingStep.FINAL_PREDICTION,
            ThinkingStep.REFLECTION,
        ]
        
        current_index = step_order.index(self.current_step)
        if current_index < len(step_order) - 1:
            self.current_step = step_order[current_index + 1]
        
        logger.info(f"Advanced to thinking step: {self.current_step.value}")
    
    def _process_step_output(self, step_output: str) -> None:
        """
        Process the output from a thinking step.
        
        Args:
            step_output: Output text from the thinking step
        """
        # Process based on current step
        if self.current_step == ThinkingStep.INITIAL_FRAMING:
            # Extract initial framing information
            self._process_initial_framing(step_output)
        
        elif self.current_step == ThinkingStep.EVIDENCE_GATHERING:
            # Extract evidence items
            self._process_evidence_gathering(step_output)
        
        elif self.current_step == ThinkingStep.EVIDENCE_EVALUATION:
            # Extract evaluated evidence
            self._process_evidence_evaluation(step_output)
        
        elif self.current_step == ThinkingStep.APPROACH_SELECTION:
            # Extract selected reasoning approach
            self._process_approach_selection(step_output)
        
        elif self.current_step == ThinkingStep.BIAS_IDENTIFICATION:
            # Extract identified biases
            self._process_bias_identification(step_output)
        
        elif self.current_step == ThinkingStep.STRUCTURED_REASONING:
            # Extract reasoning steps and conclusions
            self._process_structured_reasoning(step_output)
        
        elif self.current_step == ThinkingStep.UNCERTAINTY_QUANTIFICATION:
            # Extract uncertainty quantification
            self._process_uncertainty_quantification(step_output)
        
        elif self.current_step == ThinkingStep.FINAL_PREDICTION:
            # Extract final prediction
            self._process_final_prediction(step_output)
        
        elif self.current_step == ThinkingStep.REFLECTION:
            # Extract reflection insights
            self._process_reflection(step_output)
    
    def _process_initial_framing(self, output: str) -> None:
        """Process initial framing output."""
        # Add a framing step to the reasoning
        self.reasoning.add_step(
            ReasoningStep(
                content=output,
                step_type="initial_framing",
                confidence=ConfidenceLevel.HIGH,
            )
        )
        
        # Extract any initial hypotheses
        # This is a simplified extraction - a real implementation would parse
        # the output more thoroughly
        if "hypothesis:" in output.lower():
            hypothesis_parts = output.lower().split("hypothesis:")
            if len(hypothesis_parts) > 1:
                hypothesis = hypothesis_parts[1].split("\n")[0].strip()
                self.thinking_state["initial_hypothesis"] = hypothesis
    
    def _process_evidence_gathering(self, output: str) -> None:
        """Process evidence gathering output."""
        # Add a simple evidence gathering step
        self.reasoning.add_step(
            ReasoningStep(
                content=output,
                step_type="evidence_gathering",
                confidence=ConfidenceLevel.MEDIUM,
            )
        )
        
        # Simple evidence extraction - this would be more sophisticated in practice
        evidence_items = []
        
        # Simple pattern matching for evidence extraction
        lines = output.split("\n")
        current_evidence = None
        
        for line in lines:
            if line.strip().startswith("Evidence:") or line.strip().startswith("- Evidence:"):
                if current_evidence:
                    evidence_items.append(current_evidence)
                current_evidence = line.strip().split(":", 1)[1].strip()
            elif current_evidence and line.strip():
                current_evidence += " " + line.strip()
        
        if current_evidence:
            evidence_items.append(current_evidence)
        
        # Store evidence items
        self.thinking_state["raw_evidences"] = evidence_items
    
    def _process_evidence_evaluation(self, output: str) -> None:
        """Process evidence evaluation output."""
        # This would involve more sophisticated parsing in a real implementation
        # Here we'll do a simplified version
        
        # Parse evidences with types and reliability
        evidences = []
        
        # Very simplified parsing
        sections = output.split("Evidence:")
        
        for section in sections[1:]:  # Skip the first split which is before any "Evidence:"
            lines = section.strip().split("\n")
            
            if not lines:
                continue
            
            content = lines[0].strip()
            evidence_type = EvidenceType.FACTUAL  # Default
            confidence = ConfidenceLevel.MEDIUM  # Default
            relevance = 0.5  # Default
            reliability = 0.5  # Default
            source = None
            impact = 0  # Default neutral
            
            for line in lines[1:]:
                if "type:" in line.lower():
                    type_text = line.split(":", 1)[1].strip().lower()
                    
                    # Map text to EvidenceType
                    if "fact" in type_text:
                        evidence_type = EvidenceType.FACTUAL
                    elif "stat" in type_text:
                        evidence_type = EvidenceType.STATISTICAL
                    elif "expert" in type_text or "opinion" in type_text:
                        evidence_type = EvidenceType.EXPERT_OPINION
                    elif "histor" in type_text:
                        evidence_type = EvidenceType.HISTORICAL
                    elif "analog" in type_text or "compar" in type_text:
                        evidence_type = EvidenceType.ANALOGICAL
                    elif "anecdot" in type_text:
                        evidence_type = EvidenceType.ANECDOTAL
                    elif "theor" in type_text:
                        evidence_type = EvidenceType.THEORETICAL
                    elif "absence" in type_text:
                        evidence_type = EvidenceType.ABSENCE_OF_EVIDENCE
                
                elif "confidence:" in line.lower():
                    conf_text = line.split(":", 1)[1].strip().lower()
                    
                    # Map text to ConfidenceLevel
                    if "very high" in conf_text or "very strong" in conf_text:
                        confidence = ConfidenceLevel.VERY_HIGH
                    elif "high" in conf_text or "strong" in conf_text:
                        confidence = ConfidenceLevel.HIGH
                    elif "low" in conf_text or "weak" in conf_text:
                        confidence = ConfidenceLevel.LOW
                    elif "very low" in conf_text or "very weak" in conf_text:
                        confidence = ConfidenceLevel.VERY_LOW
                    else:
                        confidence = ConfidenceLevel.MEDIUM
                
                elif "relevance:" in line.lower():
                    rel_text = line.split(":", 1)[1].strip()
                    # Try to extract a number (if formatted as X/10)
                    if "/" in rel_text:
                        try:
                            num, denom = rel_text.split("/")
                            relevance = float(num.strip()) / float(denom.strip())
                        except (ValueError, ZeroDivisionError):
                            relevance = 0.5
                    else:
                        # Try to parse as direct value
                        try:
                            relevance = float(rel_text.strip()) / 10.0  # Assume out of 10
                        except ValueError:
                            relevance = 0.5
                
                elif "reliability:" in line.lower() or "credibility:" in line.lower():
                    rel_text = line.split(":", 1)[1].strip()
                    # Try to extract a number (if formatted as X/10)
                    if "/" in rel_text:
                        try:
                            num, denom = rel_text.split("/")
                            reliability = float(num.strip()) / float(denom.strip())
                        except (ValueError, ZeroDivisionError):
                            reliability = 0.5
                    else:
                        # Try to parse as direct value
                        try:
                            reliability = float(rel_text.strip()) / 10.0  # Assume out of 10
                        except ValueError:
                            reliability = 0.5
                
                elif "source:" in line.lower():
                    source = line.split(":", 1)[1].strip()
                
                elif "impact:" in line.lower() or "direction:" in line.lower():
                    impact_text = line.split(":", 1)[1].strip().lower()
                    
                    if any(term in impact_text for term in ["positive", "support", "for", "increases"]):
                        impact = 1
                    elif any(term in impact_text for term in ["negative", "against", "decreases", "contradicts"]):
                        impact = -1
                    else:
                        impact = 0
            
            # Create and add the evidence
            evidence = Evidence(
                content=content,
                evidence_type=evidence_type,
                confidence=confidence,
                source=source,
                relevance_score=relevance,
                reliability_score=reliability,
                impact_direction=impact,
            )
            
            evidences.append(evidence)
            
            # Add to reasoning
            self.reasoning.add_evidence(evidence)
        
        # Store processed evidences
        self.thinking_state["evidences"] = evidences
    
    def _process_approach_selection(self, output: str) -> None:
        """Process approach selection output."""
        # Simplified approach selection parsing
        approaches_considered = []
        selected_approach = None
        
        # Look for approach mentions
        for approach in ReasoningApproach:
            approach_name = approach.value.replace("_", " ")
            if approach_name in output.lower():
                approaches_considered.append(approach)
                
                # Check if this approach is selected
                if any(phrase in output.lower() for phrase in [
                    f"select {approach_name}",
                    f"chosen {approach_name}",
                    f"using {approach_name}",
                    f"apply {approach_name}",
                    f"adopt {approach_name}",
                    f"following {approach_name}",
                ]):
                    selected_approach = approach
        
        # If multiple approaches mentioned but none clearly selected, try to infer
        if not selected_approach and approaches_considered:
            # Use the approach that appears most frequently
            approach_counts = {}
            for approach in approaches_considered:
                approach_name = approach.value.replace("_", " ")
                count = output.lower().count(approach_name)
                approach_counts[approach] = count
            
            if approach_counts:
                selected_approach = max(approach_counts, key=approach_counts.get)
        
        # If still no approach selected, keep the existing one
        if selected_approach:
            self.reasoning.approach = selected_approach
        
        # Add a reasoning step
        self.reasoning.add_step(
            ReasoningStep(
                content=output,
                step_type="approach_selection",
                confidence=ConfidenceLevel.HIGH,
                intermediate_conclusion=f"Selected approach: {self.reasoning.approach.value}"
            )
        )
        
        # Store for state
        self.thinking_state["approaches_considered"] = approaches_considered
        self.thinking_state["selected_approach"] = self.reasoning.approach
    
    def _process_bias_identification(self, output: str) -> None:
        """Process bias identification output."""
        # Look for mentioned biases
        identified_biases = []
        
        for bias_name in CognitiveBiasMitigation.COMMON_BIASES:
            readable_name = bias_name.replace("_", " ")
            if readable_name in output.lower():
                identified_biases.append(bias_name)
                
                # Add to the reasoning object
                self.reasoning.add_bias(bias_name)
        
        # Add a reasoning step
        self.reasoning.add_step(
            ReasoningStep(
                content=output,
                step_type="bias_identification",
                confidence=ConfidenceLevel.MEDIUM,
                intermediate_conclusion=f"Identified potential biases: {', '.join(identified_biases)}"
            )
        )
        
        # Store for state
        self.thinking_state["biases_considered"] = identified_biases
    
    def _process_structured_reasoning(self, output: str) -> None:
        """Process structured reasoning output."""
        # This would involve sophisticated parsing of reasoning steps
        # Here we'll do a very simplified version
        
        # Simply add as a reasoning step
        self.reasoning.add_step(
            ReasoningStep(
                content=output,
                step_type="reasoning",
                confidence=ConfidenceLevel.HIGH
            )
        )
        
        # Extract any numerical values or probabilities
        # Very simplified extraction for demonstration
        import re
        
        # Look for probability statements like "probability: 0.75" or "estimate: 75%"
        probability_pattern = r"(?:probability|prob|likelihood|chance|estimate):\s*(\d+(?:\.\d+)?)%?"
        matches = re.findall(probability_pattern, output.lower())
        
        if matches:
            # Convert to float and ensure it's a probability
            try:
                probability = float(matches[-1])  # Use the last match
                if probability > 1 and probability <= 100:
                    probability /= 100  # Convert percentage to decimal
                probability = max(0, min(1, probability))  # Ensure between 0 and 1
                
                self.thinking_state["extracted_probability"] = probability
            except ValueError:
                pass
    
    def _process_uncertainty_quantification(self, output: str) -> None:
        """Process uncertainty quantification output."""
        # Extract uncertainties
        uncertainties = []
        
        # Very simplified parsing
        lines = output.split("\n")
        current_uncertainty = None
        
        for line in lines:
            if line.strip().startswith("Uncertainty:") or line.strip().startswith("- Uncertainty:"):
                if current_uncertainty:
                    uncertainties.append(current_uncertainty)
                current_uncertainty = line.strip().split(":", 1)[1].strip()
            elif current_uncertainty and line.strip():
                current_uncertainty += " " + line.strip()
        
        if current_uncertainty:
            uncertainties.append(current_uncertainty)
        
        # Add to reasoning
        for uncertainty in uncertainties:
            self.reasoning.add_uncertainty(uncertainty)
        
        # Add a reasoning step
        self.reasoning.add_step(
            ReasoningStep(
                content=output,
                step_type="uncertainty_quantification",
                confidence=ConfidenceLevel.MEDIUM,
                intermediate_conclusion=f"Identified {len(uncertainties)} key uncertainties"
            )
        )
        
        # Store for state
        self.thinking_state["uncertainties"] = uncertainties
        
        # Update overall confidence
        self.reasoning.final_confidence = self.reasoning.calculate_overall_confidence()
    
    def _process_final_prediction(self, output: str) -> None:
        """Process final prediction output."""
        # Add a reasoning step
        self.reasoning.add_step(
            ReasoningStep(
                content=output,
                step_type="final_prediction",
                confidence=self.reasoning.final_confidence,
                intermediate_conclusion="Final prediction complete"
            )
        )
        
        # This would extract the final prediction in a structured way
        # For simplicity, we'll just store the raw output
        self.thinking_state["final_prediction_output"] = output
    
    def _process_reflection(self, output: str) -> None:
        """Process reflection output."""
        # Add a reasoning step
        self.reasoning.add_step(
            ReasoningStep(
                content=output,
                step_type="reflection",
                confidence=ConfidenceLevel.MEDIUM,
                intermediate_conclusion="Reflection complete"
            )
        )
        
        # Store reflection output
        self.thinking_state["reflection"] = output
    
    def _get_initial_framing_prompt(self) -> str:
        """Get prompt for initial framing step."""
        return f"""
        # Initial Framing
        
        Please analyze this forecasting question to establish an initial framework:
        
        ## Question:
        {self.question.question_text}
        
        ## Background Information:
        {self.question.background_info or "No background information provided."}
        
        ## Resolution Criteria:
        {self.question.resolution_criteria or "No specific resolution criteria provided."}
        
        In your response, please:
        1. Clarify the key parameters and definitions in the question
        2. Identify what type of prediction is needed
        3. Establish what would count as relevant evidence
        4. Consider any initial hypotheses about the answer
        5. Note any important timelines or constraints mentioned
        
        Provide a structured initial framing of this question to guide further analysis.
        """
    
    def _get_evidence_gathering_prompt(self) -> str:
        """Get prompt for evidence gathering step."""
        return f"""
        # Evidence Gathering
        
        Based on the initial framing, please systematically identify the most relevant evidence from the research materials.
        
        ## Question:
        {self.question.question_text}
        
        ## Initial Framing:
        {self.reasoning.steps[0].content if self.reasoning.steps else "No initial framing available."}
        
        ## Research Materials:
        {self.research[:5000]}  # Truncated for prompt length management
        
        In your response, please:
        1. Identify the most relevant pieces of evidence from the research materials
        2. For each piece of evidence, provide a brief label or description
        3. Include direct quotes or specific data points where possible
        4. Focus on diversity of evidence (different sources, perspectives, types)
        5. Organize the evidence by relevance to the question
        
        Structure your response with clear "Evidence:" labels at the start of each item.
        """
    
    def _get_evidence_evaluation_prompt(self) -> str:
        """Get prompt for evidence evaluation step."""
        # Get raw evidences from thinking state
        raw_evidences = self.thinking_state.get("raw_evidences", [])
        evidences_text = "\n".join([f"- {evidence}" for evidence in raw_evidences])
        
        return f"""
        # Evidence Evaluation
        
        Now, please evaluate each piece of evidence for its reliability, relevance, and impact on the forecast.
        
        ## Question:
        {self.question.question_text}
        
        ## Identified Evidence Items:
        {evidences_text}
        
        For each piece of evidence, please provide:
        1. Type (Factual, Statistical, Expert Opinion, Historical, Analogical, Anecdotal, Theoretical, Absence of Evidence)
        2. Source reliability (scale of 1-10)
        3. Relevance to the question (scale of 1-10)
        4. Direction of impact (supports/neutral/contradicts the hypothesis)
        5. Confidence in the evidence (Very Low, Low, Medium, High, Very High)
        
        Structure your evaluation with "Evidence:" followed by the evidence, then provide the evaluation on separate lines.
        """
    
    def _get_approach_selection_prompt(self) -> str:
        """Get prompt for approach selection step."""
        # Prepare a description of available approaches
        approaches_desc = []
        for approach in ReasoningApproach:
            approach_name = approach.value.replace("_", " ").title()
            if approach == ReasoningApproach.BAYESIAN:
                desc = "Update probabilities based on evidence using Bayes' rule"
            elif approach == ReasoningApproach.FERMI:
                desc = "Break down complex estimates into component factors"
            elif approach == ReasoningApproach.OUTSIDE_VIEW:
                desc = "Use reference classes and historical data for base rates"
            elif approach == ReasoningApproach.INSIDE_VIEW:
                desc = "Model causal mechanisms and specific case factors"
            elif approach == ReasoningApproach.SCOUT_MINDSET:
                desc = "Actively consider multiple perspectives and question assumptions"
            elif approach == ReasoningApproach.COUNTERFACTUAL:
                desc = "Analyze what-if scenarios and alternative histories"
            elif approach == ReasoningApproach.TREND_EXTRAPOLATION:
                desc = "Identify trends and project them forward"
            elif approach == ReasoningApproach.ANALOG_COMPARISON:
                desc = "Compare to similar situations or precedents"
            elif approach == ReasoningApproach.DECOMPOSITION:
                desc = "Break the question into component questions"
            else:
                desc = "General reasoning approach"
            
            approaches_desc.append(f"- {approach_name}: {desc}")
        
        approaches_text = "\n".join(approaches_desc)
        
        return f"""
        # Approach Selection
        
        Based on the question type and evaluated evidence, please select the most appropriate reasoning approach.
        
        ## Question:
        {self.question.question_text}
        
        ## Question Type:
        {type(self.question).__name__}
        
        ## Available Approaches:
        {approaches_text}
        
        In your response, please:
        1. Consider which approaches would be most effective for this type of question
        2. Explain the advantages of different approaches given the available evidence
        3. Identify any limitations of certain approaches for this question
        4. Clearly state your selected approach and justify your choice
        5. Describe how you will apply the chosen approach to this question
        
        Structure your response with a clear analysis of approaches and a definitive selection.
        """
    
    def _get_bias_identification_prompt(self) -> str:
        """Get prompt for bias identification step."""
        # Get commonly relevant biases for this question
        potential_biases = CognitiveBiasMitigation.get_relevant_biases(self.question.question_text)
        
        # Format bias descriptions
        bias_descriptions = []
        for bias in potential_biases:
            description = CognitiveBiasMitigation.COMMON_BIASES.get(bias, "")
            strategy = CognitiveBiasMitigation.get_mitigation_strategy(bias)
            bias_descriptions.append(f"- {bias.replace('_', ' ').title()}: {description}\n  Mitigation: {strategy}")
        
        biases_text = "\n".join(bias_descriptions)
        
        return f"""
        # Cognitive Bias Identification
        
        Please identify potential cognitive biases that could affect the forecast and strategies to mitigate them.
        
        ## Question:
        {self.question.question_text}
        
        ## Selected Approach:
        {self.reasoning.approach.value.replace('_', ' ').title()}
        
        ## Potentially Relevant Biases:
        {biases_text}
        
        In your response, please:
        1. Identify which biases are most likely to affect reasoning on this question
        2. Explain how each bias could specifically manifest in this forecast
        3. Describe concrete strategies to mitigate each identified bias
        4. Consider if there are additional biases beyond those listed that might be relevant
        5. Reflect on which biases might be most problematic given your selected approach
        
        Structure your response with clear identification of biases and specific mitigation strategies.
        """
    
    def _get_structured_reasoning_prompt(self) -> str:
        """Get prompt for structured reasoning step."""
        # Get template for the selected approach
        approach_value = self.reasoning.approach.value
        template = APPROACH_TEMPLATES.get(approach_value, "")
        
        # If no template available, use a generic one
        if not template:
            template = f"""
            # {approach_value.replace('_', ' ').title()} Reasoning Process
            
            ## Step 1: Framing the Question
            {{question_text}}
            
            ## Step 2: Initial Analysis
            
            {{initial_analysis}}
            
            ## Step 3: Evidence Consideration
            
            {{evidence_consideration}}
            
            ## Step 4: Reasoning Steps
            
            {{reasoning_steps}}
            
            ## Step 5: Final Assessment
            
            {{final_assessment}}
            
            ## Step 6: Key Uncertainties
            
            {{uncertainties}}
            """
        
        # Get evidences from thinking state
        evidences = self.thinking_state.get("evidences", [])
        evidences_text = "\n".join([str(evidence) for evidence in evidences])
        
        # Get biases considered
        biases = self.reasoning.considered_biases
        biases_text = "\n".join([f"- {bias.replace('_', ' ').title()}" for bias in biases])
        
        return f"""
        # Structured Reasoning
        
        Please apply the {self.reasoning.approach.value.replace('_', ' ').title()} reasoning approach to forecast this question.
        
        ## Question:
        {self.question.question_text}
        
        ## Background Information:
        {self.question.background_info or "No background information provided."}
        
        ## Resolution Criteria:
        {self.question.resolution_criteria or "No specific resolution criteria provided."}
        
        ## Evidence Summary:
        {evidences_text}
        
        ## Biases to Mitigate:
        {biases_text}
        
        Please use the following template structure for your reasoning:
        
        {template}
        
        Fill in each section with your reasoning, making sure to be explicit about your assumptions, calculations, and logical steps. Include specific evidence where relevant.
        """
    
    def _get_uncertainty_quantification_prompt(self) -> str:
        """Get prompt for uncertainty quantification step."""
        return f"""
        # Uncertainty Quantification
        
        Please identify and quantify the key uncertainties in your forecast.
        
        ## Question:
        {self.question.question_text}
        
        ## Current Reasoning:
        {self.reasoning.steps[-1].content if self.reasoning.steps else "No reasoning available."}
        
        In your response, please:
        1. Identify the top 3-5 specific uncertainties that could affect your forecast
        2. For each uncertainty, estimate its potential impact (low/medium/high)
        3. Describe the direction of impact for each uncertainty
        4. Explain how each uncertainty affects your confidence in the forecast
        5. Consider both known unknowns and potential unknown unknowns
        
        Structure your response with "Uncertainty:" labels for each identified uncertainty.
        """
    
    def _get_final_prediction_prompt(self) -> str:
        """Get prompt for final prediction step."""
        # Adapt based on question type
        if isinstance(self.question, BinaryQuestion):
            prediction_guidance = """
            Provide your final prediction as a probability between 0 and 1 (or 0% to 100%)
            indicating the likelihood that the event will occur.
            """
        elif isinstance(self.question, NumericQuestion):
            prediction_guidance = f"""
            Provide your final prediction as:
            1. A point estimate (median/mean) in {self.question.unit_of_measure or "appropriate units"}
            2. A 90% confidence interval
            3. A probability distribution if possible
            """
        elif isinstance(self.question, MultipleChoiceQuestion):
            options = [f"- {opt.option}" for opt in self.question.options]
            options_text = "\n".join(options)
            prediction_guidance = f"""
            Provide your final prediction as probabilities assigned to each option, ensuring they sum to 100%:
            
            {options_text}
            """
        else:
            prediction_guidance = """
            Provide your final prediction in the most appropriate format for this question.
            """
        
        # Get confidence level
        confidence = self.reasoning.final_confidence.value.replace("_", " ")
        
        return f"""
        # Final Prediction
        
        Based on all previous steps, please provide your final prediction for this question.
        
        ## Question:
        {self.question.question_text}
        
        ## Overall Confidence:
        {confidence}
        
        ## Prediction Guidance:
        {prediction_guidance}
        
        In your response, please:
        1. State your final prediction clearly at the beginning
        2. Explain how you arrived at this specific prediction
        3. Summarize the key pieces of evidence that most influenced your prediction
        4. Explain how you accounted for uncertainties
        5. Indicate your confidence in this prediction
        
        Make your prediction as precise and well-calibrated as possible given the available information.
        """
    
    def _get_reflection_prompt(self) -> str:
        """Get prompt for reflection step."""
        return f"""
        # Reflection
        
        Please reflect on your forecasting process and final prediction.
        
        ## Question:
        {self.question.question_text}
        
        ## Final Prediction:
        {self.thinking_state.get("final_prediction_output", "No final prediction available.")}
        
        In your response, please:
        1. Identify the strengths and limitations of your approach
        2. Consider what additional information would most improve your forecast
        3. Reflect on which cognitive biases might still be affecting your prediction
        4. Identify any weak points in your reasoning
        5. Consider what you would do differently if you started over
        
        This reflection will help improve future forecasting processes and highlight areas where the current forecast might need further consideration.
        """
    
    def get_full_reasoning_summary(self) -> str:
        """
        Generate a full summary of the reasoning process.
        
        Returns:
            Formatted summary of the reasoning process
        """
        return self.reasoning.generate_reasoning_summary()
    
    def get_thinking_state(self) -> Dict[str, Any]:
        """
        Get the current thinking state.
        
        Returns:
            Dictionary with the current thinking state
        """
        return self.thinking_state.copy()
    
    def get_confidence_level(self) -> ConfidenceLevel:
        """
        Get the overall confidence level.
        
        Returns:
            Current confidence level
        """
        return self.reasoning.final_confidence 