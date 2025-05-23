"""
Structured Reasoning Framework

This module provides structured reasoning frameworks for forecasting, including
evidence evaluation, probabilistic reasoning, bias mitigation, and uncertainty
quantification techniques.
"""

import logging
import math
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    DateQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.data_models.forecast_report import (
    ReasonedPrediction,
)

logger = logging.getLogger(__name__)


class ReasoningApproach(Enum):
    """Different reasoning approaches for forecasting."""
    BAYESIAN = "bayesian"
    FERMI = "fermi"
    OUTSIDE_VIEW = "outside_view"
    INSIDE_VIEW = "inside_view"
    SCOUT_MINDSET = "scout_mindset"
    COUNTERFACTUAL = "counterfactual"
    TREND_EXTRAPOLATION = "trend_extrapolation"
    ANALOG_COMPARISON = "analog_comparison"
    DECOMPOSITION = "decomposition"


class EvidenceType(Enum):
    """Types of evidence used in reasoning."""
    FACTUAL = "factual"
    STATISTICAL = "statistical"
    EXPERT_OPINION = "expert_opinion"
    HISTORICAL = "historical"
    ANALOGICAL = "analogical"
    ANECDOTAL = "anecdotal"
    THEORETICAL = "theoretical"
    ABSENCE_OF_EVIDENCE = "absence_of_evidence"


class ConfidenceLevel(Enum):
    """Confidence levels for evidence and conclusions."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class Evidence:
    """Represents a piece of evidence used in reasoning."""
    
    def __init__(
        self,
        content: str,
        evidence_type: EvidenceType,
        confidence: ConfidenceLevel,
        source: Optional[str] = None,
        relevance_score: float = 0.5,
        reliability_score: float = 0.5,
        impact_direction: int = 0,  # -1: against, 0: neutral, 1: for
    ):
        """
        Initialize an evidence object.
        
        Args:
            content: The evidence content/statement
            evidence_type: Type of evidence
            confidence: Confidence level in the evidence
            source: Source of the evidence
            relevance_score: How relevant this evidence is (0-1)
            reliability_score: How reliable this evidence is (0-1)
            impact_direction: Direction of impact (-1, 0, 1)
        """
        self.content = content
        self.evidence_type = evidence_type
        self.confidence = confidence
        self.source = source
        self.relevance_score = max(0.0, min(1.0, relevance_score))
        self.reliability_score = max(0.0, min(1.0, reliability_score))
        self.impact_direction = impact_direction
        
    @property
    def weight(self) -> float:
        """Calculate the weight of this evidence based on relevance and reliability."""
        return self.relevance_score * self.reliability_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "evidence_type": self.evidence_type.value,
            "confidence": self.confidence.value,
            "source": self.source,
            "relevance_score": self.relevance_score,
            "reliability_score": self.reliability_score,
            "impact_direction": self.impact_direction,
            "weight": self.weight,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evidence":
        """Create an Evidence object from a dictionary."""
        return cls(
            content=data["content"],
            evidence_type=EvidenceType(data["evidence_type"]),
            confidence=ConfidenceLevel(data["confidence"]),
            source=data.get("source"),
            relevance_score=data.get("relevance_score", 0.5),
            reliability_score=data.get("reliability_score", 0.5),
            impact_direction=data.get("impact_direction", 0),
        )
    
    def __str__(self) -> str:
        """String representation of evidence."""
        source_str = f" (Source: {self.source})" if self.source else ""
        return f"{self.content}{source_str} [{self.evidence_type.value}, confidence: {self.confidence.value}]"


class ReasoningStep:
    """Represents a single step in a reasoning process."""
    
    def __init__(
        self,
        content: str,
        step_type: str,
        evidences: Optional[List[Evidence]] = None,
        confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
        intermediate_conclusion: Optional[str] = None,
    ):
        """
        Initialize a reasoning step.
        
        Args:
            content: The step content/text
            step_type: Type of reasoning step (e.g., "hypothesis", "evidence", "conclusion")
            evidences: List of evidence used in this step
            confidence: Confidence level in this step
            intermediate_conclusion: Any intermediate conclusion from this step
        """
        self.content = content
        self.step_type = step_type
        self.evidences = evidences or []
        self.confidence = confidence
        self.intermediate_conclusion = intermediate_conclusion
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "step_type": self.step_type,
            "evidences": [e.to_dict() for e in self.evidences],
            "confidence": self.confidence.value,
            "intermediate_conclusion": self.intermediate_conclusion,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningStep":
        """Create a ReasoningStep object from a dictionary."""
        return cls(
            content=data["content"],
            step_type=data["step_type"],
            evidences=[Evidence.from_dict(e) for e in data.get("evidences", [])],
            confidence=ConfidenceLevel(data["confidence"]),
            intermediate_conclusion=data.get("intermediate_conclusion"),
        )


class StructuredReasoning:
    """
    Represents a structured reasoning process with multiple steps.
    
    This class guides the reasoning process through steps like:
    1. Initial framing and hypothesis generation
    2. Evidence gathering and evaluation
    3. Bias identification and mitigation
    4. Application of reasoning approaches (Bayesian, Fermi, etc.)
    5. Uncertainty quantification
    6. Final conclusion and confidence assessment
    """
    
    def __init__(
        self,
        question: MetaculusQuestion,
        approach: ReasoningApproach,
        steps: Optional[List[ReasoningStep]] = None,
        final_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
        considered_biases: Optional[List[str]] = None,
        uncertainties: Optional[List[str]] = None,
    ):
        """
        Initialize structured reasoning.
        
        Args:
            question: The question being reasoned about
            approach: Primary reasoning approach
            steps: List of reasoning steps
            final_confidence: Final confidence level in the conclusion
            considered_biases: List of cognitive biases considered
            uncertainties: List of identified uncertainties
        """
        self.question = question
        self.approach = approach
        self.steps = steps or []
        self.final_confidence = final_confidence
        self.considered_biases = considered_biases or []
        self.uncertainties = uncertainties or []
    
    def add_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step."""
        self.steps.append(step)
    
    def add_evidence(self, evidence: Evidence, step_index: int = -1) -> None:
        """Add evidence to a specific reasoning step."""
        if not self.steps:
            logger.warning("Cannot add evidence: no steps exist")
            return
        
        if step_index == -1:
            step_index = len(self.steps) - 1
        
        if 0 <= step_index < len(self.steps):
            self.steps[step_index].evidences.append(evidence)
        else:
            logger.warning(f"Invalid step index: {step_index}")
    
    def add_bias(self, bias: str) -> None:
        """Record a cognitive bias that was considered."""
        if bias not in self.considered_biases:
            self.considered_biases.append(bias)
    
    def add_uncertainty(self, uncertainty: str) -> None:
        """Record an identified uncertainty."""
        if uncertainty not in self.uncertainties:
            self.uncertainties.append(uncertainty)
    
    def calculate_overall_confidence(self) -> ConfidenceLevel:
        """Calculate the overall confidence level based on reasoning steps."""
        if not self.steps:
            return ConfidenceLevel.MEDIUM
        
        # Map confidence levels to numeric values
        confidence_values = {
            ConfidenceLevel.VERY_LOW: 0.1,
            ConfidenceLevel.LOW: 0.3,
            ConfidenceLevel.MEDIUM: 0.5,
            ConfidenceLevel.HIGH: 0.7,
            ConfidenceLevel.VERY_HIGH: 0.9,
        }
        
        # Calculate weighted average of step confidences
        total_weight = 0
        weighted_sum = 0
        
        for step in self.steps:
            # Later steps have more weight
            step_weight = 1 + (self.steps.index(step) / len(self.steps))
            confidence_value = confidence_values[step.confidence]
            
            weighted_sum += confidence_value * step_weight
            total_weight += step_weight
        
        avg_confidence = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Adjust for number of uncertainties and biases considered
        adjustment = 0
        
        # More uncertainties recognized = lower confidence
        if len(self.uncertainties) > 3:
            adjustment -= 0.1
        
        # More biases considered = higher confidence
        if len(self.considered_biases) > 2:
            adjustment += 0.05
        
        final_value = max(0.1, min(0.9, avg_confidence + adjustment))
        
        # Convert back to ConfidenceLevel
        if final_value < 0.2:
            return ConfidenceLevel.VERY_LOW
        elif final_value < 0.4:
            return ConfidenceLevel.LOW
        elif final_value < 0.6:
            return ConfidenceLevel.MEDIUM
        elif final_value < 0.8:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    def generate_reasoning_summary(self) -> str:
        """Generate a summary of the reasoning process."""
        summary = [f"# Reasoning Summary: {self.question.question_text}"]
        summary.append(f"\n## Approach: {self.approach.value.replace('_', ' ').title()}")
        
        # Add steps
        for i, step in enumerate(self.steps, 1):
            summary.append(f"\n### Step {i}: {step.step_type.title()}")
            summary.append(step.content)
            
            # Add evidence for this step
            if step.evidences:
                summary.append("\n**Evidence:**")
                for evidence in step.evidences:
                    summary.append(f"- {evidence}")
            
            # Add intermediate conclusion if available
            if step.intermediate_conclusion:
                summary.append(f"\n**Intermediate Conclusion:** {step.intermediate_conclusion}")
        
        # Add biases considered
        if self.considered_biases:
            summary.append("\n## Cognitive Biases Considered")
            for bias in self.considered_biases:
                summary.append(f"- {bias}")
        
        # Add uncertainties
        if self.uncertainties:
            summary.append("\n## Key Uncertainties")
            for uncertainty in self.uncertainties:
                summary.append(f"- {uncertainty}")
        
        # Add final confidence
        summary.append(f"\n## Overall Confidence: {self.final_confidence.value.replace('_', ' ').title()}")
        
        return "\n".join(summary)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "question_id": self.question.id_of_post,
            "question_text": self.question.question_text,
            "approach": self.approach.value,
            "steps": [step.to_dict() for step in self.steps],
            "final_confidence": self.final_confidence.value,
            "considered_biases": self.considered_biases,
            "uncertainties": self.uncertainties,
        }


class BayesianReasoning:
    """
    Implementation of Bayesian reasoning for forecasting.
    
    This class helps with applying Bayes' theorem to update probabilities
    based on evidence.
    """
    
    @staticmethod
    def update_probability(
        prior: float,
        likelihood_ratio: float,
    ) -> float:
        """
        Update a probability using Bayes' theorem.
        
        Args:
            prior: Prior probability (0-1)
            likelihood_ratio: Likelihood ratio (P(E|H) / P(E|~H))
            
        Returns:
            Updated posterior probability
        """
        # Ensure valid inputs
        prior = max(0.01, min(0.99, prior))
        likelihood_ratio = max(0.01, likelihood_ratio)
        
        # Calculate posterior using Bayes' theorem
        prior_odds = prior / (1 - prior)
        posterior_odds = prior_odds * likelihood_ratio
        posterior = posterior_odds / (1 + posterior_odds)
        
        return posterior
    
    @staticmethod
    def calculate_likelihood_ratio(
        p_evidence_given_hypothesis: float,
        p_evidence_given_not_hypothesis: float,
    ) -> float:
        """
        Calculate likelihood ratio for Bayesian updating.
        
        Args:
            p_evidence_given_hypothesis: P(E|H)
            p_evidence_given_not_hypothesis: P(E|~H)
            
        Returns:
            Likelihood ratio
        """
        # Ensure valid inputs
        p_evidence_given_hypothesis = max(0.01, min(0.99, p_evidence_given_hypothesis))
        p_evidence_given_not_hypothesis = max(0.01, min(0.99, p_evidence_given_not_hypothesis))
        
        return p_evidence_given_hypothesis / p_evidence_given_not_hypothesis
    
    @staticmethod
    def estimate_likelihood_ratio_from_evidence(
        evidence: Evidence,
    ) -> float:
        """
        Estimate a likelihood ratio based on evidence properties.
        
        Args:
            evidence: The evidence to analyze
            
        Returns:
            Estimated likelihood ratio
        """
        # Base likelihood ratio on evidence weight and impact direction
        base_lr = 1.0
        
        # Adjust based on evidence weight (relevance * reliability)
        weight_factor = 1 + (evidence.weight * 4)  # Maps 0-1 to 1-5
        
        # Adjust direction based on impact
        if evidence.impact_direction > 0:  # Supporting evidence
            lr = weight_factor
        elif evidence.impact_direction < 0:  # Contradicting evidence
            lr = 1 / weight_factor
        else:  # Neutral evidence
            lr = 1.0
        
        # Further adjust based on confidence
        confidence_multipliers = {
            ConfidenceLevel.VERY_LOW: 0.6,
            ConfidenceLevel.LOW: 0.8,
            ConfidenceLevel.MEDIUM: 1.0,
            ConfidenceLevel.HIGH: 1.2,
            ConfidenceLevel.VERY_HIGH: 1.4,
        }
        
        lr *= confidence_multipliers[evidence.confidence]
        
        # Adjust based on evidence type
        type_multipliers = {
            EvidenceType.FACTUAL: 1.3,
            EvidenceType.STATISTICAL: 1.4,
            EvidenceType.EXPERT_OPINION: 1.2,
            EvidenceType.HISTORICAL: 1.25,
            EvidenceType.ANALOGICAL: 1.1,
            EvidenceType.ANECDOTAL: 1.05,
            EvidenceType.THEORETICAL: 1.15,
            EvidenceType.ABSENCE_OF_EVIDENCE: 1.01,
        }
        
        lr *= type_multipliers[evidence.evidence_type]
        
        return lr


class FermiEstimation:
    """
    Implementation of Fermi estimation for numeric forecasting.
    
    This helps with breaking down numeric estimates into component factors
    and calculating confidence intervals.
    """
    
    @staticmethod
    def multiply_estimates(
        estimates: List[Tuple[float, float]],
    ) -> Tuple[float, float]:
        """
        Multiply a series of estimates with uncertainties.
        
        Args:
            estimates: List of (value, uncertainty) pairs
                where uncertainty is fractional (e.g., 0.1 for 10%)
            
        Returns:
            Tuple of (final estimate, combined uncertainty)
        """
        final_value = 1.0
        combined_error_squared = 0.0
        
        for value, uncertainty in estimates:
            final_value *= value
            # Add fractional errors in quadrature
            combined_error_squared += (uncertainty / value) ** 2
        
        combined_uncertainty = final_value * math.sqrt(combined_error_squared)
        
        return (final_value, combined_uncertainty)
    
    @staticmethod
    def confidence_interval(
        mean: float,
        uncertainty: float,
        confidence_level: float = 0.9,
    ) -> Tuple[float, float]:
        """
        Calculate a confidence interval for a Fermi estimate.
        
        Args:
            mean: The mean estimate
            uncertainty: The standard deviation or uncertainty
            confidence_level: Desired confidence level (e.g., 0.9 for 90%)
            
        Returns:
            Tuple of (lower bound, upper bound)
        """
        # We assume a log-normal distribution for Fermi estimates
        if mean <= 0 or uncertainty <= 0:
            return (0, 0)
        
        # Z-scores for common confidence levels
        z_scores = {
            0.5: 0.674,
            0.68: 1.0,
            0.8: 1.282,
            0.9: 1.645,
            0.95: 1.96,
            0.99: 2.576,
        }
        
        # Find closest defined z-score or use 1.645 (90%) by default
        z = z_scores.get(confidence_level, 1.645)
        
        # Calculate confidence interval assuming log-normal distribution
        uncertainty_factor = math.exp(z * math.log(1 + uncertainty / mean))
        
        lower_bound = mean / uncertainty_factor
        upper_bound = mean * uncertainty_factor
        
        return (lower_bound, upper_bound)
    
    @staticmethod
    def combine_independent_estimates(
        estimates: List[Tuple[float, float]],
        weights: Optional[List[float]] = None,
    ) -> Tuple[float, float]:
        """
        Combine multiple independent estimates of the same quantity.
        
        Args:
            estimates: List of (value, uncertainty) pairs
            weights: Optional weights for each estimate
            
        Returns:
            Tuple of (weighted average, combined uncertainty)
        """
        if not estimates:
            return (0, 0)
        
        if weights is None:
            # Default to equal weights
            weights = [1.0] * len(estimates)
        elif len(weights) != len(estimates):
            raise ValueError("Number of weights must match number of estimates")
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return (0, 0)
        
        weights = [w / total_weight for w in weights]
        
        # Calculate weighted average
        weighted_sum = 0
        for (value, _), weight in zip(estimates, weights):
            weighted_sum += value * weight
        
        # Calculate combined uncertainty
        variance_sum = 0
        for (_, uncertainty), weight in zip(estimates, weights):
            variance_sum += (uncertainty * weight) ** 2
        
        combined_uncertainty = math.sqrt(variance_sum)
        
        return (weighted_sum, combined_uncertainty)


class CognitiveBiasMitigation:
    """
    Techniques for identifying and mitigating cognitive biases in forecasting.
    """
    
    # Common cognitive biases in forecasting
    COMMON_BIASES = {
        "anchoring_bias": "Relying too heavily on the first piece of information encountered",
        "availability_bias": "Overestimating the likelihood of events that come easily to mind",
        "confirmation_bias": "Favoring information that confirms existing beliefs",
        "overconfidence_bias": "Being more confident than is justified by the evidence",
        "hindsight_bias": "Believing events were more predictable than they actually were",
        "recency_bias": "Placing too much emphasis on recent events",
        "status_quo_bias": "Preferring things to stay the same",
        "base_rate_neglect": "Focusing on specifics while ignoring general probabilities",
        "representative_bias": "Judging probability by similarity to stereotypes",
        "conjunction_fallacy": "Believing specific conditions are more probable than general ones",
        "planning_fallacy": "Underestimating time, costs, and risks",
        "survivorship_bias": "Focusing on successful cases while ignoring failures",
        "sunk_cost_fallacy": "Continuing based on past investments rather than future returns",
        "framing_effect": "Drawing different conclusions based on how information is presented",
        "illusion_of_control": "Overestimating one's influence over external events",
    }
    
    @classmethod
    def get_relevant_biases(cls, question_text: str) -> List[str]:
        """
        Identify potentially relevant biases for a given question.
        
        Args:
            question_text: The text of the question
            
        Returns:
            List of relevant bias names
        """
        relevant_biases = []
        
        # Check for time-related questions (potential planning fallacy)
        if any(term in question_text.lower() for term in ["when", "date", "time", "by", "deadline"]):
            relevant_biases.append("planning_fallacy")
        
        # Check for comparison questions (potential anchoring)
        if any(term in question_text.lower() for term in ["compared", "relative", "versus", "vs"]):
            relevant_biases.append("anchoring_bias")
        
        # Check for trend questions (potential recency bias)
        if any(term in question_text.lower() for term in ["trend", "increase", "decrease", "change"]):
            relevant_biases.append("recency_bias")
        
        # Check for rare event questions (potential availability bias)
        if any(term in question_text.lower() for term in ["rare", "unlikely", "disaster", "crisis"]):
            relevant_biases.append("availability_bias")
        
        # Check for percentage/probability questions (potential overconfidence)
        if any(term in question_text.lower() for term in ["probability", "chance", "percentage", "likelihood"]):
            relevant_biases.append("overconfidence_bias")
            relevant_biases.append("base_rate_neglect")
        
        # Always consider these common biases
        relevant_biases.extend(["confirmation_bias", "anchoring_bias", "availability_bias"])
        
        # Return unique biases
        return list(set(relevant_biases))
    
    @staticmethod
    def get_mitigation_strategy(bias: str) -> str:
        """
        Get a strategy for mitigating a specific cognitive bias.
        
        Args:
            bias: Name of the bias
            
        Returns:
            Mitigation strategy text
        """
        strategies = {
            "anchoring_bias": "Consider multiple starting points and reference points. Explicitly acknowledge your initial anchor and try different perspectives.",
            "availability_bias": "Seek out systematic data rather than relying on examples that come to mind. Consider base rates and statistical evidence.",
            "confirmation_bias": "Actively seek disconfirming evidence. Consider alternative hypotheses and play devil's advocate with your own beliefs.",
            "overconfidence_bias": "Widen your confidence intervals. Consider what would make your estimate wrong and adjust accordingly.",
            "hindsight_bias": "Document predictions before outcomes are known. Focus on process rather than outcomes.",
            "recency_bias": "Look at longer historical trends. Give equal weight to older and newer evidence based on relevance, not recency.",
            "status_quo_bias": "Explicitly consider possibilities for change. Ask what would need to happen for the status quo to shift.",
            "base_rate_neglect": "Start with the base rate before considering specific evidence. Use Bayesian reasoning to update from base rates.",
            "representative_bias": "Focus on statistical evidence rather than stereotypes. Consider how representative your examples actually are.",
            "conjunction_fallacy": "Remember that specific conditions must be less probable than general ones. Break down complex scenarios into component probabilities.",
            "planning_fallacy": "Use reference class forecasting. Look at similar past events rather than constructing scenarios from scratch.",
            "survivorship_bias": "Seek data about failures as well as successes. Consider the full distribution, not just notable examples.",
            "sunk_cost_fallacy": "Focus on future prospects, not past investments. Evaluate options based on expected future value.",
            "framing_effect": "Reframe the question in multiple ways. Consider both gains and losses, absolutes and percentages.",
            "illusion_of_control": "Focus on external factors outside your control. Consider randomness and structural factors.",
        }
        
        return strategies.get(bias, "Consider multiple perspectives and check your assumptions carefully.")


class UncertaintyQuantification:
    """
    Methods for quantifying and expressing uncertainty in forecasts.
    """
    
    @staticmethod
    def calibrate_confidence_interval(
        point_estimate: float,
        raw_interval: Tuple[float, float],
        historical_calibration: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Calibrate a confidence interval based on historical calibration.
        
        Args:
            point_estimate: The point estimate
            raw_interval: Raw confidence interval (lower, upper)
            historical_calibration: Historical calibration factor (0-1)
                where 1 is perfectly calibrated and <1 means overconfidence
            
        Returns:
            Calibrated confidence interval (lower, upper)
        """
        lower, upper = raw_interval
        
        # Default calibration if not provided (assuming 20% overconfidence)
        calibration = historical_calibration or 0.8
        
        # Calculate raw interval width
        raw_width = upper - lower
        
        # Calculate calibrated width
        calibrated_width = raw_width / calibration
        
        # Calculate adjustment to add to each side
        adjustment = (calibrated_width - raw_width) / 2
        
        # Apply adjustment
        calibrated_lower = lower - adjustment
        calibrated_upper = upper + adjustment
        
        return (calibrated_lower, calibrated_upper)
    
    @staticmethod
    def express_uncertainty(
        mean: float,
        lower_bound: float,
        upper_bound: float,
    ) -> str:
        """
        Generate a standardized expression of uncertainty.
        
        Args:
            mean: Mean estimate
            lower_bound: Lower bound of confidence interval
            upper_bound: Upper bound of confidence interval
            
        Returns:
            Formatted uncertainty expression
        """
        # Calculate relative uncertainty
        relative_uncertainty = (upper_bound - lower_bound) / (2 * mean)
        
        # Format numbers based on magnitude
        def format_number(num):
            if abs(num) < 0.01:
                return f"{num:.6f}"
            elif abs(num) < 1:
                return f"{num:.4f}"
            elif abs(num) < 10:
                return f"{num:.2f}"
            elif abs(num) < 1000:
                return f"{num:.1f}"
            else:
                return f"{int(num):,}"
        
        # Generate uncertainty description based on relative uncertainty
        if relative_uncertainty < 0.1:
            confidence_desc = "high confidence"
        elif relative_uncertainty < 0.25:
            confidence_desc = "moderate confidence"
        elif relative_uncertainty < 0.5:
            confidence_desc = "low confidence"
        else:
            confidence_desc = "very low confidence"
        
        return (
            f"{format_number(mean)} ({format_number(lower_bound)} to "
            f"{format_number(upper_bound)}, {confidence_desc})"
        )
    
    @staticmethod
    def decompose_uncertainty(
        factors: List[Tuple[str, float, float]],
    ) -> List[Tuple[str, float]]:
        """
        Decompose overall uncertainty into contributing factors.
        
        Args:
            factors: List of (factor_name, value, uncertainty) tuples
            
        Returns:
            List of (factor_name, contribution_percentage) tuples
            sorted by contribution
        """
        if not factors:
            return []
        
        # Calculate relative uncertainty for each factor
        relative_uncertainties = []
        for name, value, uncertainty in factors:
            relative_uncertainty = uncertainty / value if value != 0 else 0
            relative_uncertainties.append((name, relative_uncertainty))
        
        # Calculate total relative uncertainty
        total_uncertainty = sum(ru[1] for ru in relative_uncertainties)
        
        if total_uncertainty == 0:
            # Equal weights if no uncertainty
            weight = 1.0 / len(factors)
            return [(name, weight * 100) for name, _ in relative_uncertainties]
        
        # Calculate contribution percentages
        contributions = []
        for name, rel_uncert in relative_uncertainties:
            contribution = (rel_uncert / total_uncertainty) * 100 if total_uncertainty > 0 else 0
            contributions.append((name, contribution))
        
        # Sort by contribution (descending)
        return sorted(contributions, key=lambda x: x[1], reverse=True)


# Factory function to create appropriate reasoning for question type
def create_reasoning_for_question(
    question: MetaculusQuestion,
    approach: Optional[ReasoningApproach] = None,
) -> StructuredReasoning:
    """
    Create an appropriate reasoning framework for a given question.
    
    Args:
        question: The question to reason about
        approach: Optional specific reasoning approach
        
    Returns:
        A StructuredReasoning object configured for the question type
    """
    # Default reasoning approach based on question type
    if approach is None:
        if isinstance(question, BinaryQuestion):
            approach = ReasoningApproach.BAYESIAN
        elif isinstance(question, NumericQuestion):
            approach = ReasoningApproach.FERMI
        elif isinstance(question, MultipleChoiceQuestion):
            approach = ReasoningApproach.ANALOG_COMPARISON
        elif isinstance(question, DateQuestion):
            approach = ReasoningApproach.TREND_EXTRAPOLATION
        else:
            approach = ReasoningApproach.SCOUT_MINDSET
    
    # Create reasoning object
    reasoning = StructuredReasoning(
        question=question,
        approach=approach,
    )
    
    # Add initial steps based on question type
    if isinstance(question, BinaryQuestion):
        reasoning.add_step(
            ReasoningStep(
                content=f"Analyzing binary question: {question.question_text}",
                step_type="framing",
                confidence=ConfidenceLevel.HIGH,
            )
        )
        reasoning.add_step(
            ReasoningStep(
                content="Establishing prior probability based on base rates and initial information",
                step_type="prior_probability",
                confidence=ConfidenceLevel.MEDIUM,
            )
        )
    elif isinstance(question, NumericQuestion):
        reasoning.add_step(
            ReasoningStep(
                content=f"Analyzing numeric question: {question.question_text}",
                step_type="framing",
                confidence=ConfidenceLevel.HIGH,
            )
        )
        reasoning.add_step(
            ReasoningStep(
                content="Breaking down the estimation into component factors",
                step_type="decomposition",
                confidence=ConfidenceLevel.MEDIUM,
            )
        )
    elif isinstance(question, MultipleChoiceQuestion):
        reasoning.add_step(
            ReasoningStep(
                content=f"Analyzing multiple choice question: {question.question_text}",
                step_type="framing",
                confidence=ConfidenceLevel.HIGH,
            )
        )
        reasoning.add_step(
            ReasoningStep(
                content="Establishing relative probabilities for each option",
                step_type="initial_assessment",
                confidence=ConfidenceLevel.MEDIUM,
            )
        )
    
    # Add bias identification step
    biases = CognitiveBiasMitigation.get_relevant_biases(question.question_text)
    reasoning.considered_biases = biases
    
    reasoning.add_step(
        ReasoningStep(
            content=f"Identifying potential cognitive biases: {', '.join(biases)}",
            step_type="bias_identification",
            confidence=ConfidenceLevel.MEDIUM,
        )
    )
    
    return reasoning 