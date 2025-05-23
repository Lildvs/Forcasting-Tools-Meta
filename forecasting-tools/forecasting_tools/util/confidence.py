"""
Confidence Scoring

This module provides mechanisms for evaluating the quality, relevance, and
reliability of information used in forecasting, producing confidence scores
that help calibrate forecasts.
"""

import logging
import math
import re
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Set

from forecasting_tools.forecast_bots.reasoning import (
    ConfidenceLevel,
    Evidence,
    EvidenceType,
)

logger = logging.getLogger(__name__)


class InformationQuality(Enum):
    """Levels of information quality for sources."""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9


class SourceType(Enum):
    """Types of information sources with associated base reliability scores."""
    PEER_REVIEWED_RESEARCH = (0.9, "Peer-reviewed academic research")
    GOVERNMENT_STATISTICS = (0.85, "Official government statistics")
    EXPERT_ANALYSIS = (0.8, "Expert analysis from recognized authorities")
    NEWS_MAINSTREAM = (0.7, "Mainstream news sources")
    NEWS_SPECIALTY = (0.65, "Specialty news publications")
    INDUSTRY_REPORT = (0.75, "Industry reports and white papers")
    COMPANY_STATEMENT = (0.6, "Company statements and press releases")
    PREPRINT_RESEARCH = (0.65, "Non-peer-reviewed preprint research")
    SOCIAL_MEDIA_VERIFIED = (0.5, "Verified social media accounts")
    SOCIAL_MEDIA_UNVERIFIED = (0.2, "Unverified social media accounts")
    BLOG_EXPERT = (0.6, "Expert blogs and columns")
    BLOG_GENERAL = (0.3, "General blogs and personal websites")
    FORUM_DISCUSSION = (0.2, "Forum discussions")
    ANECDOTAL = (0.1, "Personal anecdotes")
    UNKNOWN = (0.3, "Unknown or unspecified sources")
    
    def __init__(self, base_score: float, description: str):
        self.base_score = base_score
        self.description = description


class ConfidenceScorer:
    """
    Evaluates and scores information quality and relevance for forecasting.
    
    Features:
    - Source type identification and scoring
    - Evidence consistency and corroboration analysis
    - Information recency weighting
    - Domain-specific information recognition
    - Overall confidence calculation
    """
    
    def __init__(self):
        """Initialize the confidence scorer."""
        # Default weights for different factors
        self.weights = {
            "source_quality": 0.3,
            "information_relevance": 0.25,
            "evidence_consistency": 0.2,
            "information_recency": 0.15,
            "information_specificity": 0.1,
        }
    
    def evaluate_evidence(
        self, 
        evidence: Evidence,
        domain: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single piece of evidence.
        
        Args:
            evidence: The evidence to evaluate
            domain: Optional domain for domain-specific scoring
            
        Returns:
            Dictionary with component scores
        """
        scores = {
            "source_quality": self._score_source_quality(evidence),
            "information_relevance": evidence.relevance_score,
            "evidence_consistency": 0.5,  # Default until multiple pieces compared
            "information_recency": self._score_recency(evidence.content),
            "information_specificity": self._score_specificity(evidence.content),
        }
        
        # Domain-specific adjustment
        if domain:
            domain_relevance = self._score_domain_relevance(evidence.content, domain)
            scores["domain_relevance"] = domain_relevance
            
            # Adjust overall relevance based on domain
            scores["information_relevance"] = (
                scores["information_relevance"] * 0.6 + domain_relevance * 0.4
            )
        
        return scores
    
    def evaluate_evidence_set(
        self, 
        evidences: List[Evidence],
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a set of evidence for overall confidence.
        
        Args:
            evidences: List of evidence to evaluate
            domain: Optional domain for domain-specific scoring
            
        Returns:
            Dictionary with overall scores and analysis
        """
        if not evidences:
            return {
                "overall_confidence": 0.3,
                "confidence_level": ConfidenceLevel.LOW,
                "component_scores": {},
                "individual_scores": [],
                "strengths": ["No evidence to evaluate"],
                "weaknesses": ["No evidence to evaluate"],
            }
        
        # Score individual evidences
        individual_scores = []
        for evidence in evidences:
            scores = self.evaluate_evidence(evidence, domain)
            individual_scores.append({
                "evidence": evidence.to_dict(),
                "scores": scores,
            })
        
        # Evaluate consistency across evidences
        consistency_score = self._score_consistency(evidences)
        
        # Update individual scores with consistency
        for score_dict in individual_scores:
            score_dict["scores"]["evidence_consistency"] = consistency_score
        
        # Calculate overall component scores
        component_scores = {
            "source_quality": self._aggregate_score([s["scores"]["source_quality"] for s in individual_scores]),
            "information_relevance": self._aggregate_score([s["scores"]["information_relevance"] for s in individual_scores]),
            "evidence_consistency": consistency_score,
            "information_recency": self._aggregate_score([s["scores"]["information_recency"] for s in individual_scores]),
            "information_specificity": self._aggregate_score([s["scores"]["information_specificity"] for s in individual_scores]),
        }
        
        if domain:
            component_scores["domain_relevance"] = self._aggregate_score(
                [s["scores"].get("domain_relevance", 0.5) for s in individual_scores]
            )
        
        # Calculate overall confidence score
        overall_confidence = 0
        for component, score in component_scores.items():
            if component in self.weights:
                overall_confidence += score * self.weights[component]
            elif component == "domain_relevance" and domain:
                # If domain specified, add it as a bonus
                overall_confidence += score * 0.1  # Domain as a 10% bonus
        
        # Normalize if weights don't sum to 1
        total_weight = sum(self.weights.values())
        if total_weight != 1:
            overall_confidence /= total_weight
        
        # Determine confidence level
        confidence_level = self._score_to_confidence_level(overall_confidence)
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        # Check components for strengths/weaknesses
        threshold_high = 0.7
        threshold_low = 0.4
        
        for component, score in component_scores.items():
            component_name = component.replace("_", " ").title()
            if score >= threshold_high:
                strengths.append(f"High {component_name} ({score:.2f})")
            elif score <= threshold_low:
                weaknesses.append(f"Low {component_name} ({score:.2f})")
        
        # Check evidence types
        evidence_types = [e.evidence_type for e in evidences]
        type_counts = {}
        for e_type in evidence_types:
            type_counts[e_type] = type_counts.get(e_type, 0) + 1
        
        # Add strengths based on evidence types
        if EvidenceType.FACTUAL in type_counts and type_counts[EvidenceType.FACTUAL] >= 2:
            strengths.append(f"Strong factual evidence ({type_counts[EvidenceType.FACTUAL]} instances)")
        
        if EvidenceType.STATISTICAL in type_counts and type_counts[EvidenceType.STATISTICAL] >= 1:
            strengths.append(f"Includes statistical evidence ({type_counts[EvidenceType.STATISTICAL]} instances)")
        
        # Add weaknesses based on evidence types
        if EvidenceType.ANECDOTAL in type_counts and type_counts[EvidenceType.ANECDOTAL] > 1:
            if len(evidences) > 3 and type_counts[EvidenceType.ANECDOTAL] / len(evidences) > 0.5:
                weaknesses.append(f"Heavy reliance on anecdotal evidence ({type_counts[EvidenceType.ANECDOTAL]} of {len(evidences)} pieces)")
        
        if EvidenceType.ABSENCE_OF_EVIDENCE in type_counts:
            weaknesses.append("Relies on absence of evidence in some areas")
        
        # Return complete evaluation
        return {
            "overall_confidence": overall_confidence,
            "confidence_level": confidence_level,
            "component_scores": component_scores,
            "individual_scores": individual_scores,
            "strengths": strengths,
            "weaknesses": weaknesses,
        }
    
    def calibrate_forecast(
        self,
        raw_score: float,
        confidence_evaluation: Dict[str, Any],
        historical_calibration: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        Calibrate a forecast based on confidence evaluation.
        
        Args:
            raw_score: The raw forecast score (probability 0-1)
            confidence_evaluation: Results from evaluate_evidence_set
            historical_calibration: Optional historical calibration factor
                where 1.0 means perfectly calibrated and <1.0 means overconfident
            
        Returns:
            Tuple of (calibrated_score, lower_bound, upper_bound)
        """
        # Default historical calibration if not provided (assume 20% overconfidence)
        hist_cal = historical_calibration or 0.8
        
        # Get confidence score
        confidence = confidence_evaluation["overall_confidence"]
        
        # Calculate uncertainty based on confidence
        # Lower confidence = wider interval
        base_uncertainty = 1.0 - confidence
        
        # Apply historical calibration
        calibrated_uncertainty = base_uncertainty / hist_cal
        
        # Calculate interval width based on uncertainty
        # For binary questions (probabilities)
        if 0 <= raw_score <= 1:
            # Wider intervals for mid-range probabilities
            max_width = 0.5  # Maximum half-width
            
            # Adjust interval width based on how close to 0.5
            p_adj = abs(raw_score - 0.5) * 2  # 0 at p=0.5, 1 at p=0 or p=1
            width_adj = 1 - (p_adj * 0.7)  # Reduce width near bounds
            
            interval_half_width = min(max_width, calibrated_uncertainty * width_adj)
            
            # Ensure interval stays within bounds
            lower_bound = max(0.01, raw_score - interval_half_width)
            upper_bound = min(0.99, raw_score + interval_half_width)
            
            # Regress extreme probabilities toward the mean based on confidence
            if raw_score < 0.1:
                regression_strength = (1 - confidence) * 0.3
                calibrated_score = raw_score * (1 - regression_strength) + 0.1 * regression_strength
            elif raw_score > 0.9:
                regression_strength = (1 - confidence) * 0.3
                calibrated_score = raw_score * (1 - regression_strength) + 0.9 * regression_strength
            else:
                regression_strength = (1 - confidence) * 0.2
                calibrated_score = raw_score * (1 - regression_strength) + 0.5 * regression_strength
        
        # For numeric estimates, we use a relative uncertainty
        else:
            # Default calibration for numeric estimates
            calibrated_score = raw_score
            
            # Low confidence = wider relative interval
            relative_uncertainty = calibrated_uncertainty * 0.5  # 0-50% uncertainty
            
            # Calculate bounds
            lower_bound = raw_score * (1 - relative_uncertainty)
            upper_bound = raw_score * (1 + relative_uncertainty)
        
        return (calibrated_score, lower_bound, upper_bound)
    
    def adjust_probabilities(
        self,
        probabilities: Dict[str, float],
        confidence_evaluations: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Adjust a set of probabilities based on confidence evaluations.
        
        Args:
            probabilities: Dictionary mapping options to probabilities
            confidence_evaluations: Dictionary mapping options to confidence evaluations
            
        Returns:
            Dictionary with calibrated probabilities
        """
        if not probabilities or sum(probabilities.values()) == 0:
            return probabilities
        
        # Calculate average confidence
        avg_confidence = sum(
            eval["overall_confidence"] for eval in confidence_evaluations.values()
        ) / len(confidence_evaluations)
        
        # Initialize calibrated probabilities
        calibrated = {}
        
        # Apply regression to the mean based on confidence
        regression_strength = (1 - avg_confidence) * 0.4
        equal_weight = 1.0 / len(probabilities)
        
        for option, prob in probabilities.items():
            confidence = confidence_evaluations.get(option, {"overall_confidence": avg_confidence})
            option_conf = confidence["overall_confidence"]
            
            # Higher confidence = less regression
            option_regression = (1 - option_conf) * 0.5
            
            # Calculate regressed probability
            calibrated[option] = prob * (1 - option_regression) + equal_weight * option_regression
        
        # Normalize to ensure sum is 1.0
        total = sum(calibrated.values())
        if total > 0:
            calibrated = {k: v / total for k, v in calibrated.items()}
        
        return calibrated
    
    def _score_source_quality(self, evidence: Evidence) -> float:
        """Score the quality of the evidence source."""
        # Base score from evidence reliability
        base_score = evidence.reliability_score
        
        # Adjust based on evidence type
        type_adjustments = {
            EvidenceType.FACTUAL: 0.1,
            EvidenceType.STATISTICAL: 0.15,
            EvidenceType.EXPERT_OPINION: 0.05,
            EvidenceType.HISTORICAL: 0.0,
            EvidenceType.ANALOGICAL: -0.05,
            EvidenceType.ANECDOTAL: -0.1,
            EvidenceType.THEORETICAL: -0.02,
            EvidenceType.ABSENCE_OF_EVIDENCE: -0.2,
        }
        
        type_adjustment = type_adjustments.get(evidence.evidence_type, 0)
        
        # Adjust based on confidence level
        confidence_adjustments = {
            ConfidenceLevel.VERY_LOW: -0.2,
            ConfidenceLevel.LOW: -0.1,
            ConfidenceLevel.MEDIUM: 0,
            ConfidenceLevel.HIGH: 0.1,
            ConfidenceLevel.VERY_HIGH: 0.15,
        }
        
        confidence_adjustment = confidence_adjustments.get(evidence.confidence, 0)
        
        # Combine scores
        score = base_score + type_adjustment + confidence_adjustment
        
        # Ensure within bounds
        return max(0.0, min(1.0, score))
    
    def _score_recency(self, content: str) -> float:
        """Score the recency of the information."""
        # This is a simplified implementation
        # A real implementation would extract dates and compare to current date
        
        recency_indicators = {
            "recent": 0.8,
            "latest": 0.9,
            "current": 0.85,
            "new study": 0.8,
            "recent data": 0.85,
            "last month": 0.75,
            "last year": 0.6,
            "last week": 0.85,
            "yesterday": 0.9,
            "today": 0.95,
            "historical": 0.4,
            "decade ago": 0.3,
            "years ago": 0.4,
            "traditionally": 0.3,
            "previously": 0.5,
            "2023": 0.9,  # Adjust based on current year
            "2022": 0.8,
            "2021": 0.7,
            "2020": 0.6,
            "2019": 0.5,
            "2018": 0.4,
            "2017": 0.3,
            "2016": 0.3,
            "2015": 0.3,
            "2010": 0.2,
        }
        
        # Check for indicators in the content
        scores = []
        for indicator, score in recency_indicators.items():
            if indicator in content.lower():
                scores.append(score)
        
        # If no time indicators found, use a neutral score
        if not scores:
            return 0.5
        
        # Return the highest score (most recent indicator)
        return max(scores)
    
    def _score_specificity(self, content: str) -> float:
        """Score the specificity of the information."""
        # Check for specific numbers, percentages, statistics
        specificity_score = 0.5  # Default
        
        # Look for numeric content
        num_pattern = r"\d+(?:\.\d+)?%?"
        numbers = re.findall(num_pattern, content)
        
        # More numbers = more specific
        if numbers:
            specificity_score += min(0.3, len(numbers) * 0.05)
        
        # Look for specific details
        detail_indicators = [
            "specifically", "in particular", "exactly", "precisely",
            "detailed", "explicit", "concrete", "definite",
        ]
        
        for indicator in detail_indicators:
            if indicator in content.lower():
                specificity_score += 0.05
        
        # Cap at 1.0
        return min(1.0, specificity_score)
    
    def _score_domain_relevance(self, content: str, domain: str) -> float:
        """Score the relevance to a specific domain."""
        # This is a simplified implementation
        # A real implementation would use domain-specific keyword lists or models
        
        # Convert domain to lowercase for matching
        domain_lower = domain.lower()
        content_lower = content.lower()
        
        # Check if domain term appears directly
        if domain_lower in content_lower:
            # More frequent mentions = higher score
            count = content_lower.count(domain_lower)
            base_score = min(0.9, 0.5 + count * 0.1)
        else:
            base_score = 0.3  # Low relevance if domain not mentioned
        
        # Look for related terms (simplified)
        related_terms = self._get_domain_related_terms(domain)
        for term in related_terms:
            if term in content_lower:
                base_score += 0.05
        
        return min(1.0, base_score)
    
    def _score_consistency(self, evidences: List[Evidence]) -> float:
        """Score the consistency across multiple evidences."""
        if len(evidences) <= 1:
            return 0.5  # Neutral score for single evidence
        
        # Count supporting, contradicting, and neutral evidences
        supporting = 0
        contradicting = 0
        neutral = 0
        
        for evidence in evidences:
            if evidence.impact_direction > 0:
                supporting += 1
            elif evidence.impact_direction < 0:
                contradicting += 1
            else:
                neutral += 1
        
        # Calculate consistency ratio
        total = len(evidences)
        max_count = max(supporting, contradicting, neutral)
        consistency_ratio = max_count / total
        
        # Score based on ratio and total evidence
        base_score = consistency_ratio * 0.6 + 0.2
        
        # Adjust based on evidence count (more evidence = potentially higher score)
        evidence_bonus = min(0.2, (total - 1) * 0.05)
        
        return min(1.0, base_score + evidence_bonus)
    
    def _get_domain_related_terms(self, domain: str) -> List[str]:
        """Get terms related to a specific domain."""
        # This would be a more sophisticated lookup in practice
        domain_terms = {
            "economics": ["economy", "inflation", "recession", "gdp", "market", "financial"],
            "politics": ["government", "election", "policy", "vote", "candidate", "administration"],
            "technology": ["tech", "innovation", "digital", "software", "hardware", "internet"],
            "health": ["medical", "medicine", "disease", "treatment", "patient", "healthcare"],
            "climate": ["environment", "warming", "emissions", "temperature", "sustainable"],
            "military": ["defense", "war", "weapon", "conflict", "troops", "security"],
            "sports": ["game", "player", "team", "competition", "tournament", "athlete"],
            "science": ["research", "experiment", "theory", "data", "scientist", "discovery"],
        }
        
        # Return related terms or empty list if domain not recognized
        return domain_terms.get(domain.lower(), [])
    
    def _aggregate_score(self, scores: List[float]) -> float:
        """Aggregate a list of scores into a single score."""
        if not scores:
            return 0.5
        
        # Use weighted average, giving more weight to higher scores
        weighted_sum = sum(score ** 2 for score in scores)
        weights_sum = sum(score for score in scores)
        
        if weights_sum == 0:
            return 0.5
        
        return weighted_sum / weights_sum
    
    def _score_to_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert a numeric score to a confidence level."""
        if score < 0.2:
            return ConfidenceLevel.VERY_LOW
        elif score < 0.4:
            return ConfidenceLevel.LOW
        elif score < 0.6:
            return ConfidenceLevel.MEDIUM
        elif score < 0.8:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH 