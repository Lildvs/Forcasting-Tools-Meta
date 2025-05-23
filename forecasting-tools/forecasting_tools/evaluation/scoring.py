"""
Forecast Evaluation Framework

This module provides tools for evaluating forecast accuracy and quality,
including calibration metrics, scoring rules, and reasoning analysis.
"""

import math
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
import pandas as pd
from scipy import stats
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score


class ForecastType(str, Enum):
    """Types of forecasts supported for evaluation."""
    BINARY = "binary"
    NUMERIC = "numeric"
    MULTIPLE_CHOICE = "multiple_choice"


class ScoringRule(str, Enum):
    """Scoring rules for evaluating forecasts."""
    BRIER = "brier"
    LOG = "log"
    SPHERICAL = "spherical"
    CALIBRATION = "calibration"
    DISCRIMINATION = "discrimination"
    ACCURACY = "accuracy"


class AccuracyMetrics:
    """Calculate accuracy metrics for forecasts."""
    
    @staticmethod
    def brier_score(probabilities: List[float], outcomes: List[bool]) -> float:
        """
        Calculate Brier score for binary forecasts.
        
        Lower is better (0 is perfect).
        
        Args:
            probabilities: List of forecast probabilities
            outcomes: List of binary outcomes (True/False)
            
        Returns:
            Brier score
        """
        outcomes_01 = [1 if o else 0 for o in outcomes]
        return brier_score_loss(outcomes_01, probabilities)
    
    @staticmethod
    def log_score(probabilities: List[float], outcomes: List[bool]) -> float:
        """
        Calculate logarithmic score for binary forecasts.
        
        Higher is better.
        
        Args:
            probabilities: List of forecast probabilities
            outcomes: List of binary outcomes (True/False)
            
        Returns:
            Logarithmic score
        """
        outcomes_01 = [1 if o else 0 for o in outcomes]
        probabilities_adj = np.clip(probabilities, 1e-15, 1 - 1e-15)  # Avoid log(0)
        return -log_loss(outcomes_01, probabilities_adj)
    
    @staticmethod
    def spherical_score(probabilities: List[float], outcomes: List[bool]) -> float:
        """
        Calculate spherical score for binary forecasts.
        
        Higher is better.
        
        Args:
            probabilities: List of forecast probabilities
            outcomes: List of binary outcomes (True/False)
            
        Returns:
            Spherical score
        """
        score = 0.0
        for prob, outcome in zip(probabilities, outcomes):
            # Adjust probability based on outcome
            p = prob if outcome else 1 - prob
            score += p / math.sqrt(prob**2 + (1-prob)**2)
        
        return score / len(probabilities)
    
    @staticmethod
    def auc_score(probabilities: List[float], outcomes: List[bool]) -> float:
        """
        Calculate AUC (Area Under ROC Curve) for binary forecasts.
        
        Higher is better (1 is perfect).
        
        Args:
            probabilities: List of forecast probabilities
            outcomes: List of binary outcomes (True/False)
            
        Returns:
            AUC score
        """
        outcomes_01 = [1 if o else 0 for o in outcomes]
        return roc_auc_score(outcomes_01, probabilities)
    
    @staticmethod
    def calibration_score(
        probabilities: List[float], 
        outcomes: List[bool],
        num_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Calculate calibration metrics for binary forecasts.
        
        Args:
            probabilities: List of forecast probabilities
            outcomes: List of binary outcomes (True/False)
            num_bins: Number of bins for calibration analysis
            
        Returns:
            Dictionary with calibration metrics
        """
        # Convert outcomes to 0/1
        outcomes_01 = [1 if o else 0 for o in outcomes]
        
        # Create bins
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(probabilities, bin_edges, right=False) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)  # Ensure valid bin indices
        
        # Initialize results
        bin_counts = np.zeros(num_bins)
        bin_sums = np.zeros(num_bins)
        bin_actual_probs = np.zeros(num_bins)
        bin_mean_preds = np.zeros(num_bins)
        
        # Fill bins
        for idx, outcome in zip(bin_indices, outcomes_01):
            bin_counts[idx] += 1
            bin_sums[idx] += outcome
        
        # Calculate calibration metrics
        for i in range(num_bins):
            if bin_counts[i] > 0:
                bin_actual_probs[i] = bin_sums[i] / bin_counts[i]
                
                # Get mean prediction in this bin
                bin_mask = (bin_indices == i)
                bin_predictions = [p for p, m in zip(probabilities, bin_mask) if m]
                bin_mean_preds[i] = np.mean(bin_predictions) if bin_predictions else 0
        
        # Calculate reliability (calibration error)
        valid_bins = (bin_counts > 0)
        weighted_calibration_error = 0
        
        for i in range(num_bins):
            if valid_bins[i]:
                error = abs(bin_actual_probs[i] - bin_mean_preds[i])
                weight = bin_counts[i] / sum(bin_counts[valid_bins])
                weighted_calibration_error += error * weight
        
        # Calculate expected calibration error (ECE)
        ece = 0
        for i in range(num_bins):
            if bin_counts[i] > 0:
                ece += (bin_counts[i] / len(probabilities)) * abs(bin_actual_probs[i] - bin_mean_preds[i])
        
        # Create result dictionary
        result = {
            "reliability": 1 - weighted_calibration_error,
            "ece": ece,
            "bin_data": [
                {
                    "bin_index": i,
                    "bin_start": bin_edges[i],
                    "bin_end": bin_edges[i+1],
                    "count": int(bin_counts[i]),
                    "mean_prediction": float(bin_mean_preds[i]),
                    "actual_probability": float(bin_actual_probs[i])
                }
                for i in range(num_bins) if bin_counts[i] > 0
            ]
        }
        
        return result
    
    @staticmethod
    def numeric_forecast_error(
        predictions: List[float],
        outcomes: List[float]
    ) -> Dict[str, float]:
        """
        Calculate error metrics for numeric forecasts.
        
        Args:
            predictions: List of forecast median/point values
            outcomes: List of actual outcomes
            
        Returns:
            Dictionary with error metrics
        """
        abs_errors = [abs(p - o) for p, o in zip(predictions, outcomes)]
        sq_errors = [(p - o)**2 for p, o in zip(predictions, outcomes)]
        pct_errors = [abs(p - o) / abs(o) if o != 0 else float('inf') for p, o in zip(predictions, outcomes)]
        
        # Filter out infinite values
        valid_pct_errors = [e for e in pct_errors if e != float('inf')]
        
        return {
            "mae": np.mean(abs_errors),
            "rmse": np.sqrt(np.mean(sq_errors)),
            "mape": np.mean(valid_pct_errors) if valid_pct_errors else float('inf')
        }
    
    @staticmethod
    def distribution_scoring(
        distributions: List[Dict[str, float]],
        outcomes: List[float]
    ) -> Dict[str, float]:
        """
        Score distributional forecasts.
        
        Args:
            distributions: List of distribution parameters (median, p10, p90)
            outcomes: List of actual outcomes
            
        Returns:
            Dictionary with scoring metrics
        """
        inside_90_pct = 0
        crps_scores = []
        
        for dist, outcome in zip(distributions, outcomes):
            # Check if outcome is within 90% interval
            if dist.get("p10", 0) <= outcome <= dist.get("p90", float('inf')):
                inside_90_pct += 1
            
            # Calculate CRPS (Continuous Ranked Probability Score) - approximation
            # Assuming normal distribution based on p10 and p90
            if "p10" in dist and "p90" in dist:
                # Estimate normal parameters from percentiles
                p10, p90 = dist["p10"], dist["p90"]
                median = dist.get("median", (p10 + p90) / 2)
                
                # Estimate standard deviation
                z_90 = 1.282  # z-score for 90th percentile
                sigma = (p90 - p10) / (2 * z_90)
                
                # Calculate CRPS for normal distribution
                z = (outcome - median) / sigma
                crps = sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1/np.sqrt(np.pi))
                crps_scores.append(abs(crps))
        
        return {
            "coverage_90pct": inside_90_pct / len(distributions) if distributions else 0,
            "crps": np.mean(crps_scores) if crps_scores else 0
        }


class ForecastEvaluator:
    """
    Evaluate forecasts against outcomes.
    
    Provides comprehensive evaluation of different forecast types.
    """
    
    def __init__(self, forecast_file: Optional[str] = None):
        """
        Initialize evaluator, optionally with a forecast file.
        
        Args:
            forecast_file: Path to JSON file with forecasts and outcomes
        """
        self.forecasts = []
        
        if forecast_file:
            self.load_forecasts(forecast_file)
    
    def load_forecasts(self, file_path: str) -> None:
        """
        Load forecasts from a JSON file.
        
        Args:
            file_path: Path to JSON file
        """
        with open(file_path, "r") as f:
            data = json.load(f)
            
            if "predictions" in data:
                self.forecasts = data["predictions"]
            elif isinstance(data, list):
                self.forecasts = data
            else:
                raise ValueError("Invalid forecast data format")
    
    def evaluate_binary_forecasts(self) -> Dict[str, Any]:
        """
        Evaluate binary forecasts.
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Filter binary forecasts with outcomes
        binary_forecasts = [
            f for f in self.forecasts
            if f.get("forecast_type") == "binary" and 
               "prediction" in f and 
               "outcome" in f
        ]
        
        if not binary_forecasts:
            return {"error": "No binary forecasts with outcomes found"}
        
        # Extract probabilities and outcomes
        probabilities = []
        outcomes = []
        
        for forecast in binary_forecasts:
            if "probability" in forecast["prediction"] and "result" in forecast["outcome"]:
                probabilities.append(forecast["prediction"]["probability"])
                outcomes.append(forecast["outcome"]["result"])
        
        # Calculate metrics
        metrics = {}
        
        if probabilities and outcomes:
            metrics["count"] = len(probabilities)
            metrics["brier_score"] = AccuracyMetrics.brier_score(probabilities, outcomes)
            metrics["log_score"] = AccuracyMetrics.log_score(probabilities, outcomes)
            metrics["spherical_score"] = AccuracyMetrics.spherical_score(probabilities, outcomes)
            
            if len(set(outcomes)) > 1:  # Need both positive and negative outcomes for AUC
                metrics["auc"] = AccuracyMetrics.auc_score(probabilities, outcomes)
            
            metrics["calibration"] = AccuracyMetrics.calibration_score(probabilities, outcomes)
            
            # Calculate accuracy (if prediction > 0.5 and outcome is True or prediction < 0.5 and outcome is False)
            correct_predictions = sum(
                1 for p, o in zip(probabilities, outcomes)
                if (p > 0.5 and o) or (p < 0.5 and not o)
            )
            metrics["accuracy"] = correct_predictions / len(probabilities)
            
            # Calculate overconfidence/underconfidence
            avg_prediction = sum(probabilities) / len(probabilities)
            actual_rate = sum(1 for o in outcomes if o) / len(outcomes)
            metrics["overconfidence"] = avg_prediction - actual_rate
        
        return metrics
    
    def evaluate_numeric_forecasts(self) -> Dict[str, Any]:
        """
        Evaluate numeric forecasts.
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Filter numeric forecasts with outcomes
        numeric_forecasts = [
            f for f in self.forecasts
            if f.get("forecast_type") == "numeric" and 
               "prediction" in f and 
               "outcome" in f
        ]
        
        if not numeric_forecasts:
            return {"error": "No numeric forecasts with outcomes found"}
        
        # For point predictions
        point_predictions = []
        point_outcomes = []
        
        # For distributional predictions
        distributions = []
        dist_outcomes = []
        
        for forecast in numeric_forecasts:
            if "outcome" in forecast and "value" in forecast["outcome"]:
                outcome = forecast["outcome"]["value"]
                
                if "prediction" in forecast:
                    pred = forecast["prediction"]
                    
                    # Check if it's a distributional forecast
                    if "distribution" in pred:
                        dist = pred["distribution"]
                        if "median" in dist:
                            point_predictions.append(dist["median"])
                            point_outcomes.append(outcome)
                        
                        if "p10" in dist and "p90" in dist:
                            distributions.append(dist)
                            dist_outcomes.append(outcome)
                    # Or a point forecast
                    elif "value" in pred:
                        point_predictions.append(pred["value"])
                        point_outcomes.append(outcome)
        
        # Calculate metrics
        metrics = {}
        
        if point_predictions and point_outcomes:
            metrics["count"] = len(point_predictions)
            metrics["error_metrics"] = AccuracyMetrics.numeric_forecast_error(
                point_predictions, point_outcomes
            )
        
        if distributions and dist_outcomes:
            metrics["distribution_metrics"] = AccuracyMetrics.distribution_scoring(
                distributions, dist_outcomes
            )
        
        return metrics
    
    def evaluate_multiple_choice_forecasts(self) -> Dict[str, Any]:
        """
        Evaluate multiple choice forecasts.
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Filter multiple choice forecasts with outcomes
        mc_forecasts = [
            f for f in self.forecasts
            if f.get("forecast_type") == "multiple_choice" and 
               "prediction" in f and 
               "outcome" in f
        ]
        
        if not mc_forecasts:
            return {"error": "No multiple choice forecasts with outcomes found"}
        
        # Extract probabilities and outcomes
        correct_probs = []
        all_probs = []
        all_outcomes = []
        
        for forecast in mc_forecasts:
            if ("options" in forecast["prediction"] and 
                "result" in forecast["outcome"]):
                
                options = forecast["prediction"]["options"]
                result = forecast["outcome"]["result"]
                
                # Find probability assigned to correct outcome
                correct_prob = 0
                for option in options:
                    if option["option"] == result:
                        correct_prob = option["probability"]
                        break
                
                correct_probs.append(correct_prob)
                
                # Also prepare data for multi-class metrics
                probs = [opt["probability"] for opt in options]
                outcome_idx = next(
                    (i for i, opt in enumerate(options) if opt["option"] == result),
                    None
                )
                
                if outcome_idx is not None:
                    all_probs.append(probs)
                    outcome_vector = [0] * len(options)
                    outcome_vector[outcome_idx] = 1
                    all_outcomes.append(outcome_vector)
        
        # Calculate metrics
        metrics = {}
        
        if correct_probs:
            metrics["count"] = len(correct_probs)
            metrics["avg_correct_probability"] = sum(correct_probs) / len(correct_probs)
            
            # Log score for multiple choice is similar to binary but using the probability
            # assigned to the correct outcome
            log_scores = [math.log(max(p, 1e-15)) for p in correct_probs]
            metrics["log_score"] = sum(log_scores) / len(log_scores)
        
        if all_probs and all_outcomes:
            # Calculate accuracy (highest probability option matches outcome)
            correct_predictions = 0
            for probs, outcome in zip(all_probs, all_outcomes):
                pred_idx = probs.index(max(probs))
                if outcome[pred_idx] == 1:
                    correct_predictions += 1
            
            metrics["accuracy"] = correct_predictions / len(all_probs)
        
        return metrics
    
    def evaluate_all(self) -> Dict[str, Any]:
        """
        Evaluate all forecasts.
        
        Returns:
            Dictionary with all evaluation metrics
        """
        return {
            "binary": self.evaluate_binary_forecasts(),
            "numeric": self.evaluate_numeric_forecasts(),
            "multiple_choice": self.evaluate_multiple_choice_forecasts()
        }
    
    def evaluate_by_category(self) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate forecasts grouped by category.
        
        Returns:
            Dictionary with evaluation metrics by category
        """
        # Extract all categories
        categories = set()
        for forecast in self.forecasts:
            if "category" in forecast:
                categories.add(forecast["category"])
        
        # Evaluate each category
        results = {}
        
        for category in categories:
            # Create a temporary evaluator with only forecasts from this category
            temp_evaluator = ForecastEvaluator()
            temp_evaluator.forecasts = [
                f for f in self.forecasts if f.get("category") == category
            ]
            
            results[category] = temp_evaluator.evaluate_all()
        
        return results
    
    def evaluate_by_difficulty(self) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate forecasts grouped by difficulty.
        
        Returns:
            Dictionary with evaluation metrics by difficulty
        """
        # Extract all difficulty levels
        difficulties = set()
        for forecast in self.forecasts:
            if "metadata" in forecast and "difficulty" in forecast["metadata"]:
                difficulties.add(forecast["metadata"]["difficulty"])
        
        # Evaluate each difficulty level
        results = {}
        
        for difficulty in difficulties:
            # Create a temporary evaluator with only forecasts from this difficulty
            temp_evaluator = ForecastEvaluator()
            temp_evaluator.forecasts = [
                f for f in self.forecasts
                if "metadata" in f and f["metadata"].get("difficulty") == difficulty
            ]
            
            results[difficulty] = temp_evaluator.evaluate_all()
        
        return results
    
    def generate_calibration_plot_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate data for calibration plots.
        
        Returns:
            Dictionary with calibration data by forecast type
        """
        results = {}
        
        # Binary forecasts
        binary_forecasts = [
            f for f in self.forecasts
            if f.get("forecast_type") == "binary" and 
               "prediction" in f and 
               "outcome" in f
        ]
        
        if binary_forecasts:
            probabilities = []
            outcomes = []
            
            for forecast in binary_forecasts:
                if "probability" in forecast["prediction"] and "result" in forecast["outcome"]:
                    probabilities.append(forecast["prediction"]["probability"])
                    outcomes.append(forecast["outcome"]["result"])
            
            if probabilities and outcomes:
                calibration_data = AccuracyMetrics.calibration_score(
                    probabilities, outcomes, num_bins=10
                )
                results["binary"] = calibration_data["bin_data"]
        
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Returns:
            Dictionary with all evaluation results
        """
        return {
            "overall": self.evaluate_all(),
            "by_category": self.evaluate_by_category(),
            "by_difficulty": self.evaluate_by_difficulty(),
            "calibration_data": self.generate_calibration_plot_data()
        }


class ReasoningEvaluator:
    """
    Evaluate the quality of forecast reasoning.
    
    Uses heuristics to assess reasoning quality.
    """
    
    @staticmethod
    def evaluate_reasoning(reasoning: str) -> Dict[str, Any]:
        """
        Evaluate reasoning quality using simple heuristics.
        
        Args:
            reasoning: Reasoning text to evaluate
            
        Returns:
            Dictionary with reasoning quality metrics
        """
        if not reasoning:
            return {
                "length": 0,
                "word_count": 0,
                "quality_score": 0,
                "has_quantitative": False,
                "has_comparative": False,
                "has_causal": False,
                "has_conditional": False
            }
        
        # Count words
        words = reasoning.split()
        word_count = len(words)
        
        # Check for quantitative reasoning
        quantitative_terms = [
            "percent", "%", "probability", "likelihood", "chance", 
            "frequency", "rate", "statistics", "data", "evidence",
            "increase", "decrease", "growth", "decline", "trend",
            "number", "quantity", "amount", "measure", "ratio",
            "proportion", "fraction", "multiply", "divide", "calculate"
        ]
        
        has_quantitative = any(term in reasoning.lower() for term in quantitative_terms)
        
        # Check for comparative reasoning
        comparative_terms = [
            "compared to", "comparison", "contrast", "greater than", "less than",
            "more than", "fewer than", "higher", "lower", "similar to",
            "different from", "exceeds", "falls short of", "outperforms",
            "underperforms", "better", "worse", "stronger", "weaker"
        ]
        
        has_comparative = any(term in reasoning.lower() for term in comparative_terms)
        
        # Check for causal reasoning
        causal_terms = [
            "because", "cause", "effect", "impact", "influence", "result in",
            "lead to", "due to", "consequence", "therefore", "thus", "hence",
            "so that", "in order to", "explains why", "contributes to",
            "stems from", "originates from", "drives", "triggers"
        ]
        
        has_causal = any(term in reasoning.lower() for term in causal_terms)
        
        # Check for conditional reasoning
        conditional_terms = [
            "if", "then", "unless", "assuming", "provided that", "in case",
            "conditional on", "given that", "depends on", "subject to",
            "would", "could", "might", "may", "possible", "scenario",
            "alternative", "contingent", "prerequisite"
        ]
        
        has_conditional = any(term in reasoning.lower() for term in conditional_terms)
        
        # Calculate overall quality score (very simple heuristic)
        length_score = min(1.0, word_count / 100)  # Cap at 100 words
        feature_score = sum([
            has_quantitative, 
            has_comparative, 
            has_causal, 
            has_conditional
        ]) / 4.0
        
        quality_score = (length_score + feature_score) / 2
        
        return {
            "length": len(reasoning),
            "word_count": word_count,
            "quality_score": quality_score,
            "has_quantitative": has_quantitative,
            "has_comparative": has_comparative,
            "has_causal": has_causal,
            "has_conditional": has_conditional
        }
    
    @staticmethod
    def batch_evaluate_reasoning(forecasts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate reasoning quality for a batch of forecasts.
        
        Args:
            forecasts: List of forecast dictionaries
            
        Returns:
            Dictionary with aggregated reasoning quality metrics
        """
        evaluated = []
        
        for forecast in forecasts:
            if "prediction" in forecast and "reasoning" in forecast["prediction"]:
                reasoning = forecast["prediction"]["reasoning"]
                eval_result = ReasoningEvaluator.evaluate_reasoning(reasoning)
                eval_result["forecast_id"] = forecast.get("id", "unknown")
                
                if "question" in forecast:
                    eval_result["question"] = forecast["question"]
                
                evaluated.append(eval_result)
        
        if not evaluated:
            return {"error": "No forecasts with reasoning found"}
        
        # Calculate aggregate metrics
        total_score = sum(item["quality_score"] for item in evaluated)
        avg_score = total_score / len(evaluated)
        
        quantitative_count = sum(1 for item in evaluated if item["has_quantitative"])
        comparative_count = sum(1 for item in evaluated if item["has_comparative"])
        causal_count = sum(1 for item in evaluated if item["has_causal"])
        conditional_count = sum(1 for item in evaluated if item["has_conditional"])
        
        avg_length = sum(item["length"] for item in evaluated) / len(evaluated)
        avg_word_count = sum(item["word_count"] for item in evaluated) / len(evaluated)
        
        return {
            "count": len(evaluated),
            "average_quality_score": avg_score,
            "average_length": avg_length,
            "average_word_count": avg_word_count,
            "quantitative_reasoning_percentage": quantitative_count / len(evaluated),
            "comparative_reasoning_percentage": comparative_count / len(evaluated),
            "causal_reasoning_percentage": causal_count / len(evaluated),
            "conditional_reasoning_percentage": conditional_count / len(evaluated),
            "individual_evaluations": evaluated
        }


def calculate_proper_scoring_rule(
    probabilities: List[float], 
    outcomes: List[bool], 
    rule: ScoringRule
) -> float:
    """
    Calculate a proper scoring rule for binary forecasts.
    
    Args:
        probabilities: Forecast probabilities
        outcomes: Actual outcomes
        rule: Scoring rule to use
        
    Returns:
        Score value
    """
    if rule == ScoringRule.BRIER:
        return AccuracyMetrics.brier_score(probabilities, outcomes)
    elif rule == ScoringRule.LOG:
        return AccuracyMetrics.log_score(probabilities, outcomes)
    elif rule == ScoringRule.SPHERICAL:
        return AccuracyMetrics.spherical_score(probabilities, outcomes)
    elif rule == ScoringRule.CALIBRATION:
        return AccuracyMetrics.calibration_score(probabilities, outcomes)["reliability"]
    elif rule == ScoringRule.DISCRIMINATION:
        return AccuracyMetrics.auc_score(probabilities, outcomes)
    else:
        raise ValueError(f"Unknown scoring rule: {rule}")


def load_and_evaluate(
    forecast_file: str, 
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load forecasts from a file and evaluate them.
    
    Args:
        forecast_file: Path to forecast file
        output_file: Optional path to save evaluation results
        
    Returns:
        Evaluation results
    """
    evaluator = ForecastEvaluator(forecast_file)
    results = evaluator.generate_report()
    
    # Include reasoning evaluation
    reasoning_evaluator = ReasoningEvaluator()
    results["reasoning"] = reasoning_evaluator.batch_evaluate_reasoning(evaluator.forecasts)
    
    # Save results if output file specified
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
    
    return results 