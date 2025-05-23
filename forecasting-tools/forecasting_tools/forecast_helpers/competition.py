"""
Competition Framework

This module provides a framework for evaluating and comparing forecast
performance, with a focus on tracking results by personality type.
"""

import logging
import json
import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    NumericQuestion,
    MultipleChoiceQuestion,
    MetaculusQuestion
)
from forecasting_tools.data_models.forecast_report import (
    BinaryReport,
    NumericReport,
    MultipleChoiceReport,
    ForecastReport
)
from forecasting_tools.personality_management.config import PersonalityConfig


logger = logging.getLogger(__name__)


class CompetitionMetric(str, Enum):
    """Metrics for evaluating forecast performance."""
    BASELINE_SCORE = "baseline_score"
    EXPECTED_BASELINE = "expected_baseline"
    BRIER_SCORE = "brier_score"
    DEVIATION_SCORE = "deviation_score"
    CALIBRATION = "calibration"


@dataclass
class CompetitionEntry:
    """An entry in a forecasting competition."""
    bot_id: str
    personality_name: Optional[str] = None
    personality_config: Optional[PersonalityConfig] = None
    reports: List[ForecastReport] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def entry_id(self) -> str:
        """Get a unique identifier for this entry."""
        if self.personality_name:
            return f"{self.bot_id}_{self.personality_name}"
        return self.bot_id


class CompetitionTracker:
    """
    Tracks and analyzes competition results.
    
    This class tracks forecast performance across different bots and personalities,
    calculating various performance metrics and providing analysis tools.
    """
    
    def __init__(self, name: str = "forecasting_competition"):
        """
        Initialize the competition tracker.
        
        Args:
            name: Name of the competition
        """
        self.name = name
        self.entries: Dict[str, CompetitionEntry] = {}
        self.questions: Dict[str, MetaculusQuestion] = {}
        self.results_by_question: Dict[str, Dict[str, Any]] = {}
        self.metrics_by_domain: Dict[str, Dict[str, Dict[str, float]]] = {}
        
    def add_entry(
        self, 
        bot_id: str, 
        personality_name: Optional[str] = None,
        personality_config: Optional[PersonalityConfig] = None
    ) -> CompetitionEntry:
        """
        Add a new entry to the competition.
        
        Args:
            bot_id: Identifier for the bot
            personality_name: Optional name of the personality
            personality_config: Optional personality configuration
            
        Returns:
            The created competition entry
        """
        entry = CompetitionEntry(
            bot_id=bot_id,
            personality_name=personality_name,
            personality_config=personality_config
        )
        
        self.entries[entry.entry_id] = entry
        return entry
    
    def add_report(
        self, 
        bot_id: str, 
        report: ForecastReport,
        personality_name: Optional[str] = None
    ) -> None:
        """
        Add a forecast report to an entry.
        
        Args:
            bot_id: Identifier for the bot
            report: The forecast report
            personality_name: Optional name of the personality
        """
        # Create entry ID
        entry_id = f"{bot_id}_{personality_name}" if personality_name else bot_id
        
        # Create entry if it doesn't exist
        if entry_id not in self.entries:
            self.add_entry(bot_id, personality_name)
        
        # Add the report
        self.entries[entry_id].reports.append(report)
        
        # Add the question if it's new
        question = report.question
        question_id = str(question.id)
        if question_id not in self.questions:
            self.questions[question_id] = question
    
    def calculate_metrics(self, metrics: Optional[List[CompetitionMetric]] = None) -> None:
        """
        Calculate metrics for all entries.
        
        Args:
            metrics: Optional list of metrics to calculate
        """
        if metrics is None:
            metrics = [
                CompetitionMetric.BASELINE_SCORE,
                CompetitionMetric.EXPECTED_BASELINE,
                CompetitionMetric.BRIER_SCORE,
                CompetitionMetric.DEVIATION_SCORE,
                CompetitionMetric.CALIBRATION
            ]
            
        # Calculate metrics for each entry
        for entry_id, entry in self.entries.items():
            if not entry.reports:
                continue
                
            for metric in metrics:
                if metric == CompetitionMetric.BASELINE_SCORE:
                    # Average baseline score
                    scores = []
                    for report in entry.reports:
                        if hasattr(report, "baseline_score") and report.baseline_score is not None:
                            scores.append(report.baseline_score)
                    if scores:
                        entry.metrics[metric] = sum(scores) / len(scores)
                
                elif metric == CompetitionMetric.EXPECTED_BASELINE:
                    # Average expected baseline score
                    scores = []
                    for report in entry.reports:
                        if hasattr(report, "expected_baseline_score") and report.expected_baseline_score is not None:
                            scores.append(report.expected_baseline_score)
                    if scores:
                        entry.metrics[metric] = sum(scores) / len(scores)
                
                elif metric == CompetitionMetric.BRIER_SCORE:
                    # Average Brier score (for binary questions)
                    scores = []
                    for report in entry.reports:
                        if isinstance(report, BinaryReport) and report.question.resolution is not None:
                            # Lower is better for Brier score
                            actual = 1.0 if report.question.resolution else 0.0
                            predicted = report.prediction
                            brier_score = (actual - predicted) ** 2
                            scores.append(brier_score)
                    if scores:
                        entry.metrics[metric] = sum(scores) / len(scores)
                
                elif metric == CompetitionMetric.DEVIATION_SCORE:
                    # Average deviation from community prediction
                    scores = []
                    for report in entry.reports:
                        if report.community_prediction is not None:
                            if isinstance(report, BinaryReport):
                                deviation = abs(report.prediction - report.community_prediction)
                                scores.append(deviation)
                            elif isinstance(report, NumericReport):
                                # For numeric, use z-score difference
                                if report.community_prediction.stdev > 0:
                                    z_diff = abs(
                                        (report.prediction.mean - report.community_prediction.mean) 
                                        / report.community_prediction.stdev
                                    )
                                    scores.append(z_diff)
                    if scores:
                        entry.metrics[metric] = sum(scores) / len(scores)
                
                elif metric == CompetitionMetric.CALIBRATION:
                    # Calibration score (for binary questions)
                    if any(isinstance(r, BinaryReport) for r in entry.reports):
                        calibration_score = self._calculate_calibration_score(entry)
                        if calibration_score is not None:
                            entry.metrics[metric] = calibration_score
    
    def calculate_metrics_by_domain(self, domain_field: str = "category") -> None:
        """
        Calculate metrics by question domain.
        
        Args:
            domain_field: Field to use for domain categorization
        """
        # Get all domains
        domains = set()
        for question in self.questions.values():
            if hasattr(question, domain_field):
                domain = getattr(question, domain_field)
                if domain:
                    domains.add(domain)
        
        # Initialize metrics by domain
        self.metrics_by_domain = {domain: {} for domain in domains}
        
        # Calculate metrics for each entry by domain
        for entry_id, entry in self.entries.items():
            if not entry.reports:
                continue
                
            # Group reports by domain
            reports_by_domain = {}
            for report in entry.reports:
                question = report.question
                if hasattr(question, domain_field):
                    domain = getattr(question, domain_field)
                    if domain:
                        if domain not in reports_by_domain:
                            reports_by_domain[domain] = []
                        reports_by_domain[domain].append(report)
            
            # Calculate metrics for each domain
            for domain, reports in reports_by_domain.items():
                if domain not in self.metrics_by_domain:
                    self.metrics_by_domain[domain] = {}
                
                domain_metrics = {}
                
                # Baseline score
                baseline_scores = [
                    r.baseline_score for r in reports 
                    if hasattr(r, "baseline_score") and r.baseline_score is not None
                ]
                if baseline_scores:
                    domain_metrics[CompetitionMetric.BASELINE_SCORE] = sum(baseline_scores) / len(baseline_scores)
                
                # Expected baseline score
                expected_baseline_scores = [
                    r.expected_baseline_score for r in reports 
                    if hasattr(r, "expected_baseline_score") and r.expected_baseline_score is not None
                ]
                if expected_baseline_scores:
                    domain_metrics[CompetitionMetric.EXPECTED_BASELINE] = sum(expected_baseline_scores) / len(expected_baseline_scores)
                
                # Brier score for binary questions
                binary_reports = [r for r in reports if isinstance(r, BinaryReport) and r.question.resolution is not None]
                if binary_reports:
                    brier_scores = [
                        (1.0 if r.question.resolution else 0.0 - r.prediction) ** 2
                        for r in binary_reports
                    ]
                    domain_metrics[CompetitionMetric.BRIER_SCORE] = sum(brier_scores) / len(brier_scores)
                
                self.metrics_by_domain[domain][entry_id] = domain_metrics
    
    def get_leaderboard(
        self, metric: CompetitionMetric = CompetitionMetric.EXPECTED_BASELINE
    ) -> List[Tuple[str, float]]:
        """
        Get a sorted leaderboard based on a specific metric.
        
        Args:
            metric: The metric to sort by
            
        Returns:
            List of (entry_id, score) tuples sorted by score
        """
        leaderboard = []
        
        for entry_id, entry in self.entries.items():
            if metric in entry.metrics:
                leaderboard.append((entry_id, entry.metrics[metric]))
        
        # Sort by metric (higher is better)
        reverse = True
        if metric == CompetitionMetric.BRIER_SCORE or metric == CompetitionMetric.DEVIATION_SCORE:
            # Lower is better for these metrics
            reverse = False
            
        return sorted(leaderboard, key=lambda x: x[1], reverse=reverse)
    
    def get_domain_leaderboard(
        self, 
        domain: str,
        metric: CompetitionMetric = CompetitionMetric.EXPECTED_BASELINE
    ) -> List[Tuple[str, float]]:
        """
        Get a sorted leaderboard for a specific domain.
        
        Args:
            domain: The domain to get a leaderboard for
            metric: The metric to sort by
            
        Returns:
            List of (entry_id, score) tuples sorted by score
        """
        if domain not in self.metrics_by_domain:
            return []
            
        leaderboard = []
        
        for entry_id, metrics in self.metrics_by_domain[domain].items():
            if metric in metrics:
                leaderboard.append((entry_id, metrics[metric]))
        
        # Sort by metric (higher is better)
        reverse = True
        if metric == CompetitionMetric.BRIER_SCORE or metric == CompetitionMetric.DEVIATION_SCORE:
            # Lower is better for these metrics
            reverse = False
            
        return sorted(leaderboard, key=lambda x: x[1], reverse=reverse)
    
    def get_personality_performance(
        self, 
        personality_name: str,
        metric: CompetitionMetric = CompetitionMetric.EXPECTED_BASELINE
    ) -> Dict[str, float]:
        """
        Get performance metrics for a specific personality across all bots.
        
        Args:
            personality_name: Name of the personality
            metric: The metric to analyze
            
        Returns:
            Dictionary of performance metrics
        """
        # Get all entries with this personality
        entries = [
            entry for entry_id, entry in self.entries.items()
            if entry.personality_name == personality_name
        ]
        
        if not entries:
            return {}
            
        # Get metric values
        values = [
            entry.metrics[metric] for entry in entries
            if metric in entry.metrics
        ]
        
        if not values:
            return {}
            
        # Calculate statistics
        return {
            "mean": sum(values) / len(values),
            "median": sorted(values)[len(values) // 2],
            "min": min(values),
            "max": max(values),
            "std_dev": (sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)) ** 0.5,
            "count": len(values)
        }
    
    def compare_personalities(
        self, 
        metric: CompetitionMetric = CompetitionMetric.EXPECTED_BASELINE
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare performance across different personalities.
        
        Args:
            metric: The metric to analyze
            
        Returns:
            Dictionary mapping personality names to performance metrics
        """
        # Get all personality names
        personality_names = set(
            entry.personality_name for entry in self.entries.values()
            if entry.personality_name is not None
        )
        
        # Get performance for each personality
        return {
            personality_name: self.get_personality_performance(personality_name, metric)
            for personality_name in personality_names
        }
    
    def get_question_results(self, question_id: str) -> Dict[str, Any]:
        """
        Get results for a specific question.
        
        Args:
            question_id: ID of the question
            
        Returns:
            Dictionary of results
        """
        if question_id not in self.questions:
            return {}
            
        question = self.questions[question_id]
        
        # Get predictions for this question
        predictions = {}
        for entry_id, entry in self.entries.items():
            for report in entry.reports:
                if str(report.question.id) == question_id:
                    if isinstance(report, BinaryReport):
                        predictions[entry_id] = report.prediction
                    elif isinstance(report, NumericReport):
                        predictions[entry_id] = {
                            "mean": report.prediction.mean,
                            "stdev": report.prediction.stdev
                        }
                    elif isinstance(report, MultipleChoiceReport):
                        predictions[entry_id] = {
                            option: prob
                            for option, prob in zip(report.question.option_names, report.prediction.probabilities)
                        }
        
        # Get community prediction if available
        community_prediction = None
        for entry_id, entry in self.entries.items():
            for report in entry.reports:
                if str(report.question.id) == question_id and report.community_prediction is not None:
                    community_prediction = report.community_prediction
                    break
            if community_prediction is not None:
                break
        
        return {
            "question": {
                "id": question_id,
                "text": question.question_text,
                "type": type(question).__name__,
                "resolution": question.resolution if hasattr(question, "resolution") else None
            },
            "predictions": predictions,
            "community_prediction": community_prediction
        }
    
    def save_results(self, file_path: str) -> None:
        """
        Save competition results to a file.
        
        Args:
            file_path: Path to save the results to
        """
        # Create a serializable representation
        data = {
            "name": self.name,
            "timestamp": datetime.datetime.now().isoformat(),
            "entries": {
                entry_id: {
                    "bot_id": entry.bot_id,
                    "personality_name": entry.personality_name,
                    "metrics": entry.metrics
                }
                for entry_id, entry in self.entries.items()
            },
            "questions": {
                question_id: {
                    "id": question.id,
                    "text": question.question_text,
                    "type": type(question).__name__,
                    "resolution": question.resolution if hasattr(question, "resolution") else None
                }
                for question_id, question in self.questions.items()
            },
            "metrics_by_domain": self.metrics_by_domain
        }
        
        # Save to file
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load_results(self, file_path: str) -> None:
        """
        Load competition results from a file.
        
        Args:
            file_path: Path to load the results from
        """
        with open(file_path, "r") as f:
            data = json.load(f)
            
        self.name = data["name"]
        
        # Load entries
        self.entries = {}
        for entry_id, entry_data in data["entries"].items():
            entry = CompetitionEntry(
                bot_id=entry_data["bot_id"],
                personality_name=entry_data["personality_name"]
            )
            entry.metrics = entry_data["metrics"]
            self.entries[entry_id] = entry
            
        # Load metrics by domain
        self.metrics_by_domain = data["metrics_by_domain"]
    
    def create_performance_visualization(self, metric: CompetitionMetric = CompetitionMetric.EXPECTED_BASELINE) -> str:
        """
        Create a visualization of personality performance.
        
        Args:
            metric: The metric to visualize
            
        Returns:
            Base64-encoded PNG image
        """
        # Get data for visualization
        personality_performance = self.compare_personalities(metric)
        
        if not personality_performance:
            # Create empty figure with message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No personality data available", ha="center", va="center", fontsize=14)
            ax.axis("off")
        else:
            # Create figure with bar chart
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Extract data
            personalities = []
            mean_scores = []
            std_devs = []
            
            for personality, stats in personality_performance.items():
                if "mean" in stats and "std_dev" in stats:
                    personalities.append(personality)
                    mean_scores.append(stats["mean"])
                    std_devs.append(stats["std_dev"])
            
            # Sort by mean score
            sorted_indices = np.argsort(mean_scores)
            if metric != CompetitionMetric.BRIER_SCORE and metric != CompetitionMetric.DEVIATION_SCORE:
                # Higher is better for most metrics
                sorted_indices = sorted_indices[::-1]
                
            personalities = [personalities[i] for i in sorted_indices]
            mean_scores = [mean_scores[i] for i in sorted_indices]
            std_devs = [std_devs[i] for i in sorted_indices]
            
            # Create bar chart
            bars = ax.bar(personalities, mean_scores, yerr=std_devs, capsize=5)
            
            # Customize chart
            ax.set_title(f"Personality Performance - {metric.value}", fontsize=16)
            ax.set_xlabel("Personality", fontsize=12)
            ax.set_ylabel(f"{metric.value} (mean ± std dev)", fontsize=12)
            ax.tick_params(axis="x", rotation=45)
            
            # Add values on bars
            for bar, score in zip(bars, mean_scores):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f"{score:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10
                )
            
            plt.tight_layout()
        
        # Convert to base64 encoded string
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        buffer.seek(0)
        image_str = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close(fig)
        
        return f"data:image/png;base64,{image_str}"
    
    def _calculate_calibration_score(self, entry: CompetitionEntry) -> Optional[float]:
        """
        Calculate calibration score for binary questions.
        
        Args:
            entry: The competition entry
            
        Returns:
            Calibration score between 0.0 (worst) and 1.0 (best)
        """
        # Get binary reports with resolutions
        binary_reports = [
            report for report in entry.reports
            if isinstance(report, BinaryReport) and report.question.resolution is not None
        ]
        
        if not binary_reports:
            return None
            
        # Group predictions by probability bucket
        buckets = {}
        for report in binary_reports:
            # Round prediction to nearest 0.1
            bucket = round(report.prediction * 10) / 10
            if bucket not in buckets:
                buckets[bucket] = {"count": 0, "correct": 0}
            
            buckets[bucket]["count"] += 1
            if report.question.resolution:
                buckets[bucket]["correct"] += 1
        
        # Calculate calibration error
        calibration_error = 0.0
        total_predictions = len(binary_reports)
        
        for bucket, stats in buckets.items():
            # Calculate the fraction that were actually correct
            actual_fraction = stats["correct"] / stats["count"]
            
            # Calculate squared error, weighted by number of predictions
            bucket_error = ((bucket - actual_fraction) ** 2) * (stats["count"] / total_predictions)
            calibration_error += bucket_error
        
        # Convert to score (1.0 - error)
        return 1.0 - calibration_error 