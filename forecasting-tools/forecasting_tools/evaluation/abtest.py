"""
A/B Testing Framework for Forecasting Approaches

This module provides a framework for comparing different forecasting approaches
through controlled experiments and statistical analysis.
"""

import json
import random
import statistics
import logging
import uuid
from typing import Dict, List, Any, Callable, Optional, Union, Tuple
from enum import Enum
import datetime
import pandas as pd
import numpy as np
from scipy import stats

from forecasting_tools.evaluation.scoring import (
    AccuracyMetrics, ForecastEvaluator, ScoringRule, calculate_proper_scoring_rule
)


class VariantType(str, Enum):
    """Types of experiment variants."""
    CONTROL = "control"
    TREATMENT = "treatment"


class ExperimentStatus(str, Enum):
    """Status of an experiment."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ANALYZED = "analyzed"


class ForecastingExperiment:
    """
    An experiment comparing different forecasting approaches.
    
    This class manages the experiment configuration, assignment of questions,
    data collection, and result analysis.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        variants: Dict[str, Dict[str, Any]],
        traffic_split: Optional[Dict[str, float]] = None,
        metrics: Optional[List[ScoringRule]] = None,
        min_samples: int = 30,
        confidence_level: float = 0.95
    ):
        """
        Initialize a forecasting experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            variants: Dict mapping variant names to their configurations
            traffic_split: Dict mapping variant names to their traffic allocation (0-1)
            metrics: List of metrics to evaluate variants on
            min_samples: Minimum number of samples per variant
            confidence_level: Confidence level for statistical tests (0-1)
        """
        self.name = name
        self.description = description
        self.id = str(uuid.uuid4())
        self.created_at = datetime.datetime.now().isoformat()
        self.status = ExperimentStatus.CREATED
        
        # Validate and store variants
        if len(variants) < 2:
            raise ValueError("At least two variants (control and treatment) are required")
            
        self.variants = variants
        
        # Set default traffic split if not provided (equal split)
        if traffic_split is None:
            equal_split = 1.0 / len(variants)
            self.traffic_split = {name: equal_split for name in variants.keys()}
        else:
            # Validate traffic split
            if sum(traffic_split.values()) != 1.0:
                raise ValueError("Traffic split must sum to 1.0")
            if not all(0 <= split <= 1 for split in traffic_split.values()):
                raise ValueError("Traffic split values must be between 0 and 1")
            self.traffic_split = traffic_split
        
        # Set metrics
        self.metrics = metrics or [
            ScoringRule.BRIER, 
            ScoringRule.CALIBRATION, 
            ScoringRule.ACCURACY
        ]
        
        # Statistical parameters
        self.min_samples = min_samples
        self.confidence_level = confidence_level
        
        # Results storage
        self.results = {
            variant_name: [] for variant_name in variants.keys()
        }
        
        # Analysis results
        self.analysis = {}
        
        # Logging
        self.logger = logging.getLogger(f"experiment.{name}")
        self.logger.info(f"Created experiment: {name}")
    
    def assign_variant(self, question_id: str) -> str:
        """
        Assign a variant to a question.
        
        Uses consistent hashing to ensure the same question always gets the same variant,
        while maintaining the desired traffic split.
        
        Args:
            question_id: Unique identifier for the question
            
        Returns:
            Assigned variant name
        """
        # Use the question_id to seed random for consistent assignment
        random.seed(question_id)
        
        # Simple weighted random selection
        r = random.random()
        cumulative = 0.0
        
        for variant_name, split in self.traffic_split.items():
            cumulative += split
            if r <= cumulative:
                return variant_name
        
        # Fallback to the last variant if there's any floating-point precision issue
        return list(self.variants.keys())[-1]
    
    def start(self) -> None:
        """Start the experiment."""
        if self.status != ExperimentStatus.CREATED and self.status != ExperimentStatus.PAUSED:
            raise ValueError(f"Cannot start experiment in {self.status} status")
        
        self.status = ExperimentStatus.RUNNING
        self.logger.info(f"Started experiment: {self.name}")
    
    def pause(self) -> None:
        """Pause the experiment."""
        if self.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Cannot pause experiment in {self.status} status")
        
        self.status = ExperimentStatus.PAUSED
        self.logger.info(f"Paused experiment: {self.name}")
    
    def complete(self) -> None:
        """Mark the experiment as completed."""
        if self.status != ExperimentStatus.RUNNING and self.status != ExperimentStatus.PAUSED:
            raise ValueError(f"Cannot complete experiment in {self.status} status")
        
        self.status = ExperimentStatus.COMPLETED
        self.logger.info(f"Completed experiment: {self.name}")
    
    def record_result(
        self,
        variant_name: str,
        question_id: str,
        question: str,
        probability: float,
        outcome: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a result for a variant.
        
        Args:
            variant_name: Name of the variant
            question_id: Unique identifier for the question
            question: The forecast question
            probability: Forecast probability
            outcome: Actual outcome (True/False)
            metadata: Additional metadata about the result
        """
        if variant_name not in self.variants:
            raise ValueError(f"Unknown variant: {variant_name}")
        
        if self.status != ExperimentStatus.RUNNING:
            self.logger.warning(f"Recording result while experiment is in {self.status} status")
        
        result = {
            "question_id": question_id,
            "question": question,
            "probability": probability,
            "outcome": outcome,
            "timestamp": datetime.datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.results[variant_name].append(result)
        self.logger.debug(f"Recorded result for {variant_name}: {question_id}")
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the experiment results.
        
        Compares variants using the specified metrics and performs
        statistical significance testing.
        
        Returns:
            Analysis results
        """
        if self.status != ExperimentStatus.COMPLETED:
            self.logger.warning(f"Analyzing experiment in {self.status} status")
        
        # Check if we have enough samples
        for variant_name, results in self.results.items():
            if len(results) < self.min_samples:
                self.logger.warning(
                    f"Variant {variant_name} has only {len(results)} samples, "
                    f"which is less than the minimum {self.min_samples}"
                )
        
        # Analyze metrics for each variant
        variant_metrics = {}
        
        for variant_name, results in self.results.items():
            if not results:
                continue
                
            # Extract probabilities and outcomes
            probabilities = [r["probability"] for r in results]
            outcomes = [r["outcome"] for r in results]
            
            # Calculate metrics
            metrics_values = {}
            for metric in self.metrics:
                metrics_values[metric.value] = calculate_proper_scoring_rule(
                    probabilities, outcomes, metric
                )
            
            # Add standard metrics
            metrics_values["sample_size"] = len(results)
            metrics_values["average_probability"] = statistics.mean(probabilities)
            metrics_values["outcome_rate"] = sum(outcomes) / len(outcomes)
            
            variant_metrics[variant_name] = metrics_values
        
        # Compare variants
        comparisons = {}
        
        # Find control variant
        control_variant = next(
            (name for name, config in self.variants.items() 
             if config.get("type") == VariantType.CONTROL),
            list(self.variants.keys())[0]  # Default to first variant if no control specified
        )
        
        # Compare each treatment to control
        for variant_name, metrics in variant_metrics.items():
            if variant_name == control_variant:
                continue
                
            variant_comparisons = {}
            
            # For each metric, perform statistical test
            for metric_name, metric_value in metrics.items():
                if metric_name in ("sample_size", "average_probability", "outcome_rate"):
                    continue
                    
                control_value = variant_metrics[control_variant][metric_name]
                
                # Absolute difference
                diff = metric_value - control_value
                
                # Relative difference (percentage)
                rel_diff = diff / abs(control_value) if control_value != 0 else float('inf')
                
                # Perform statistical test (t-test for now)
                treatment_data = [
                    calculate_proper_scoring_rule(
                        [r["probability"]], [r["outcome"]], 
                        ScoringRule(metric_name)
                    )
                    for r in self.results[variant_name]
                ]
                
                control_data = [
                    calculate_proper_scoring_rule(
                        [r["probability"]], [r["outcome"]], 
                        ScoringRule(metric_name)
                    )
                    for r in self.results[control_variant]
                ]
                
                t_stat, p_value = stats.ttest_ind(
                    treatment_data, 
                    control_data,
                    equal_var=False  # Welch's t-test (doesn't assume equal variance)
                )
                
                # Determine if difference is statistically significant
                alpha = 1 - self.confidence_level
                is_significant = p_value < alpha
                
                # Determine if result is positive (improvement) or negative
                is_positive = diff > 0
                
                variant_comparisons[metric_name] = {
                    "control_value": control_value,
                    "treatment_value": metric_value,
                    "absolute_diff": diff,
                    "relative_diff": rel_diff,
                    "p_value": p_value,
                    "is_significant": is_significant,
                    "is_positive": is_positive,
                    "confidence_level": self.confidence_level
                }
            
            comparisons[variant_name] = variant_comparisons
        
        # Store analysis results
        self.analysis = {
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": variant_metrics,
            "comparisons": comparisons,
            "sample_sizes": {variant: len(results) for variant, results in self.results.items()}
        }
        
        self.status = ExperimentStatus.ANALYZED
        self.logger.info(f"Analyzed experiment: {self.name}")
        
        return self.analysis
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the experiment to a dictionary.
        
        Returns:
            Dictionary representation of the experiment
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "status": self.status.value,
            "variants": self.variants,
            "traffic_split": self.traffic_split,
            "metrics": [m.value for m in self.metrics],
            "min_samples": self.min_samples,
            "confidence_level": self.confidence_level,
            "sample_sizes": {variant: len(results) for variant, results in self.results.items()},
            "analysis": self.analysis
        }
    
    def to_json(self) -> str:
        """
        Convert the experiment to a JSON string.
        
        Returns:
            JSON representation of the experiment
        """
        return json.dumps(self.to_dict(), indent=2)
    
    def save(self, file_path: str) -> None:
        """
        Save the experiment to a file.
        
        Args:
            file_path: Path to save the experiment to
        """
        with open(file_path, "w") as f:
            f.write(self.to_json())
        
        self.logger.info(f"Saved experiment to {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> "ForecastingExperiment":
        """
        Load an experiment from a file.
        
        Args:
            file_path: Path to load the experiment from
            
        Returns:
            Loaded experiment
        """
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Create experiment instance
        experiment = cls(
            name=data["name"],
            description=data["description"],
            variants=data["variants"],
            traffic_split=data["traffic_split"],
            metrics=[ScoringRule(m) for m in data["metrics"]],
            min_samples=data["min_samples"],
            confidence_level=data["confidence_level"]
        )
        
        # Restore experiment state
        experiment.id = data["id"]
        experiment.created_at = data["created_at"]
        experiment.status = ExperimentStatus(data["status"])
        experiment.analysis = data.get("analysis", {})
        
        # Log info
        experiment.logger.info(f"Loaded experiment: {experiment.name}")
        
        return experiment
    
    def generate_report(self) -> pd.DataFrame:
        """
        Generate a DataFrame report of the experiment results.
        
        Returns:
            DataFrame with experiment results
        """
        if not self.analysis:
            raise ValueError("Experiment has not been analyzed yet")
        
        # Create DataFrame from results
        data = []
        
        for variant_name, results in self.results.items():
            for result in results:
                row = {
                    "variant": variant_name,
                    "question_id": result["question_id"],
                    "question": result["question"],
                    "probability": result["probability"],
                    "outcome": result["outcome"],
                    "timestamp": result["timestamp"]
                }
                
                # Add variant type
                variant_type = self.variants[variant_name].get("type", "unknown")
                row["variant_type"] = variant_type
                
                # Add metadata
                for k, v in result.get("metadata", {}).items():
                    row[f"metadata_{k}"] = v
                
                data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        if not df.empty:
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        return df


class ExperimentManager:
    """
    Manages multiple forecasting experiments.
    
    Provides functions for creating, tracking, and analyzing experiments.
    """
    
    def __init__(self):
        """Initialize the experiment manager."""
        self.experiments = {}
        self.logger = logging.getLogger("experiment_manager")
    
    def create_experiment(
        self,
        name: str,
        description: str,
        variants: Dict[str, Dict[str, Any]],
        traffic_split: Optional[Dict[str, float]] = None,
        metrics: Optional[List[ScoringRule]] = None,
        min_samples: int = 30,
        confidence_level: float = 0.95,
        start: bool = False
    ) -> ForecastingExperiment:
        """
        Create a new experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            variants: Dict mapping variant names to their configurations
            traffic_split: Dict mapping variant names to their traffic allocation (0-1)
            metrics: List of metrics to evaluate variants on
            min_samples: Minimum number of samples per variant
            confidence_level: Confidence level for statistical tests (0-1)
            start: Whether to start the experiment immediately
            
        Returns:
            Created experiment
        """
        # Check if experiment with this name already exists
        if name in self.experiments:
            raise ValueError(f"Experiment with name '{name}' already exists")
        
        # Create experiment
        experiment = ForecastingExperiment(
            name=name,
            description=description,
            variants=variants,
            traffic_split=traffic_split,
            metrics=metrics,
            min_samples=min_samples,
            confidence_level=confidence_level
        )
        
        # Start if requested
        if start:
            experiment.start()
        
        # Add to experiments dict
        self.experiments[name] = experiment
        
        self.logger.info(f"Created experiment: {name}")
        return experiment
    
    def get_experiment(self, name: str) -> ForecastingExperiment:
        """
        Get an experiment by name.
        
        Args:
            name: Experiment name
            
        Returns:
            Experiment instance
        """
        if name not in self.experiments:
            raise ValueError(f"Experiment '{name}' not found")
        
        return self.experiments[name]
    
    def list_experiments(self) -> Dict[str, Dict[str, Any]]:
        """
        List all experiments.
        
        Returns:
            Dict mapping experiment names to their basic information
        """
        return {
            name: {
                "id": exp.id,
                "name": exp.name,
                "description": exp.description,
                "status": exp.status.value,
                "created_at": exp.created_at,
                "variants": list(exp.variants.keys()),
                "sample_sizes": {variant: len(results) for variant, results in exp.results.items()}
            }
            for name, exp in self.experiments.items()
        }
    
    def record_forecast(
        self,
        experiment_name: str,
        question_id: str,
        question: str,
        probability: float,
        outcome: bool,
        metadata: Optional[Dict[str, Any]] = None,
        variant_name: Optional[str] = None
    ) -> None:
        """
        Record a forecast result in an experiment.
        
        If variant_name is not provided, assigns a variant based on the
        experiment's traffic split configuration.
        
        Args:
            experiment_name: Name of the experiment
            question_id: Unique identifier for the question
            question: The forecast question
            probability: Forecast probability
            outcome: Actual outcome (True/False)
            metadata: Additional metadata about the result
            variant_name: Optional variant to assign (overrides automatic assignment)
        """
        experiment = self.get_experiment(experiment_name)
        
        # Assign variant if not provided
        if variant_name is None:
            variant_name = experiment.assign_variant(question_id)
        
        # Record result
        experiment.record_result(
            variant_name=variant_name,
            question_id=question_id,
            question=question,
            probability=probability,
            outcome=outcome,
            metadata=metadata
        )
    
    def save_all_experiments(self, directory: str) -> None:
        """
        Save all experiments to files in the specified directory.
        
        Args:
            directory: Directory to save experiments to
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save each experiment
        for name, experiment in self.experiments.items():
            file_path = os.path.join(directory, f"{name}.json")
            experiment.save(file_path)
        
        self.logger.info(f"Saved {len(self.experiments)} experiments to {directory}")
    
    def load_experiments(self, directory: str) -> None:
        """
        Load experiments from files in the specified directory.
        
        Args:
            directory: Directory to load experiments from
        """
        import os
        import glob
        
        # Find experiment JSON files
        pattern = os.path.join(directory, "*.json")
        files = glob.glob(pattern)
        
        # Load each experiment
        for file_path in files:
            try:
                experiment = ForecastingExperiment.load(file_path)
                self.experiments[experiment.name] = experiment
            except Exception as e:
                self.logger.error(f"Error loading experiment from {file_path}: {e}")
        
        self.logger.info(f"Loaded {len(files)} experiments from {directory}")


def create_ab_test(
    control_config: Dict[str, Any],
    treatment_config: Dict[str, Any],
    name: str = "forecast_ab_test",
    description: str = "A/B test for forecasting approaches",
    traffic_split: Optional[Dict[str, float]] = None,
    metrics: Optional[List[ScoringRule]] = None
) -> ForecastingExperiment:
    """
    Convenience function to create a simple A/B test with one control and one treatment.
    
    Args:
        control_config: Configuration for the control variant
        treatment_config: Configuration for the treatment variant
        name: Experiment name
        description: Experiment description
        traffic_split: Dict mapping variant names to traffic allocation
        metrics: List of metrics to evaluate variants on
        
    Returns:
        Created experiment
    """
    # Set default traffic split if not provided (50/50)
    if traffic_split is None:
        traffic_split = {"control": 0.5, "treatment": 0.5}
    
    # Set variant types
    control_config["type"] = VariantType.CONTROL
    treatment_config["type"] = VariantType.TREATMENT
    
    # Create variants dictionary
    variants = {
        "control": control_config,
        "treatment": treatment_config
    }
    
    # Create experiment
    experiment = ForecastingExperiment(
        name=name,
        description=description,
        variants=variants,
        traffic_split=traffic_split,
        metrics=metrics
    )
    
    return experiment 