import unittest
import os
import tempfile
import json
from datetime import datetime

import pytest
import pandas as pd

from forecasting_tools.evaluation.abtest import (
    ForecastingExperiment, ExperimentManager, VariantType, 
    ExperimentStatus, create_ab_test
)
from forecasting_tools.evaluation.scoring import ScoringRule


class TestForecastingExperiment(unittest.TestCase):
    """Tests for the ForecastingExperiment class."""
    
    def setUp(self):
        """Set up test environment."""
        # Define variants for testing
        self.variants = {
            "control": {
                "type": VariantType.CONTROL,
                "model": "baseline",
                "config": {"param1": "value1"}
            },
            "treatment": {
                "type": VariantType.TREATMENT,
                "model": "improved",
                "config": {"param1": "value2"}
            }
        }
        
        # Create an experiment
        self.experiment = ForecastingExperiment(
            name="test_experiment",
            description="Test experiment",
            variants=self.variants,
            metrics=[ScoringRule.BRIER, ScoringRule.ACCURACY]
        )
    
    def test_experiment_initialization(self):
        """Test experiment initialization."""
        # Check attributes
        self.assertEqual(self.experiment.name, "test_experiment")
        self.assertEqual(self.experiment.description, "Test experiment")
        self.assertEqual(self.experiment.variants, self.variants)
        self.assertEqual(self.experiment.status, ExperimentStatus.CREATED)
        
        # Check traffic split (should be 50/50 by default)
        self.assertEqual(self.experiment.traffic_split, {"control": 0.5, "treatment": 0.5})
        
        # Check metrics
        self.assertEqual(len(self.experiment.metrics), 2)
        self.assertEqual(self.experiment.metrics[0], ScoringRule.BRIER)
        self.assertEqual(self.experiment.metrics[1], ScoringRule.ACCURACY)
    
    def test_variant_assignment(self):
        """Test variant assignment."""
        # Test consistency (same question ID should always get same variant)
        question_id = "test_question_1"
        variant1 = self.experiment.assign_variant(question_id)
        variant2 = self.experiment.assign_variant(question_id)
        self.assertEqual(variant1, variant2)
        
        # Test distribution with a large sample
        variants_count = {"control": 0, "treatment": 0}
        for i in range(1000):
            variant = self.experiment.assign_variant(f"question_{i}")
            variants_count[variant] += 1
        
        # Should be roughly 50/50 (allow for some random variation)
        control_ratio = variants_count["control"] / 1000
        self.assertGreater(control_ratio, 0.45)
        self.assertLess(control_ratio, 0.55)
    
    def test_experiment_lifecycle(self):
        """Test experiment lifecycle state changes."""
        # Initial status
        self.assertEqual(self.experiment.status, ExperimentStatus.CREATED)
        
        # Start
        self.experiment.start()
        self.assertEqual(self.experiment.status, ExperimentStatus.RUNNING)
        
        # Pause
        self.experiment.pause()
        self.assertEqual(self.experiment.status, ExperimentStatus.PAUSED)
        
        # Start again
        self.experiment.start()
        self.assertEqual(self.experiment.status, ExperimentStatus.RUNNING)
        
        # Complete
        self.experiment.complete()
        self.assertEqual(self.experiment.status, ExperimentStatus.COMPLETED)
        
        # Should not be able to start once completed
        with self.assertRaises(ValueError):
            self.experiment.start()
    
    def test_record_result(self):
        """Test recording results."""
        # Start the experiment
        self.experiment.start()
        
        # Record a result
        self.experiment.record_result(
            variant_name="control",
            question_id="q1",
            question="Will event X happen?",
            probability=0.7,
            outcome=True,
            metadata={"source": "test"}
        )
        
        # Check that the result was recorded
        self.assertEqual(len(self.experiment.results["control"]), 1)
        self.assertEqual(len(self.experiment.results["treatment"]), 0)
        
        # Check the result content
        result = self.experiment.results["control"][0]
        self.assertEqual(result["question_id"], "q1")
        self.assertEqual(result["question"], "Will event X happen?")
        self.assertEqual(result["probability"], 0.7)
        self.assertEqual(result["outcome"], True)
        self.assertEqual(result["metadata"], {"source": "test"})
    
    def test_analyze_without_enough_samples(self):
        """Test analyzing with insufficient samples."""
        # Start and complete the experiment
        self.experiment.start()
        self.experiment.complete()
        
        # Record just a few results
        self.experiment.record_result(
            variant_name="control",
            question_id="q1",
            question="Will event X happen?",
            probability=0.7,
            outcome=True
        )
        self.experiment.record_result(
            variant_name="treatment",
            question_id="q1",
            question="Will event X happen?",
            probability=0.8,
            outcome=True
        )
        
        # Should still run but log warnings
        analysis = self.experiment.analyze()
        
        # Check that analysis ran
        self.assertEqual(self.experiment.status, ExperimentStatus.ANALYZED)
        self.assertIn("metrics", analysis)
        self.assertIn("comparisons", analysis)
    
    def test_analyze_with_samples(self):
        """Test analyzing with sufficient samples."""
        # Start and complete the experiment
        self.experiment.start()
        
        # Record 50 results for each variant (better and worse)
        for i in range(50):
            # Control: 70% accurate
            control_outcome = i < 35  # 35/50 = 70% true outcomes
            self.experiment.record_result(
                variant_name="control",
                question_id=f"q{i}",
                question=f"Will event {i} happen?",
                probability=0.7,
                outcome=control_outcome
            )
            
            # Treatment: 80% accurate
            treatment_outcome = i < 40  # 40/50 = 80% true outcomes
            self.experiment.record_result(
                variant_name="treatment",
                question_id=f"q{i}",
                question=f"Will event {i} happen?",
                probability=0.8,
                outcome=treatment_outcome
            )
        
        self.experiment.complete()
        analysis = self.experiment.analyze()
        
        # Check metrics
        self.assertIn("metrics", analysis)
        self.assertIn("control", analysis["metrics"])
        self.assertIn("treatment", analysis["metrics"])
        
        # Check comparisons
        self.assertIn("comparisons", analysis)
        self.assertIn("treatment", analysis["comparisons"])
        
        # Treatment should have better accuracy
        accuracy_comparison = analysis["comparisons"]["treatment"].get("accuracy", {})
        self.assertGreater(
            accuracy_comparison.get("treatment_value", 0),
            accuracy_comparison.get("control_value", 0)
        )
    
    def test_save_and_load(self):
        """Test saving and loading an experiment."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Start experiment and record some results
            self.experiment.start()
            self.experiment.record_result(
                variant_name="control",
                question_id="q1",
                question="Will event X happen?",
                probability=0.7,
                outcome=True
            )
            
            # Save the experiment
            self.experiment.save(temp_path)
            
            # Load the experiment
            loaded_experiment = ForecastingExperiment.load(temp_path)
            
            # Check that key attributes match
            self.assertEqual(loaded_experiment.id, self.experiment.id)
            self.assertEqual(loaded_experiment.name, self.experiment.name)
            self.assertEqual(loaded_experiment.status, self.experiment.status)
            self.assertEqual(loaded_experiment.variants, self.experiment.variants)
            
            # Check traffic split and metrics
            self.assertEqual(loaded_experiment.traffic_split, self.experiment.traffic_split)
            self.assertEqual(
                [m.value for m in loaded_experiment.metrics], 
                [m.value for m in self.experiment.metrics]
            )
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_generate_report(self):
        """Test generating a report."""
        # Start experiment and record some results
        self.experiment.start()
        
        # Add some results
        for i in range(5):
            for variant in ["control", "treatment"]:
                self.experiment.record_result(
                    variant_name=variant,
                    question_id=f"q{i}",
                    question=f"Will event {i} happen?",
                    probability=0.6 if variant == "control" else 0.7,
                    outcome=i % 2 == 0,  # Alternating outcomes
                    metadata={"source": "test", "importance": i}
                )
        
        # Complete and analyze
        self.experiment.complete()
        self.experiment.analyze()
        
        # Generate report
        df = self.experiment.generate_report()
        
        # Check DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 10)  # 5 results for each variant
        
        # Check columns
        expected_columns = [
            "variant", "question_id", "question", "probability", 
            "outcome", "timestamp", "variant_type", 
            "metadata_source", "metadata_importance"
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)


class TestExperimentManager(unittest.TestCase):
    """Tests for the ExperimentManager class."""
    
    def setUp(self):
        """Set up test environment."""
        self.manager = ExperimentManager()
        
        # Define variants for testing
        self.variants = {
            "control": {
                "type": VariantType.CONTROL,
                "model": "baseline"
            },
            "treatment": {
                "type": VariantType.TREATMENT,
                "model": "improved"
            }
        }
    
    def test_create_experiment(self):
        """Test creating an experiment."""
        # Create an experiment
        experiment = self.manager.create_experiment(
            name="test_experiment",
            description="Test experiment",
            variants=self.variants,
            start=True
        )
        
        # Check that the experiment was created
        self.assertEqual(experiment.name, "test_experiment")
        self.assertEqual(experiment.status, ExperimentStatus.RUNNING)
        
        # Check that it's in the manager
        self.assertIn("test_experiment", self.manager.experiments)
        
        # Should not be able to create another with same name
        with self.assertRaises(ValueError):
            self.manager.create_experiment(
                name="test_experiment",
                description="Duplicate experiment",
                variants=self.variants
            )
    
    def test_get_experiment(self):
        """Test getting an experiment."""
        # Create an experiment
        self.manager.create_experiment(
            name="test_experiment",
            description="Test experiment",
            variants=self.variants
        )
        
        # Get the experiment
        experiment = self.manager.get_experiment("test_experiment")
        self.assertEqual(experiment.name, "test_experiment")
        
        # Should raise for non-existent experiment
        with self.assertRaises(ValueError):
            self.manager.get_experiment("non_existent")
    
    def test_list_experiments(self):
        """Test listing experiments."""
        # Create a couple of experiments
        self.manager.create_experiment(
            name="test1",
            description="Test experiment 1",
            variants=self.variants
        )
        self.manager.create_experiment(
            name="test2",
            description="Test experiment 2",
            variants=self.variants
        )
        
        # List experiments
        experiments = self.manager.list_experiments()
        
        # Check result
        self.assertEqual(len(experiments), 2)
        self.assertIn("test1", experiments)
        self.assertIn("test2", experiments)
        self.assertEqual(experiments["test1"]["description"], "Test experiment 1")
        self.assertEqual(experiments["test2"]["description"], "Test experiment 2")
    
    def test_record_forecast(self):
        """Test recording a forecast."""
        # Create an experiment
        self.manager.create_experiment(
            name="test_experiment",
            description="Test experiment",
            variants=self.variants,
            start=True
        )
        
        # Record a forecast with auto-assignment
        self.manager.record_forecast(
            experiment_name="test_experiment",
            question_id="q1",
            question="Will event X happen?",
            probability=0.7,
            outcome=True
        )
        
        # Record with explicit variant
        self.manager.record_forecast(
            experiment_name="test_experiment",
            question_id="q2",
            question="Will event Y happen?",
            probability=0.6,
            outcome=False,
            variant_name="treatment"
        )
        
        # Check that forecasts were recorded
        experiment = self.manager.get_experiment("test_experiment")
        total_results = len(experiment.results["control"]) + len(experiment.results["treatment"])
        self.assertEqual(total_results, 2)
    
    def test_save_and_load_experiments(self):
        """Test saving and loading all experiments."""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some experiments
            self.manager.create_experiment(
                name="test1",
                description="Test experiment 1",
                variants=self.variants,
                start=True
            )
            self.manager.create_experiment(
                name="test2",
                description="Test experiment 2",
                variants=self.variants
            )
            
            # Record some results
            self.manager.record_forecast(
                experiment_name="test1",
                question_id="q1",
                question="Will event X happen?",
                probability=0.7,
                outcome=True
            )
            
            # Save all experiments
            self.manager.save_all_experiments(temp_dir)
            
            # Check that files were created
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "test1.json")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "test2.json")))
            
            # Create a new manager and load experiments
            new_manager = ExperimentManager()
            new_manager.load_experiments(temp_dir)
            
            # Check that experiments were loaded
            self.assertEqual(len(new_manager.experiments), 2)
            self.assertIn("test1", new_manager.experiments)
            self.assertIn("test2", new_manager.experiments)
            
            # Check that results were loaded
            test1 = new_manager.get_experiment("test1")
            total_results = sum(len(results) for results in test1.results.values())
            self.assertEqual(total_results, 1)


class TestCreateABTest(unittest.TestCase):
    """Tests for the create_ab_test convenience function."""
    
    def test_create_ab_test(self):
        """Test creating an A/B test."""
        # Create configurations
        control_config = {
            "model": "baseline",
            "params": {"temperature": 0.7}
        }
        treatment_config = {
            "model": "improved",
            "params": {"temperature": 0.5}
        }
        
        # Create an A/B test
        experiment = create_ab_test(
            control_config=control_config,
            treatment_config=treatment_config,
            name="test_ab_test",
            description="Test A/B test"
        )
        
        # Check experiment
        self.assertEqual(experiment.name, "test_ab_test")
        self.assertEqual(experiment.description, "Test A/B test")
        
        # Check variants
        self.assertEqual(len(experiment.variants), 2)
        self.assertIn("control", experiment.variants)
        self.assertIn("treatment", experiment.variants)
        
        # Check type was added
        self.assertEqual(experiment.variants["control"]["type"], VariantType.CONTROL)
        self.assertEqual(experiment.variants["treatment"]["type"], VariantType.TREATMENT)
        
        # Check configuration was preserved
        self.assertEqual(experiment.variants["control"]["model"], "baseline")
        self.assertEqual(experiment.variants["treatment"]["model"], "improved")
        self.assertEqual(experiment.variants["control"]["params"]["temperature"], 0.7)
        self.assertEqual(experiment.variants["treatment"]["params"]["temperature"], 0.5)
        
        # Check traffic split
        self.assertEqual(experiment.traffic_split, {"control": 0.5, "treatment": 0.5})


if __name__ == "__main__":
    unittest.main() 