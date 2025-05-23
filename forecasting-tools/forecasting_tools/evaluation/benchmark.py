"""
Benchmark script for evaluating forecast accuracy.

This script loads historical forecasts and evaluates their accuracy, 
generating detailed reports and visualizations.
"""

import os
import glob
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecasting_tools.evaluation.scoring import (
    ForecastEvaluator, ReasoningEvaluator, AccuracyMetrics, ScoringRule
)


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def find_forecast_files(fixtures_dir: str) -> List[str]:
    """
    Find all forecast JSON files in the fixtures directory.
    
    Args:
        fixtures_dir: Directory containing forecast fixtures
        
    Returns:
        List of file paths
    """
    # Look for JSON files in fixtures directory
    json_pattern = os.path.join(fixtures_dir, "**", "*.json")
    return glob.glob(json_pattern, recursive=True)


def load_forecast_file(file_path: str) -> Dict[str, Any]:
    """
    Load a forecast file.
    
    Args:
        file_path: Path to forecast file
        
    Returns:
        Loaded forecast data
    """
    with open(file_path, "r") as f:
        return json.load(f)


def create_calibration_plot(
    calibration_data: List[Dict[str, Any]], 
    output_path: str
) -> None:
    """
    Create a calibration plot.
    
    Args:
        calibration_data: Calibration data from evaluator
        output_path: Path to save the plot
    """
    # Extract data
    bin_edges = [item["bin_start"] for item in calibration_data]
    bin_edges.append(calibration_data[-1]["bin_end"])
    bin_midpoints = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)]
    
    mean_predictions = [item["mean_prediction"] for item in calibration_data]
    actual_probabilities = [item["actual_probability"] for item in calibration_data]
    counts = [item["count"] for item in calibration_data]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot calibration curve
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax1.scatter(mean_predictions, actual_probabilities, 
               s=[c * 20 for c in counts],  # Size by count
               alpha=0.7)
    
    # Add labels
    for i, (x, y, count) in enumerate(zip(mean_predictions, actual_probabilities, counts)):
        ax1.annotate(f"{count}", (x, y), textcoords="offset points", 
                    xytext=(0, 5), ha="center")
    
    ax1.set_xlabel("Predicted probability")
    ax1.set_ylabel("Actual probability")
    ax1.set_title("Calibration Curve")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Plot prediction distribution
    ax2.bar(bin_midpoints, counts, width=(bin_edges[1] - bin_edges[0])*0.8, alpha=0.7)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Distribution")
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_distribution_plot(
    predictions: List[Dict[str, Any]],
    output_path: str
) -> None:
    """
    Create a plot showing the distribution of predictions vs outcomes.
    
    Args:
        predictions: List of predictions with distributions and outcomes
        output_path: Path to save the plot
    """
    # Filter to numeric predictions with distributions
    numeric_preds = [
        p for p in predictions 
        if p.get("forecast_type") == "numeric" and 
           "prediction" in p and "distribution" in p["prediction"] and
           "outcome" in p and "value" in p["outcome"]
    ]
    
    if not numeric_preds:
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data for plot
    questions = [p.get("question", f"Question {i+1}") for i, p in enumerate(numeric_preds)]
    medians = [p["prediction"]["distribution"].get("median", 0) for p in numeric_preds]
    p10s = [p["prediction"]["distribution"].get("p10", 0) for p in numeric_preds]
    p90s = [p["prediction"]["distribution"].get("p90", 0) for p in numeric_preds]
    outcomes = [p["outcome"]["value"] for p in numeric_preds]
    
    # Truncate long questions
    short_questions = [q[:30] + "..." if len(q) > 30 else q for q in questions]
    
    # Set up x-axis
    y_pos = range(len(short_questions))
    
    # Plot intervals and points
    for i, (p10, median, p90, outcome) in enumerate(zip(p10s, medians, p90s, outcomes)):
        # Plot interval
        ax.plot([p10, p90], [i, i], color="blue", linewidth=2, alpha=0.7)
        
        # Plot median
        ax.scatter([median], [i], color="blue", s=50, zorder=3)
        
        # Plot outcome
        ax.scatter([outcome], [i], color="red", marker="x", s=100, zorder=3)
    
    # Add labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_questions)
    ax.set_xlabel("Value")
    ax.set_title("Numeric Predictions vs Outcomes")
    
    # Add legend
    ax.scatter([], [], color="blue", label="Median prediction")
    ax.plot([], [], color="blue", label="P10-P90 interval")
    ax.scatter([], [], color="red", marker="x", label="Actual outcome")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_html_report(
    evaluation_results: Dict[str, Any],
    output_dir: str,
    plot_paths: Dict[str, str]
) -> str:
    """
    Generate an HTML report from evaluation results.
    
    Args:
        evaluation_results: Evaluation results from evaluator
        output_dir: Directory to save the report
        plot_paths: Paths to generated plots
        
    Returns:
        Path to the generated HTML report
    """
    # Create HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Forecast Accuracy Benchmark Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
            h1, h2, h3 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .metric-good { color: green; font-weight: bold; }
            .metric-bad { color: red; font-weight: bold; }
            .metric-neutral { color: black; }
            .section { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            img { max-width: 100%; }
        </style>
    </head>
    <body>
        <h1>Forecast Accuracy Benchmark Report</h1>
        <p>Generated on: {date}</p>
    """.format(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Add overall results section
    overall = evaluation_results.get("overall", {})
    html += """
        <div class="section">
            <h2>Overall Results</h2>
    """
    
    # Binary forecasts
    binary = overall.get("binary", {})
    if "error" not in binary:
        html += """
            <h3>Binary Forecasts</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Count</td>
                    <td>{count}</td>
                </tr>
                <tr>
                    <td>Brier Score</td>
                    <td class="{brier_class}">{brier:.4f}</td>
                </tr>
                <tr>
                    <td>Log Score</td>
                    <td class="{log_class}">{log:.4f}</td>
                </tr>
                <tr>
                    <td>Calibration Reliability</td>
                    <td class="{cal_class}">{cal:.4f}</td>
                </tr>
                <tr>
                    <td>Accuracy</td>
                    <td class="{acc_class}">{acc:.2%}</td>
                </tr>
                <tr>
                    <td>Overconfidence</td>
                    <td class="{over_class}">{over:.4f}</td>
                </tr>
            </table>
        """.format(
            count=binary.get("count", 0),
            brier=binary.get("brier_score", 0),
            brier_class="metric-good" if binary.get("brier_score", 1) < 0.2 else "metric-bad",
            log=binary.get("log_score", 0),
            log_class="metric-good" if binary.get("log_score", -1) > -0.5 else "metric-bad",
            cal=binary.get("calibration", {}).get("reliability", 0),
            cal_class="metric-good" if binary.get("calibration", {}).get("reliability", 0) > 0.8 else "metric-bad",
            acc=binary.get("accuracy", 0),
            acc_class="metric-good" if binary.get("accuracy", 0) > 0.7 else "metric-bad",
            over=binary.get("overconfidence", 0),
            over_class="metric-good" if abs(binary.get("overconfidence", 0)) < 0.1 else "metric-bad"
        )
        
        # Add calibration plot if available
        if "calibration_plot" in plot_paths:
            html += """
            <h4>Calibration Plot</h4>
            <img src="{plot_path}" alt="Calibration Plot">
            """.format(plot_path=os.path.basename(plot_paths["calibration_plot"]))
    
    # Numeric forecasts
    numeric = overall.get("numeric", {})
    if "error" not in numeric:
        html += """
            <h3>Numeric Forecasts</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Count</td>
                    <td>{count}</td>
                </tr>
        """.format(count=numeric.get("count", 0))
        
        # Add error metrics if available
        error_metrics = numeric.get("error_metrics", {})
        if error_metrics:
            html += """
                <tr>
                    <td>Mean Absolute Error (MAE)</td>
                    <td>{mae:.4f}</td>
                </tr>
                <tr>
                    <td>Root Mean Square Error (RMSE)</td>
                    <td>{rmse:.4f}</td>
                </tr>
                <tr>
                    <td>Mean Absolute Percentage Error (MAPE)</td>
                    <td>{mape:.2%}</td>
                </tr>
            """.format(
                mae=error_metrics.get("mae", 0),
                rmse=error_metrics.get("rmse", 0),
                mape=error_metrics.get("mape", 0)
            )
        
        # Add distribution metrics if available
        dist_metrics = numeric.get("distribution_metrics", {})
        if dist_metrics:
            html += """
                <tr>
                    <td>90% Interval Coverage</td>
                    <td class="{cov_class}">{cov:.2%}</td>
                </tr>
                <tr>
                    <td>CRPS</td>
                    <td>{crps:.4f}</td>
                </tr>
            """.format(
                cov=dist_metrics.get("coverage_90pct", 0),
                cov_class="metric-good" if abs(dist_metrics.get("coverage_90pct", 0) - 0.9) < 0.1 else "metric-bad",
                crps=dist_metrics.get("crps", 0)
            )
        
        html += """
            </table>
        """
        
        # Add distribution plot if available
        if "distribution_plot" in plot_paths:
            html += """
            <h4>Distribution Plot</h4>
            <img src="{plot_path}" alt="Distribution Plot">
            """.format(plot_path=os.path.basename(plot_paths["distribution_plot"]))
    
    # Multiple choice forecasts
    mc = overall.get("multiple_choice", {})
    if "error" not in mc:
        html += """
            <h3>Multiple Choice Forecasts</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Count</td>
                    <td>{count}</td>
                </tr>
                <tr>
                    <td>Average Probability Assigned to Correct Option</td>
                    <td class="{prob_class}">{prob:.2%}</td>
                </tr>
                <tr>
                    <td>Log Score</td>
                    <td>{log:.4f}</td>
                </tr>
                <tr>
                    <td>Accuracy</td>
                    <td class="{acc_class}">{acc:.2%}</td>
                </tr>
            </table>
        """.format(
            count=mc.get("count", 0),
            prob=mc.get("avg_correct_probability", 0),
            prob_class="metric-good" if mc.get("avg_correct_probability", 0) > 0.5 else "metric-bad",
            log=mc.get("log_score", 0),
            acc=mc.get("accuracy", 0),
            acc_class="metric-good" if mc.get("accuracy", 0) > 0.7 else "metric-bad"
        )
    
    html += """
        </div>
    """
    
    # Add reasoning evaluation section
    reasoning = evaluation_results.get("reasoning", {})
    if "error" not in reasoning:
        html += """
        <div class="section">
            <h2>Reasoning Evaluation</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Count</td>
                    <td>{count}</td>
                </tr>
                <tr>
                    <td>Average Quality Score</td>
                    <td class="{quality_class}">{quality:.2f}</td>
                </tr>
                <tr>
                    <td>Average Word Count</td>
                    <td>{words:.1f}</td>
                </tr>
                <tr>
                    <td>Quantitative Reasoning</td>
                    <td>{quant:.2%}</td>
                </tr>
                <tr>
                    <td>Comparative Reasoning</td>
                    <td>{comp:.2%}</td>
                </tr>
                <tr>
                    <td>Causal Reasoning</td>
                    <td>{causal:.2%}</td>
                </tr>
                <tr>
                    <td>Conditional Reasoning</td>
                    <td>{cond:.2%}</td>
                </tr>
            </table>
        </div>
        """.format(
            count=reasoning.get("count", 0),
            quality=reasoning.get("average_quality_score", 0),
            quality_class="metric-good" if reasoning.get("average_quality_score", 0) > 0.7 else "metric-bad",
            words=reasoning.get("average_word_count", 0),
            quant=reasoning.get("quantitative_reasoning_percentage", 0),
            comp=reasoning.get("comparative_reasoning_percentage", 0),
            causal=reasoning.get("causal_reasoning_percentage", 0),
            cond=reasoning.get("conditional_reasoning_percentage", 0)
        )
    
    # Add by category section
    by_category = evaluation_results.get("by_category", {})
    if by_category:
        html += """
        <div class="section">
            <h2>Results by Category</h2>
            <table>
                <tr>
                    <th>Category</th>
                    <th>Binary Count</th>
                    <th>Binary Brier Score</th>
                    <th>Binary Accuracy</th>
                    <th>Numeric Count</th>
                    <th>Numeric RMSE</th>
                    <th>Multiple Choice Count</th>
                    <th>Multiple Choice Accuracy</th>
                </tr>
        """
        
        for category, results in by_category.items():
            binary = results.get("binary", {})
            numeric = results.get("numeric", {})
            mc = results.get("multiple_choice", {})
            
            html += """
                <tr>
                    <td>{category}</td>
                    <td>{binary_count}</td>
                    <td>{brier:.4f}</td>
                    <td>{binary_acc:.2%}</td>
                    <td>{numeric_count}</td>
                    <td>{rmse:.4f}</td>
                    <td>{mc_count}</td>
                    <td>{mc_acc:.2%}</td>
                </tr>
            """.format(
                category=category,
                binary_count=binary.get("count", 0) if "error" not in binary else 0,
                brier=binary.get("brier_score", 0) if "error" not in binary else 0,
                binary_acc=binary.get("accuracy", 0) if "error" not in binary else 0,
                numeric_count=numeric.get("count", 0) if "error" not in numeric else 0,
                rmse=numeric.get("error_metrics", {}).get("rmse", 0) if "error" not in numeric else 0,
                mc_count=mc.get("count", 0) if "error" not in mc else 0,
                mc_acc=mc.get("accuracy", 0) if "error" not in mc else 0
            )
        
        html += """
            </table>
        </div>
        """
    
    # Close HTML
    html += """
    </body>
    </html>
    """
    
    # Write HTML to file
    output_path = os.path.join(output_dir, "benchmark_report.html")
    with open(output_path, "w") as f:
        f.write(html)
    
    return output_path


def run_benchmark(fixtures_dir: str, output_dir: str) -> None:
    """
    Run benchmark evaluation on forecast fixtures.
    
    Args:
        fixtures_dir: Directory containing forecast fixtures
        output_dir: Directory to save benchmark results
    """
    # Set up logging
    setup_logging()
    logger = logging.getLogger("benchmark")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find forecast files
    forecast_files = find_forecast_files(fixtures_dir)
    logger.info(f"Found {len(forecast_files)} forecast file(s)")
    
    if not forecast_files:
        logger.error(f"No forecast files found in {fixtures_dir}")
        return
    
    # Process each file
    all_results = {}
    plot_paths = {}
    
    for file_path in forecast_files:
        file_name = os.path.basename(file_path)
        logger.info(f"Processing {file_name}")
        
        # Load and evaluate forecasts
        evaluator = ForecastEvaluator(file_path)
        results = evaluator.generate_report()
        
        # Add reasoning evaluation
        reasoning_evaluator = ReasoningEvaluator()
        results["reasoning"] = reasoning_evaluator.batch_evaluate_reasoning(evaluator.forecasts)
        
        # Save results as JSON
        output_json = os.path.join(output_dir, f"{file_name}_evaluation.json")
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        
        # Create calibration plot if we have binary forecasts
        calibration_data = results.get("calibration_data", {}).get("binary")
        if calibration_data:
            plot_path = os.path.join(output_dir, f"{file_name}_calibration.png")
            create_calibration_plot(calibration_data, plot_path)
            plot_paths["calibration_plot"] = plot_path
            logger.info(f"Created calibration plot: {plot_path}")
        
        # Create distribution plot if we have numeric forecasts
        forecasts = evaluator.forecasts
        if any(f.get("forecast_type") == "numeric" for f in forecasts):
            plot_path = os.path.join(output_dir, f"{file_name}_distribution.png")
            create_distribution_plot(forecasts, plot_path)
            plot_paths["distribution_plot"] = plot_path
            logger.info(f"Created distribution plot: {plot_path}")
        
        # Generate HTML report
        html_path = generate_html_report(results, output_dir, plot_paths)
        logger.info(f"Generated HTML report: {html_path}")
        
        all_results[file_name] = results
    
    # Generate combined results if we processed multiple files
    if len(all_results) > 1:
        combined_path = os.path.join(output_dir, "combined_results.json")
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Generated combined results: {combined_path}")
    
    logger.info("Benchmark complete")


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(description="Forecast accuracy benchmark")
    parser.add_argument(
        "--fixtures-dir",
        type=str,
        default="tests/fixtures",
        help="Directory containing forecast fixtures"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark-results",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Run benchmark
    run_benchmark(args.fixtures_dir, args.output_dir)


if __name__ == "__main__":
    main() 