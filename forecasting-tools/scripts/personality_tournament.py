#!/usr/bin/env python
"""
Personality Tournament

This script runs a tournament to compare the performance of different
personality configurations across various forecasting questions.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add forecasting-tools to path
sys.path.append(str(Path(__file__).parent.parent))

from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_bots.basic_bot import BasicBot
from forecasting_tools.forecast_bots.research_bot import ResearchBot
from forecasting_tools.forecast_bots.calibrated_bot import CalibratedBot
from forecasting_tools.forecast_bots.economist_bot import EconomistBot
from forecasting_tools.forecast_bots.bayesian_bot import BayesianBot
from forecasting_tools.forecast_bots.ensemble_bot import EnsembleBot
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    NumericQuestion,
    MultipleChoiceQuestion,
    MetaculusQuestion
)
from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.diversity import (
    PersonalityDiversityScorer,
    calculate_ensemble_diversity_score
)
from forecasting_tools.forecast_helpers.competition import (
    CompetitionTracker,
    CompetitionMetric
)
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi


logger = logging.getLogger(__name__)


async def run_tournament(
    question_ids: List[str],
    personality_names: List[str],
    bot_types: List[str],
    output_dir: str,
    competition_name: str = "personality_tournament",
    num_questions: int = None,
    resume: bool = False,
    save_reports: bool = True
) -> CompetitionTracker:
    """
    Run a tournament comparing personalities across questions.
    
    Args:
        question_ids: List of question IDs to evaluate
        personality_names: List of personality names to evaluate
        bot_types: List of bot types to use
        output_dir: Directory to save results
        competition_name: Name of the competition
        num_questions: Optional limit on number of questions to process
        resume: Whether to resume from previous state
        save_reports: Whether to save individual reports
        
    Returns:
        Competition tracker with results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize competition tracker
    competition = CompetitionTracker(name=competition_name)
    
    # Check for resume file
    resume_file = os.path.join(output_dir, f"{competition_name}_resume.json")
    completed_forecasts = set()
    
    if resume and os.path.exists(resume_file):
        with open(resume_file, "r") as f:
            resume_data = json.load(f)
            completed_forecasts = set(resume_data.get("completed_forecasts", []))
        logger.info(f"Resuming from previous run with {len(completed_forecasts)} completed forecasts")
    
    # Load questions
    questions = []
    for question_id in question_ids:
        try:
            question = MetaculusApi.get_question_by_post_id(int(question_id))
            questions.append(question)
        except Exception as e:
            logger.error(f"Error loading question {question_id}: {e}")
    
    logger.info(f"Loaded {len(questions)} questions")
    
    # Limit number of questions if specified
    if num_questions is not None and num_questions < len(questions):
        questions = questions[:num_questions]
        logger.info(f"Limited to {len(questions)} questions")
    
    # Create bot factory
    bot_factory = {
        "basic": lambda personality_name: BasicBot(personality_name=personality_name),
        "research": lambda personality_name: ResearchBot(personality_name=personality_name),
        "calibrated": lambda personality_name: CalibratedBot(personality_name=personality_name),
        "economist": lambda personality_name: EconomistBot(personality_name=personality_name),
        "bayesian": lambda personality_name: BayesianBot(personality_name=personality_name),
        "ensemble": lambda personality_name: EnsembleBot(personality_names=[personality_name])
    }
    
    # Set up progress tracking
    total_combinations = len(questions) * len(personality_names) * len(bot_types)
    completed = 0
    
    # Create a personality manager to load configurations
    personality_manager = PersonalityManager()
    
    # Preload personality configurations
    personality_configs = {}
    for personality_name in personality_names:
        try:
            personality_configs[personality_name] = personality_manager.load_personality(personality_name)
        except Exception as e:
            logger.error(f"Error loading personality {personality_name}: {e}")
    
    # Add entries to competition for each bot and personality
    for bot_type in bot_types:
        for personality_name in personality_names:
            if personality_name in personality_configs:
                competition.add_entry(
                    bot_id=bot_type,
                    personality_name=personality_name,
                    personality_config=personality_configs.get(personality_name)
                )
    
    # Process each question
    for question in questions:
        question_id = str(question.id)
        
        # Skip if already completed in previous run
        if question_id in completed_forecasts:
            logger.info(f"Skipping already completed question {question_id}")
            continue
        
        logger.info(f"Processing question {question_id}: {question.question_text}")
        
        # Process each bot type and personality
        for bot_type in bot_types:
            if bot_type not in bot_factory:
                logger.warning(f"Unknown bot type: {bot_type}")
                continue
                
            for personality_name in personality_names:
                forecast_id = f"{bot_type}_{personality_name}_{question_id}"
                
                # Skip if already completed in this run
                if forecast_id in completed_forecasts:
                    continue
                
                try:
                    # Create bot with personality
                    bot = bot_factory[bot_type](personality_name)
                    
                    # Run forecast
                    logger.info(f"Forecasting with {bot_type} bot and {personality_name} personality")
                    report = await bot.forecast_question(question)
                    
                    # Add report to competition
                    competition.add_report(bot_type, report, personality_name)
                    
                    # Save report if requested
                    if save_reports:
                        report_file = os.path.join(output_dir, f"report_{forecast_id}.json")
                        with open(report_file, "w") as f:
                            f.write(report.to_json())
                    
                    # Mark as completed
                    completed_forecasts.add(forecast_id)
                    completed += 1
                    
                    # Save progress
                    with open(resume_file, "w") as f:
                        json.dump({
                            "completed_forecasts": list(completed_forecasts)
                        }, f)
                        
                    # Log progress
                    logger.info(f"Progress: {completed}/{total_combinations} ({completed/total_combinations:.1%})")
                    
                except Exception as e:
                    logger.error(f"Error forecasting with {bot_type} bot and {personality_name} personality: {e}")
    
    # Calculate metrics
    competition.calculate_metrics()
    competition.calculate_metrics_by_domain()
    
    # Save competition results
    results_file = os.path.join(output_dir, f"{competition_name}_results.json")
    competition.save_results(results_file)
    
    # Create visualizations
    for metric in [
        CompetitionMetric.EXPECTED_BASELINE,
        CompetitionMetric.BASELINE_SCORE,
        CompetitionMetric.CALIBRATION
    ]:
        try:
            visualization = competition.create_performance_visualization(metric)
            viz_file = os.path.join(output_dir, f"{competition_name}_{metric.value}.png")
            
            # Extract the base64 image data and save
            if visualization.startswith("data:image/png;base64,"):
                img_data = visualization.split(",")[1]
                with open(viz_file, "wb") as f:
                    import base64
                    f.write(base64.b64decode(img_data))
        except Exception as e:
            logger.error(f"Error creating visualization for {metric.value}: {e}")
    
    return competition


def analyze_tournament_results(results_file: str, output_dir: str) -> None:
    """
    Analyze tournament results and generate reports.
    
    Args:
        results_file: Path to tournament results file
        output_dir: Directory to save analysis
    """
    # Load competition
    competition = CompetitionTracker()
    competition.load_results(results_file)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate reports
    
    # 1. Overall leaderboard
    leaderboard = competition.get_leaderboard(CompetitionMetric.EXPECTED_BASELINE)
    leaderboard_df = pd.DataFrame(leaderboard, columns=["Entry", "Score"])
    leaderboard_df.to_csv(os.path.join(output_dir, "leaderboard.csv"), index=False)
    
    # 2. Personality effectiveness by domain
    domains = list(competition.metrics_by_domain.keys())
    
    domain_effectiveness = {}
    for domain in domains:
        domain_leaderboard = competition.get_domain_leaderboard(domain, CompetitionMetric.EXPECTED_BASELINE)
        
        personality_scores = {}
        for entry_id, score in domain_leaderboard:
            # Extract personality from entry_id (format: bot_type_personality)
            if "_" in entry_id:
                personality = entry_id.split("_", 1)[1]
                if personality not in personality_scores:
                    personality_scores[personality] = []
                personality_scores[personality].append(score)
        
        # Calculate average score for each personality
        domain_effectiveness[domain] = {
            personality: sum(scores) / len(scores)
            for personality, scores in personality_scores.items()
            if scores
        }
    
    # Convert to DataFrame
    effectiveness_data = []
    for domain, scores in domain_effectiveness.items():
        for personality, score in scores.items():
            effectiveness_data.append({
                "Domain": domain,
                "Personality": personality,
                "Score": score
            })
    
    effectiveness_df = pd.DataFrame(effectiveness_data)
    effectiveness_df.to_csv(os.path.join(output_dir, "personality_by_domain.csv"), index=False)
    
    # Create heatmap visualization
    if effectiveness_data:
        plt.figure(figsize=(12, 8))
        
        # Pivot data for heatmap
        pivot_df = effectiveness_df.pivot(index="Personality", columns="Domain", values="Score")
        
        # Create heatmap
        sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title("Personality Effectiveness by Domain")
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, "personality_domain_heatmap.png"), dpi=100)
        plt.close()
    
    # 3. Bot type effectiveness by personality
    bot_personality_data = []
    
    for entry_id in competition.entries:
        if "_" in entry_id:
            bot_type, personality = entry_id.split("_", 1)
            
            entry = competition.entries[entry_id]
            if CompetitionMetric.EXPECTED_BASELINE in entry.metrics:
                bot_personality_data.append({
                    "Bot Type": bot_type,
                    "Personality": personality,
                    "Score": entry.metrics[CompetitionMetric.EXPECTED_BASELINE]
                })
    
    bot_personality_df = pd.DataFrame(bot_personality_data)
    bot_personality_df.to_csv(os.path.join(output_dir, "bot_by_personality.csv"), index=False)
    
    # Create heatmap visualization
    if bot_personality_data:
        plt.figure(figsize=(12, 8))
        
        # Pivot data for heatmap
        pivot_df = bot_personality_df.pivot(index="Personality", columns="Bot Type", values="Score")
        
        # Create heatmap
        sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title("Bot Type Effectiveness by Personality")
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, "bot_personality_heatmap.png"), dpi=100)
        plt.close()
    
    # 4. Personality performance summary
    personality_comparison = competition.compare_personalities(CompetitionMetric.EXPECTED_BASELINE)
    
    summary_data = []
    for personality, stats in personality_comparison.items():
        if "mean" in stats:
            summary_data.append({
                "Personality": personality,
                "Mean Score": stats.get("mean", 0),
                "Median Score": stats.get("median", 0),
                "Min Score": stats.get("min", 0),
                "Max Score": stats.get("max", 0),
                "Std Dev": stats.get("std_dev", 0),
                "Count": stats.get("count", 0)
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values("Mean Score", ascending=False)
    summary_df.to_csv(os.path.join(output_dir, "personality_summary.csv"), index=False)
    
    # Create summary visualization
    if summary_data:
        plt.figure(figsize=(12, 6))
        
        bars = plt.bar(
            summary_df["Personality"],
            summary_df["Mean Score"],
            yerr=summary_df["Std Dev"],
            capsize=5
        )
        
        # Add value labels
        for bar, score in zip(bars, summary_df["Mean Score"]):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=10
            )
        
        plt.title("Personality Performance Summary")
        plt.xlabel("Personality")
        plt.ylabel("Mean Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, "personality_summary.png"), dpi=100)
        plt.close()


def main():
    """Run the personality tournament."""
    parser = argparse.ArgumentParser(description="Run a personality tournament")
    parser.add_argument("--questions", type=str, help="Comma-separated list of question IDs")
    parser.add_argument("--question-file", type=str, help="File containing question IDs (one per line)")
    parser.add_argument("--personalities", type=str, default="balanced,bayesian,economist,creative,cautious", 
                      help="Comma-separated list of personality names")
    parser.add_argument("--bot-types", type=str, default="basic,research,calibrated", 
                      help="Comma-separated list of bot types")
    parser.add_argument("--output-dir", type=str, default="tournament_results", 
                      help="Directory to save results")
    parser.add_argument("--name", type=str, default="personality_tournament",
                      help="Name of the tournament")
    parser.add_argument("--num-questions", type=int, help="Limit number of questions")
    parser.add_argument("--resume", action="store_true", help="Resume from previous state")
    parser.add_argument("--analyze-only", type=str, help="Only analyze results from the specified file")
    parser.add_argument("--metaculus-cup", action="store_true", 
                      help="Use current Metaculus Cup questions")
    parser.add_argument("--ai-competition", action="store_true",
                      help="Use current AI Competition questions")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{args.name}.log")
        ]
    )
    
    # Only analyze results if specified
    if args.analyze_only:
        analyze_tournament_results(args.analyze_only, args.output_dir)
        return
    
    # Get question IDs
    question_ids = []
    
    if args.questions:
        question_ids = args.questions.split(",")
    elif args.question_file:
        with open(args.question_file, "r") as f:
            question_ids = [line.strip() for line in f if line.strip()]
    elif args.metaculus_cup:
        # Get current Metaculus Cup questions
        tournament_id = MetaculusApi.CURRENT_METACULUS_CUP_ID
        questions = MetaculusApi.get_questions_in_tournament(tournament_id)
        question_ids = [str(q.id) for q in questions]
    elif args.ai_competition:
        # Get current AI Competition questions
        tournament_id = MetaculusApi.CURRENT_AI_COMPETITION_ID
        questions = MetaculusApi.get_questions_in_tournament(tournament_id)
        question_ids = [str(q.id) for q in questions]
    else:
        logger.error("No questions specified")
        parser.print_help()
        return
    
    # Get personality names
    personality_names = args.personalities.split(",")
    
    # Get bot types
    bot_types = args.bot_types.split(",")
    
    # Run tournament
    loop = asyncio.get_event_loop()
    competition = loop.run_until_complete(
        run_tournament(
            question_ids=question_ids,
            personality_names=personality_names,
            bot_types=bot_types,
            output_dir=args.output_dir,
            competition_name=args.name,
            num_questions=args.num_questions,
            resume=args.resume
        )
    )
    
    # Analyze results
    results_file = os.path.join(args.output_dir, f"{args.name}_results.json")
    analyze_tournament_results(results_file, args.output_dir)


if __name__ == "__main__":
    main() 