"""
API Routes for Forecasting Tools

This module defines the API endpoints for the forecasting tools application,
with a focus on personality management.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse

from forecasting_tools.personality_management import PersonalityManager
from forecasting_tools.personality_management.config import PersonalityConfig
from forecasting_tools.personality_management.diversity import PersonalityDiversityScorer
from forecasting_tools.forecast_helpers.competition import CompetitionTracker, CompetitionMetric

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Define models for API requests and responses
class PersonalityBasic(BaseModel):
    name: str
    description: Optional[str] = None
    thinking_style: str
    uncertainty_approach: str
    reasoning_depth: str

class PersonalityDetail(PersonalityBasic):
    traits: Dict[str, Any] = Field(default_factory=dict)
    template_variables: Dict[str, Any] = Field(default_factory=dict)
    temperature: Optional[float] = None
    expert_persona: Optional[str] = None

class PersonalityPerformance(BaseModel):
    name: str
    accuracy: float
    calibration: float
    information_score: float
    expected_score: float
    domains: Dict[str, float] = Field(default_factory=dict)

class DiversityAnalysis(BaseModel):
    overall_diversity: float
    trait_diversity: Dict[str, float] = Field(default_factory=dict)
    coverage_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    uniqueness_scores: Dict[str, float] = Field(default_factory=dict)

class DomainRecommendation(BaseModel):
    domain: str
    recommended_personalities: List[str]
    reasoning: str

class ForecastRequest(BaseModel):
    question_text: str
    question_type: str = "binary"
    background_info: Optional[str] = None
    resolution_criteria: Optional[str] = None
    fine_print: Optional[str] = None
    personality_name: Optional[str] = "balanced"
    domain: Optional[str] = None

class ForecastResponse(BaseModel):
    question_text: str
    personality_used: str
    prediction: Any
    reasoning: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Endpoint implementations
@router.get("/api/personalities", response_model=List[PersonalityBasic])
async def list_personalities():
    """List all available personalities with basic information."""
    try:
        personality_manager = PersonalityManager()
        personalities = []
        
        for name in personality_manager.list_available_personalities():
            try:
                config = personality_manager.load_personality(name)
                personalities.append(
                    PersonalityBasic(
                        name=name,
                        description=config.description,
                        thinking_style=config.thinking_style.value,
                        uncertainty_approach=config.uncertainty_approach.value,
                        reasoning_depth=config.reasoning_depth.value
                    )
                )
            except Exception as e:
                logger.error(f"Error loading personality {name}: {str(e)}")
        
        return personalities
    except Exception as e:
        logger.error(f"Error listing personalities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing personalities: {str(e)}")

@router.get("/api/personalities/{name}", response_model=PersonalityDetail)
async def get_personality(name: str):
    """Get detailed information about a specific personality."""
    try:
        personality_manager = PersonalityManager()
        config = personality_manager.load_personality(name)
        
        # Convert traits to a simple dictionary
        traits_dict = {}
        for trait_name, trait in config.traits.items():
            traits_dict[trait_name] = trait.value
        
        return PersonalityDetail(
            name=name,
            description=config.description,
            thinking_style=config.thinking_style.value,
            uncertainty_approach=config.uncertainty_approach.value,
            reasoning_depth=config.reasoning_depth.value,
            traits=traits_dict,
            template_variables=config.template_variables,
            temperature=getattr(config, "temperature", None),
            expert_persona=getattr(config, "expert_persona", None)
        )
    except Exception as e:
        logger.error(f"Error getting personality {name}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Personality not found: {str(e)}")

@router.get("/api/personalities/performance", response_model=List[PersonalityPerformance])
async def get_personality_performance():
    """Get performance metrics for all personalities."""
    try:
        # In a real implementation, this would use the CompetitionTracker to get actual data
        # For now, use mock data for demonstration
        personality_manager = PersonalityManager()
        performance_data = []
        
        for name in personality_manager.list_available_personalities():
            # Create mock performance data based on personality name
            if "bayesian" in name.lower():
                perf = PersonalityPerformance(
                    name=name,
                    accuracy=0.78,
                    calibration=0.85,
                    information_score=0.65,
                    expected_score=0.76,
                    domains={
                        "science": 0.85,
                        "economics": 0.82,
                        "politics": 0.75,
                        "health": 0.73
                    }
                )
            elif "economist" in name.lower():
                perf = PersonalityPerformance(
                    name=name,
                    accuracy=0.75,
                    calibration=0.80,
                    information_score=0.70,
                    expected_score=0.75,
                    domains={
                        "economics": 0.90,
                        "finance": 0.88,
                        "politics": 0.72,
                        "technology": 0.65
                    }
                )
            elif "creative" in name.lower():
                perf = PersonalityPerformance(
                    name=name,
                    accuracy=0.70,
                    calibration=0.72,
                    information_score=0.80,
                    expected_score=0.74,
                    domains={
                        "technology": 0.82,
                        "entertainment": 0.85,
                        "science": 0.75,
                        "politics": 0.68
                    }
                )
            elif "cautious" in name.lower():
                perf = PersonalityPerformance(
                    name=name,
                    accuracy=0.76,
                    calibration=0.88,
                    information_score=0.60,
                    expected_score=0.75,
                    domains={
                        "health": 0.85,
                        "environment": 0.83,
                        "economics": 0.75,
                        "technology": 0.68
                    }
                )
            else:
                perf = PersonalityPerformance(
                    name=name,
                    accuracy=0.74,
                    calibration=0.78,
                    information_score=0.70,
                    expected_score=0.74,
                    domains={
                        "science": 0.75,
                        "economics": 0.75,
                        "politics": 0.75,
                        "health": 0.75
                    }
                )
            
            performance_data.append(perf)
        
        return performance_data
    except Exception as e:
        logger.error(f"Error getting personality performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting personality performance: {str(e)}")

@router.post("/api/personalities/analyze-diversity", response_model=DiversityAnalysis)
async def analyze_diversity(personality_names: List[str]):
    """Analyze diversity metrics for a set of personalities."""
    try:
        if not personality_names:
            raise HTTPException(status_code=400, detail="No personalities provided")
        
        personality_manager = PersonalityManager()
        personality_configs = {}
        
        # Load personality configurations
        for name in personality_names:
            try:
                personality_configs[name] = personality_manager.load_personality(name)
            except Exception as e:
                logger.warning(f"Could not load personality {name}: {str(e)}")
        
        if not personality_configs:
            raise HTTPException(status_code=404, detail="No valid personalities found")
        
        # Calculate diversity metrics
        diversity_scorer = PersonalityDiversityScorer()
        diversity_metrics = diversity_scorer.calculate_ensemble_diversity(personality_configs)
        
        return diversity_metrics
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing diversity: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing diversity: {str(e)}")

@router.get("/api/domains/{domain}/recommend", response_model=DomainRecommendation)
async def recommend_for_domain(domain: str, count: int = Query(3, ge=1, le=5)):
    """Get personality recommendations for a specific domain."""
    try:
        personality_manager = PersonalityManager()
        available_personalities = personality_manager.list_available_personalities()
        
        # In a real implementation, this would use performance data for recommendations
        # For this demonstration, use a simple mapping
        domain_mapping = {
            "economics": ["economist", "bayesian", "balanced"],
            "finance": ["economist", "bayesian", "cautious"],
            "politics": ["bayesian", "creative", "balanced"],
            "technology": ["creative", "bayesian", "balanced"],
            "science": ["bayesian", "balanced", "cautious"],
            "health": ["cautious", "balanced", "bayesian"],
            "sports": ["bayesian", "creative", "balanced"],
            "entertainment": ["creative", "balanced", "bayesian"],
            "geopolitics": ["bayesian", "cautious", "economist"],
            "environment": ["cautious", "balanced", "bayesian"],
            "energy": ["economist", "bayesian", "balanced"],
            "social": ["creative", "balanced", "cautious"],
        }
        
        # Get recommended personalities for this domain
        domain_lower = domain.lower()
        if domain_lower in domain_mapping:
            recommended = [p for p in domain_mapping[domain_lower] if p in available_personalities]
        else:
            # Default to balanced for unknown domains
            recommended = ["balanced"]
            if "bayesian" in available_personalities:
                recommended.append("bayesian")
            if "creative" in available_personalities:
                recommended.append("creative")
        
        # Limit to requested count
        recommended = recommended[:count]
        
        # Generate reasoning
        if domain_lower in domain_mapping:
            if domain_lower in ["economics", "finance"]:
                reasoning = f"For {domain} questions, analytical thinking styles like Economist are most effective, with Bayesian approaches providing good calibration."
            elif domain_lower in ["politics", "geopolitics"]:
                reasoning = f"For {domain} questions, Bayesian approaches handle uncertainty well, while Creative personalities can identify unexpected scenarios."
            elif domain_lower in ["technology", "entertainment"]:
                reasoning = f"For {domain} questions, Creative personalities excel at identifying novel possibilities, with Balanced approaches providing solid foundations."
            elif domain_lower in ["science", "health", "environment"]:
                reasoning = f"For {domain} questions, careful analysis is key, with Cautious personalities avoiding overconfidence and Bayesian approaches handling uncertainty well."
            else:
                reasoning = f"These personalities have demonstrated strong performance in {domain} forecasting."
        else:
            reasoning = f"For general {domain} questions, a combination of Balanced forecasting with specialized approaches provides the best results."
        
        return DomainRecommendation(
            domain=domain,
            recommended_personalities=recommended,
            reasoning=reasoning
        )
    except Exception as e:
        logger.error(f"Error recommending for domain {domain}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@router.post("/api/forecast", response_model=ForecastResponse)
async def create_forecast(request: ForecastRequest, background_tasks: BackgroundTasks):
    """Create a forecast using a specific personality."""
    try:
        # In a real implementation, this would create a forecast using the specified personality
        # For this demonstration, return a mock response
        
        # If domain is provided but no personality, get a recommendation
        personality_name = request.personality_name or "balanced"
        if request.domain and not request.personality_name:
            try:
                recommendation = await recommend_for_domain(request.domain)
                if recommendation.recommended_personalities:
                    personality_name = recommendation.recommended_personalities[0]
            except Exception:
                # Fall back to default if recommendation fails
                pass
        
        # Create a mock forecast response
        if request.question_type == "binary":
            if "will" in request.question_text.lower() or "won't" in request.question_text.lower():
                prediction = 0.65
            else:
                prediction = 0.5
                
            # Adjust prediction based on personality
            if personality_name == "cautious":
                # Move closer to 50%
                prediction = 0.5 + (prediction - 0.5) * 0.8
            elif personality_name == "bayesian":
                # Add some precision
                prediction = round(prediction * 100) / 100
        else:
            prediction = {"mean": 50, "stdev": 10}
        
        # Create reasoning based on personality
        if personality_name == "bayesian":
            reasoning = "Based on Bayesian analysis of the available evidence, I estimate a probability of {:.0%}. The key factors are...".format(prediction)
        elif personality_name == "economist":
            reasoning = "Economic analysis suggests a probability of {:.0%}. Market indicators and incentive structures point to...".format(prediction)
        elif personality_name == "creative":
            reasoning = "Considering multiple scenarios, I estimate a probability of {:.0%}. Some unconventional factors to consider include...".format(prediction)
        elif personality_name == "cautious":
            reasoning = "Taking a conservative approach, I estimate a probability of {:.0%}. The evidence suggests...".format(prediction)
        else:
            reasoning = "After analyzing the available evidence, I estimate a probability of {:.0%}. Key considerations include...".format(prediction)
        
        return ForecastResponse(
            question_text=request.question_text,
            personality_used=personality_name,
            prediction=prediction,
            reasoning=reasoning,
            metadata={
                "domain": request.domain,
                "question_type": request.question_type
            }
        )
    except Exception as e:
        logger.error(f"Error creating forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating forecast: {str(e)}")

@router.post("/api/personalities/custom", response_model=PersonalityDetail)
async def create_custom_personality(personality: PersonalityDetail):
    """Create a custom personality."""
    try:
        # In a real implementation, this would create a custom personality
        # For this demonstration, just return the input with a confirmation message
        
        # Add a message to indicate this is just a demonstration
        personality.description = f"{personality.description or ''} (Custom personality created via API)"
        
        return personality
    except Exception as e:
        logger.error(f"Error creating custom personality: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating custom personality: {str(e)}")

@router.get("/api/user/preferences", response_model=Dict[str, Any])
async def get_user_preferences():
    """Get user-specific personality preferences."""
    try:
        # In a real implementation, this would retrieve user preferences from a database
        # For this demonstration, return mock preferences
        return {
            "default_personality": "balanced",
            "favorite_personalities": ["bayesian", "economist", "creative"],
            "domain_preferences": {
                "economics": "economist",
                "politics": "bayesian",
                "technology": "creative",
                "general": "balanced"
            },
            "ensemble_preferences": {
                "enabled": True,
                "ensemble_size": 3,
                "diversity_weight": 0.7
            }
        }
    except Exception as e:
        logger.error(f"Error getting user preferences: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting user preferences: {str(e)}")

@router.post("/api/user/preferences", response_model=Dict[str, Any])
async def save_user_preferences(preferences: Dict[str, Any]):
    """Save user-specific personality preferences."""
    try:
        # In a real implementation, this would save user preferences to a database
        # For this demonstration, just return the input with a confirmation message
        preferences["saved"] = True
        return preferences
    except Exception as e:
        logger.error(f"Error saving user preferences: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving user preferences: {str(e)}") 