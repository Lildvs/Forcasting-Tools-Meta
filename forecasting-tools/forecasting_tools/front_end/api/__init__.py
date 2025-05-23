"""
Forecasting Tools API Initialization

This module initializes the FastAPI application and registers all routes.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from forecasting_tools.front_end.api.routes import router as personality_router

# Create FastAPI app
app = FastAPI(
    title="Forecasting Tools API",
    description="API for the Forecasting Tools application",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(personality_router)

# Add a simple root endpoint
@app.get("/")
async def root():
    """Root endpoint to verify API is running."""
    return {
        "message": "Forecasting Tools API is running",
        "version": "1.0.0",
        "docs_url": "/docs",
    } 