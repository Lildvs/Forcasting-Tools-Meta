"""
Database Models

This module provides database models for structured storage of forecasts,
research data, and user interactions. Models use SQLAlchemy ORM for 
database interaction.
"""

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, 
    ForeignKey, Text, JSON, Index, Table, Enum, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.sql import func
from typing import Dict, List, Optional, Any, Union
from enum import Enum as PyEnum
import datetime
import logging
import json
import uuid
from contextlib import contextmanager

# Create base model class
Base = declarative_base()

# Define common enums
class ForecastType(str, PyEnum):
    BINARY = "binary"
    NUMERIC = "numeric"
    MULTIPLE_CHOICE = "multiple_choice"

class UserRole(str, PyEnum):
    USER = "user"
    ADMIN = "admin"
    ANALYST = "analyst"

# Define association tables
forecast_tag_association = Table(
    "forecast_tag_association",
    Base.metadata,
    Column("forecast_id", String(36), ForeignKey("forecasts.id")),
    Column("tag_id", String(36), ForeignKey("tags.id")),
)

research_tag_association = Table(
    "research_tag_association",
    Base.metadata,
    Column("research_id", String(36), ForeignKey("research_data.id")),
    Column("tag_id", String(36), ForeignKey("tags.id")),
)

# Define model classes
class User(Base):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default=UserRole.USER.value)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationships
    forecasts = relationship("Forecast", back_populates="user")
    interactions = relationship("UserInteraction", back_populates="user")
    
    # Indexes
    __table_args__ = (
        Index("idx_users_username", "username"),
        Index("idx_users_email", "email"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_active": self.is_active,
        }


class Forecast(Base):
    """Forecast model for storing forecast data."""
    
    __tablename__ = "forecasts"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    question_text = Column(Text, nullable=False)
    forecast_type = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Binary forecast fields
    probability = Column(Float, nullable=True)
    community_prediction = Column(Float, nullable=True)
    
    # Numeric forecast fields
    mean = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    unit = Column(String(50), nullable=True)
    
    # Multiple choice forecast fields
    options = Column(JSON, nullable=True)
    probabilities = Column(JSON, nullable=True)
    
    # Shared fields
    reasoning = Column(Text, nullable=True)
    confidence_level = Column(String(50), nullable=True)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="forecasts")
    research_sources = relationship("ResearchData", back_populates="forecast")
    history = relationship("ForecastHistory", back_populates="forecast")
    tags = relationship("Tag", secondary=forecast_tag_association, back_populates="forecasts")
    
    # Indexes
    __table_args__ = (
        Index("idx_forecasts_user_id", "user_id"),
        Index("idx_forecasts_created_at", "created_at"),
        Index("idx_forecasts_forecast_type", "forecast_type"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "question_text": self.question_text,
            "forecast_type": self.forecast_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "probability": self.probability,
            "community_prediction": self.community_prediction,
            "mean": self.mean,
            "low": self.low,
            "high": self.high,
            "unit": self.unit,
            "options": self.options,
            "probabilities": self.probabilities,
            "reasoning": self.reasoning,
            "confidence_level": self.confidence_level,
            "metadata": self.metadata,
            "tags": [tag.name for tag in self.tags],
        }


class ForecastHistory(Base):
    """History of changes to a forecast."""
    
    __tablename__ = "forecast_history"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    forecast_id = Column(String(36), ForeignKey("forecasts.id"), nullable=False)
    timestamp = Column(DateTime, default=func.now())
    change_type = Column(String(50), nullable=False)
    previous_value = Column(JSON, nullable=True)
    new_value = Column(JSON, nullable=False)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    forecast = relationship("Forecast", back_populates="history")
    
    # Indexes
    __table_args__ = (
        Index("idx_forecast_history_forecast_id", "forecast_id"),
        Index("idx_forecast_history_timestamp", "timestamp"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "forecast_id": self.forecast_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "change_type": self.change_type,
            "previous_value": self.previous_value,
            "new_value": self.new_value,
            "user_id": self.user_id,
            "metadata": self.metadata,
        }


class ResearchData(Base):
    """Research data used for forecasts."""
    
    __tablename__ = "research_data"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    forecast_id = Column(String(36), ForeignKey("forecasts.id"), nullable=True)
    source = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)
    relevance_score = Column(Float, nullable=True)
    reliability_score = Column(Float, nullable=True)
    evidence_type = Column(String(50), nullable=True)
    impact_direction = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=func.now())
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    forecast = relationship("Forecast", back_populates="research_sources")
    tags = relationship("Tag", secondary=research_tag_association, back_populates="research_sources")
    
    # Indexes
    __table_args__ = (
        Index("idx_research_data_forecast_id", "forecast_id"),
        Index("idx_research_data_evidence_type", "evidence_type"),
        Index("idx_research_data_created_at", "created_at"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "forecast_id": self.forecast_id,
            "source": self.source,
            "content": self.content,
            "relevance_score": self.relevance_score,
            "reliability_score": self.reliability_score,
            "evidence_type": self.evidence_type,
            "impact_direction": self.impact_direction,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.metadata,
            "tags": [tag.name for tag in self.tags],
        }


class Tag(Base):
    """Tags for categorizing forecasts and research data."""
    
    __tablename__ = "tags"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    forecasts = relationship("Forecast", secondary=forecast_tag_association, back_populates="tags")
    research_sources = relationship("ResearchData", secondary=research_tag_association, back_populates="tags")
    
    # Indexes
    __table_args__ = (
        Index("idx_tags_name", "name"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class UserInteraction(Base):
    """Track user interactions with the system."""
    
    __tablename__ = "user_interactions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    interaction_type = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=func.now())
    data = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(255), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="interactions")
    
    # Indexes
    __table_args__ = (
        Index("idx_user_interactions_user_id", "user_id"),
        Index("idx_user_interactions_timestamp", "timestamp"),
        Index("idx_user_interactions_type", "interaction_type"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "interaction_type": self.interaction_type,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "data": self.data,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
        }


class JobQueue(Base):
    """Queue for background jobs."""
    
    __tablename__ = "job_queue"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_type = Column(String(50), nullable=False)
    status = Column(String(50), default="pending")
    priority = Column(Integer, default=0)
    data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error = Column(Text, nullable=True)
    retries = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    next_retry_at = Column(DateTime, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index("idx_job_queue_status", "status"),
        Index("idx_job_queue_priority", "priority"),
        Index("idx_job_queue_created_at", "created_at"),
        Index("idx_job_queue_next_retry", "next_retry_at"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "job_type": self.job_type,
            "status": self.status,
            "priority": self.priority,
            "data": self.data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "next_retry_at": self.next_retry_at.isoformat() if self.next_retry_at else None,
        }


class DatabaseManager:
    """
    Manage database connections and session handling.
    
    Provides connection pooling, session management, and transaction support.
    """
    
    def __init__(
        self,
        connection_string: str,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        echo: bool = False,
    ):
        self.connection_string = connection_string
        self.engine = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            echo=echo
        )
        self.Session = scoped_session(sessionmaker(bind=self.engine))
    
    def create_tables(self):
        """Create all defined tables in the database."""
        Base.metadata.create_all(self.engine)
    
    def drop_tables(self):
        """Drop all defined tables from the database."""
        Base.metadata.drop_all(self.engine)
    
    @contextmanager
    def session_scope(self):
        """
        Provide a transactional scope around a series of operations.
        
        Usage:
            with database_manager.session_scope() as session:
                user = User(username="example", email="example@example.com")
                session.add(user)
        """
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logging.error(f"Database error: {str(e)}")
            raise
        finally:
            session.close()
    
    def close(self):
        """Close all connections in the pool."""
        self.Session.remove()
        self.engine.dispose() 