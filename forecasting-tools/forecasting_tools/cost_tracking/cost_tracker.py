"""
Cost tracking and calculation utilities.

This module provides functionality to track and calculate costs for forecasts
based on token usage and model pricing.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import json
import os
import sqlite3
import threading
from pathlib import Path
import uuid
import logging

logger = logging.getLogger(__name__)

@dataclass
class ForecastCost:
    """Represents the cost information for a single forecast."""
    
    question_id: str
    question_text: str
    timestamp: datetime
    tokens_used: int
    cost_usd: float
    personality_name: Optional[str] = None
    model_name: str = "default"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "timestamp": self.timestamp.isoformat(),
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "personality_name": self.personality_name,
            "model_name": self.model_name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ForecastCost':
        """Create a ForecastCost instance from a dictionary."""
        return cls(
            question_id=data["question_id"],
            question_text=data["question_text"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            tokens_used=data["tokens_used"],
            cost_usd=data["cost_usd"],
            personality_name=data.get("personality_name"),
            model_name=data.get("model_name", "default")
        )

class CostTracker:
    """Tracks and calculates costs for forecasts based on token usage."""
    
    # Thread lock for database operations
    _lock = threading.RLock()
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize cost tracker with optional custom database path.
        
        Args:
            db_path: Custom path for the SQLite database. If None, uses the default
                     path in the user's home directory.
        """
        if db_path is None:
            # Use user data directory
            data_dir = Path(os.path.expanduser("~/.forecasting-tools/data"))
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "forecast_costs.db")
            
        self.db_path = db_path
        self._initialize_db()
        
        # Model pricing information ($ per 1K tokens)
        self.model_rates = {
            # OpenAI models
            "gpt-4": 0.03,        # $0.03 per 1K tokens
            "gpt-4-0613": 0.03,   # $0.03 per 1K tokens
            "gpt-4-32k": 0.06,    # $0.06 per 1K tokens
            "gpt-3.5-turbo": 0.002,  # $0.002 per 1K tokens
            "gpt-3.5-turbo-16k": 0.004,  # $0.004 per 1K tokens
            
            # Anthropic models
            "claude-3-opus": 0.015,  # $0.015 per 1K tokens
            "claude-3-sonnet": 0.008,  # $0.008 per 1K tokens
            "claude-3-haiku": 0.003,  # $0.003 per 1K tokens
            
            # Default fallback rate
            "default": 0.01  # $0.01 per 1K tokens
        }
    
    def _initialize_db(self) -> None:
        """Create database tables if they don't exist."""
        with self._lock:
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Create forecast costs table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS forecast_costs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_id TEXT NOT NULL,
                    question_text TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    tokens_used INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    personality_name TEXT,
                    model_name TEXT NOT NULL
                )
                ''')
                
                # Create indices for faster queries
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON forecast_costs(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_model ON forecast_costs(model_name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_question_id ON forecast_costs(question_id)')
                
                conn.commit()
                logger.debug("Initialized cost tracking database at %s", self.db_path)
            except sqlite3.Error as e:
                logger.error("Database initialization error: %s", str(e))
                raise
            finally:
                if conn:
                    conn.close()
    
    def calculate_cost(self, tokens_used: int, model_name: str = "default") -> float:
        """
        Calculate cost in USD based on tokens used and model.
        
        Args:
            tokens_used: Number of tokens used in the forecast
            model_name: Name of the LLM model used
            
        Returns:
            Cost in USD
        """
        # Get rate for model ($/1K tokens), defaulting to generic if model not found
        rate = self.model_rates.get(model_name.lower(), self.model_rates["default"])
        return (tokens_used / 1000) * rate
    
    def track_forecast(self, 
                      question_id: Optional[str],
                      question_text: str, 
                      tokens_used: int,
                      model_name: str,
                      personality_name: Optional[str] = None) -> ForecastCost:
        """
        Track a forecast cost and return the cost details.
        
        Args:
            question_id: Unique identifier for the question
            question_text: Text of the forecast question
            tokens_used: Number of tokens used in the forecast
            model_name: Name of the LLM model used
            personality_name: Optional name of the personality used
            
        Returns:
            ForecastCost object with the cost details
        """
        # Generate question_id if not provided
        if question_id is None:
            question_id = str(uuid.uuid4())
            
        cost_usd = self.calculate_cost(tokens_used, model_name)
        timestamp = datetime.now()
        
        # Create cost record
        cost_record = ForecastCost(
            question_id=question_id,
            question_text=question_text,
            timestamp=timestamp,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            personality_name=personality_name,
            model_name=model_name
        )
        
        # Save to database
        self._save_cost(cost_record)
        
        logger.info(
            "Tracked forecast cost: $%.4f (%d tokens, model: %s, personality: %s)",
            cost_usd, tokens_used, model_name, personality_name or "None"
        )
        
        return cost_record
    
    def _save_cost(self, cost: ForecastCost) -> None:
        """
        Save cost record to database.
        
        Args:
            cost: ForecastCost object to save
        """
        with self._lock:
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO forecast_costs 
                (question_id, question_text, timestamp, tokens_used, cost_usd, personality_name, model_name) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    cost.question_id,
                    cost.question_text,
                    cost.timestamp.isoformat(),
                    cost.tokens_used,
                    cost.cost_usd,
                    cost.personality_name,
                    cost.model_name
                ))
                
                conn.commit()
                logger.debug("Saved cost record to database: %s", cost.question_id)
            except sqlite3.Error as e:
                logger.error("Error saving cost record: %s", str(e))
                if conn:
                    conn.rollback()
                raise
            finally:
                if conn:
                    conn.close()
    
    def get_cost_history(self, limit: int = 100, offset: int = 0) -> List[ForecastCost]:
        """
        Retrieve cost history, most recent first.
        
        Args:
            limit: Maximum number of records to retrieve
            offset: Starting offset for pagination
            
        Returns:
            List of ForecastCost objects
        """
        with self._lock:
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT question_id, question_text, timestamp, tokens_used, cost_usd, personality_name, model_name
                FROM forecast_costs
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                ''', (limit, offset))
                
                rows = cursor.fetchall()
                costs = []
                
                for row in rows:
                    costs.append(ForecastCost(
                        question_id=row[0],
                        question_text=row[1],
                        timestamp=datetime.fromisoformat(row[2]),
                        tokens_used=row[3],
                        cost_usd=row[4],
                        personality_name=row[5],
                        model_name=row[6]
                    ))
                
                return costs
            except sqlite3.Error as e:
                logger.error("Error retrieving cost history: %s", str(e))
                return []
            finally:
                if conn:
                    conn.close()
    
    def get_total_cost(self) -> float:
        """
        Get the total cost of all forecasts.
        
        Returns:
            Total cost in USD
        """
        with self._lock:
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('SELECT SUM(cost_usd) FROM forecast_costs')
                total = cursor.fetchone()[0]
                
                return total or 0.0
            except sqlite3.Error as e:
                logger.error("Error retrieving total cost: %s", str(e))
                return 0.0
            finally:
                if conn:
                    conn.close()
    
    def get_cost_by_date_range(self, start_date: datetime, end_date: datetime) -> float:
        """
        Get total cost within a date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            Total cost in USD for the specified date range
        """
        with self._lock:
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT SUM(cost_usd) FROM forecast_costs
                WHERE timestamp BETWEEN ? AND ?
                ''', (start_date.isoformat(), end_date.isoformat()))
                
                total = cursor.fetchone()[0]
                
                return total or 0.0
            except sqlite3.Error as e:
                logger.error("Error retrieving cost by date range: %s", str(e))
                return 0.0
            finally:
                if conn:
                    conn.close()
    
    def get_cost_by_model(self) -> Dict[str, float]:
        """
        Get total costs grouped by model.
        
        Returns:
            Dictionary mapping model names to total costs
        """
        with self._lock:
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT model_name, SUM(cost_usd) FROM forecast_costs
                GROUP BY model_name
                ''')
                
                model_costs = {}
                for row in cursor.fetchall():
                    model_costs[row[0]] = row[1]
                
                return model_costs
            except sqlite3.Error as e:
                logger.error("Error retrieving costs by model: %s", str(e))
                return {}
            finally:
                if conn:
                    conn.close()
    
    def get_cost_by_personality(self) -> Dict[str, float]:
        """
        Get total costs grouped by personality.
        
        Returns:
            Dictionary mapping personality names to total costs
        """
        with self._lock:
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT 
                    COALESCE(personality_name, 'None') as personality, 
                    SUM(cost_usd) 
                FROM forecast_costs
                GROUP BY personality
                ''')
                
                personality_costs = {}
                for row in cursor.fetchall():
                    personality_costs[row[0]] = row[1]
                
                return personality_costs
            except sqlite3.Error as e:
                logger.error("Error retrieving costs by personality: %s", str(e))
                return {}
            finally:
                if conn:
                    conn.close()
    
    def get_daily_costs(self, days: int = 30) -> Dict[str, float]:
        """
        Get daily costs for the specified number of days.
        
        Args:
            days: Number of days to retrieve, starting from today
            
        Returns:
            Dictionary mapping date strings (YYYY-MM-DD) to total costs
        """
        with self._lock:
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                
                # Enable datetime functions
                conn.create_function("DATE", 1, lambda x: x.split('T')[0])
                
                cursor = conn.cursor()
                
                # Calculate date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days - 1)  # -1 to include today
                
                cursor.execute('''
                SELECT DATE(timestamp) as date, SUM(cost_usd) as total_cost
                FROM forecast_costs
                WHERE timestamp BETWEEN ? AND ?
                GROUP BY date
                ORDER BY date
                ''', (start_date.isoformat(), end_date.isoformat()))
                
                daily_costs = {}
                for row in cursor.fetchall():
                    daily_costs[row[0]] = row[1]
                
                # Fill in missing days with zero
                current_date = start_date
                while current_date <= end_date:
                    date_str = current_date.strftime('%Y-%m-%d')
                    if date_str not in daily_costs:
                        daily_costs[date_str] = 0.0
                    current_date += timedelta(days=1)
                
                return dict(sorted(daily_costs.items()))
            except sqlite3.Error as e:
                logger.error("Error retrieving daily costs: %s", str(e))
                return {}
            finally:
                if conn:
                    conn.close()
    
    def get_cost_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cost statistics.
        
        Returns:
            Dictionary with cost statistics (total, average, max, etc.)
        """
        with self._lock:
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Get overall statistics
                cursor.execute('''
                SELECT 
                    COUNT(*) as count,
                    SUM(cost_usd) as total_cost,
                    AVG(cost_usd) as avg_cost,
                    MAX(cost_usd) as max_cost,
                    SUM(tokens_used) as total_tokens,
                    AVG(tokens_used) as avg_tokens
                FROM forecast_costs
                ''')
                
                row = cursor.fetchone()
                
                # Get recent statistics (last 7 days)
                seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
                cursor.execute('''
                SELECT 
                    COUNT(*) as count,
                    SUM(cost_usd) as total_cost
                FROM forecast_costs
                WHERE timestamp >= ?
                ''', (seven_days_ago,))
                
                recent_row = cursor.fetchone()
                
                return {
                    "total_forecasts": row[0],
                    "total_cost": row[1] or 0.0,
                    "average_cost": row[2] or 0.0,
                    "max_cost": row[3] or 0.0,
                    "total_tokens": row[4] or 0,
                    "average_tokens": row[5] or 0,
                    "recent_forecasts": recent_row[0],
                    "recent_cost": recent_row[1] or 0.0
                }
            except sqlite3.Error as e:
                logger.error("Error retrieving cost statistics: %s", str(e))
                return {
                    "total_forecasts": 0,
                    "total_cost": 0.0,
                    "average_cost": 0.0,
                    "max_cost": 0.0,
                    "total_tokens": 0,
                    "average_tokens": 0,
                    "recent_forecasts": 0,
                    "recent_cost": 0.0
                }
            finally:
                if conn:
                    conn.close()
    
    def clear_history(self) -> bool:
        """
        Clear all cost history.
        
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM forecast_costs')
                conn.commit()
                logger.info("Cleared cost history")
                return True
            except sqlite3.Error as e:
                logger.error("Error clearing cost history: %s", str(e))
                if conn:
                    conn.rollback()
                return False
            finally:
                if conn:
                    conn.close()
    
    def export_to_csv(self, file_path: str) -> bool:
        """
        Export cost history to CSV file.
        
        Args:
            file_path: Path to save the CSV file
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            conn = None
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT 
                    question_id, question_text, timestamp, tokens_used, 
                    cost_usd, personality_name, model_name
                FROM forecast_costs
                ORDER BY timestamp DESC
                ''')
                
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    f.write("question_id,question_text,timestamp,tokens_used,cost_usd,personality_name,model_name\n")
                    for row in cursor:
                        # Basic CSV escaping for the question text
                        question_text = row[1].replace('"', '""')
                        f.write(f'"{row[0]}","{question_text}",{row[2]},{row[3]},{row[4]},"{row[5] or "None"}","{row[6]}"\n')
                
                logger.info("Exported cost history to %s", file_path)
                return True
            except (sqlite3.Error, IOError) as e:
                logger.error("Error exporting cost history: %s", str(e))
                return False
            finally:
                if conn:
                    conn.close() 