"""
Personality Telemetry System

This module provides telemetry tracking for personality usage and performance,
enabling data-driven optimization and anomaly detection.
"""

import os
import json
import time
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
import statistics

from forecasting_tools.personality_management.config import PersonalityConfig

logger = logging.getLogger(__name__)


class PersonalityTelemetryTracker:
    """
    Tracker for personality usage and performance metrics.
    
    This class provides:
    - Usage tracking for personalities
    - Performance metric collection
    - Anomaly detection
    - Historical data aggregation
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Implement thread-safe singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PersonalityTelemetryTracker, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the telemetry tracker.
        
        Args:
            db_path: Path to SQLite database file (optional)
        """
        # Avoid re-initialization
        if getattr(self, "_initialized", False):
            return
            
        if db_path is None:
            # Use default location
            data_dir = os.environ.get("FORECASTING_TOOLS_DATA", os.path.expanduser("~/.forecasting-tools"))
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "personality_telemetry.db")
        
        self.db_path = db_path
        self._setup_database()
        
        # Set up in-memory caches
        self._usage_cache: Dict[str, Dict[str, Any]] = {}
        self._performance_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._anomaly_thresholds: Dict[str, Dict[str, float]] = {}
        
        # Flag to track initialization
        self._initialized = True
        
        logger.debug(f"Personality telemetry tracker initialized with database at {db_path}")
    
    def _setup_database(self) -> None:
        """Set up the SQLite database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create usage table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS personality_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    personality_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    context TEXT,
                    duration_ms INTEGER,
                    template_name TEXT,
                    token_count INTEGER,
                    session_id TEXT
                )
                """)
                
                # Create performance metrics table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS personality_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    personality_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    domain TEXT,
                    question_id TEXT,
                    timestamp TEXT NOT NULL
                )
                """)
                
                # Create anomaly log table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS personality_anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    personality_name TEXT NOT NULL,
                    anomaly_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    metric_name TEXT,
                    metric_value REAL,
                    expected_range TEXT,
                    timestamp TEXT NOT NULL
                )
                """)
                
                # Create indexes for faster queries
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_usage_personality ON personality_usage (personality_name)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON personality_usage (timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_personality ON personality_performance (personality_name)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_metric ON personality_performance (metric_name)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_anomalies_personality ON personality_anomalies (personality_name)")
                
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error(f"Database setup error: {str(e)}")
            # Fall back to in-memory only mode if database setup fails
            self.db_path = ":memory:"
            logger.warning(f"Falling back to in-memory database")
            self._setup_database()
    
    def record_usage(
        self,
        personality_name: str,
        context: Optional[str] = None,
        duration_ms: Optional[int] = None,
        template_name: Optional[str] = None,
        token_count: Optional[int] = None,
        session_id: Optional[str] = None
    ) -> None:
        """
        Record usage of a personality.
        
        Args:
            personality_name: Name of the personality
            context: Usage context (e.g., "forecast", "chat")
            duration_ms: Execution time in milliseconds
            template_name: Name of the template used
            token_count: Number of tokens used
            session_id: Unique session identifier
        """
        timestamp = datetime.now().isoformat()
        
        # Record in cache
        if personality_name not in self._usage_cache:
            self._usage_cache[personality_name] = {
                "count": 0,
                "last_used": None,
                "total_tokens": 0,
                "contexts": set()
            }
        
        self._usage_cache[personality_name]["count"] += 1
        self._usage_cache[personality_name]["last_used"] = timestamp
        
        if token_count:
            self._usage_cache[personality_name]["total_tokens"] += token_count
        
        if context:
            self._usage_cache[personality_name]["contexts"].add(context)
        
        # Store in database
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO personality_usage 
                    (personality_name, timestamp, context, duration_ms, template_name, token_count, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (personality_name, timestamp, context, duration_ms, template_name, token_count, session_id)
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to record personality usage: {str(e)}")
    
    def record_performance(
        self,
        personality_name: str,
        metric_name: str,
        metric_value: float,
        domain: Optional[str] = None,
        question_id: Optional[str] = None
    ) -> None:
        """
        Record performance metric for a personality.
        
        Args:
            personality_name: Name of the personality
            metric_name: Name of the performance metric
            metric_value: Value of the metric
            domain: Domain of the forecast/question
            question_id: Unique identifier for the question
        """
        timestamp = datetime.now().isoformat()
        
        # Record in cache
        if personality_name not in self._performance_cache:
            self._performance_cache[personality_name] = []
        
        self._performance_cache[personality_name].append({
            "metric": metric_name,
            "value": metric_value,
            "domain": domain,
            "timestamp": timestamp
        })
        
        # Limit cache size to prevent memory issues
        if len(self._performance_cache[personality_name]) > 100:
            self._performance_cache[personality_name] = self._performance_cache[personality_name][-100:]
        
        # Store in database
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO personality_performance 
                    (personality_name, metric_name, metric_value, domain, question_id, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (personality_name, metric_name, metric_value, domain, question_id, timestamp)
                )
                conn.commit()
                
            # Check for anomalies
            self._check_for_anomalies(personality_name, metric_name, metric_value, domain)
                
        except sqlite3.Error as e:
            logger.error(f"Failed to record personality performance: {str(e)}")
    
    def _check_for_anomalies(
        self,
        personality_name: str,
        metric_name: str,
        metric_value: float,
        domain: Optional[str] = None
    ) -> None:
        """
        Check if a metric value is anomalous.
        
        Args:
            personality_name: Name of the personality
            metric_name: Name of the performance metric
            metric_value: Value of the metric
            domain: Domain of the forecast/question
        """
        # Get historical values for this metric
        historical_values = self.get_historical_performance(
            personality_name, 
            metric_name, 
            domain=domain, 
            days=30
        )
        
        # Need at least 5 data points to detect anomalies
        if len(historical_values) < 5:
            return
            
        # Calculate mean and standard deviation
        values = [v[1] for v in historical_values]
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        
        # Define anomaly thresholds (3 standard deviations)
        lower_threshold = mean - (3 * stdev)
        upper_threshold = mean + (3 * stdev)
        
        # Update threshold cache
        key = f"{personality_name}:{metric_name}"
        if domain:
            key += f":{domain}"
            
        self._anomaly_thresholds[key] = {
            "lower": lower_threshold,
            "upper": upper_threshold,
            "mean": mean,
            "stdev": stdev
        }
        
        # Check if current value is outside thresholds
        if metric_value < lower_threshold or metric_value > upper_threshold:
            severity = "high" if abs(metric_value - mean) > (4 * stdev) else "medium"
            
            anomaly_type = "low_value" if metric_value < lower_threshold else "high_value"
            description = f"Anomalous {metric_name} value detected for {personality_name}"
            
            if domain:
                description += f" in domain {domain}"
                
            expected_range = f"{lower_threshold:.3f} - {upper_threshold:.3f}"
            
            # Log the anomaly
            self.record_anomaly(
                personality_name=personality_name,
                anomaly_type=anomaly_type,
                description=description,
                severity=severity,
                metric_name=metric_name,
                metric_value=metric_value,
                expected_range=expected_range
            )
    
    def record_anomaly(
        self,
        personality_name: str,
        anomaly_type: str,
        description: str,
        severity: str,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None,
        expected_range: Optional[str] = None
    ) -> None:
        """
        Record an anomaly in personality behavior.
        
        Args:
            personality_name: Name of the personality
            anomaly_type: Type of anomaly
            description: Description of the anomaly
            severity: Severity level ("low", "medium", "high")
            metric_name: Name of the associated metric
            metric_value: Value of the associated metric
            expected_range: Expected range for the metric
        """
        timestamp = datetime.now().isoformat()
        
        # Log the anomaly
        logger.warning(
            f"Personality anomaly: {description} - {severity} severity "
            f"(value: {metric_value}, expected: {expected_range})"
        )
        
        # Store in database
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO personality_anomalies 
                    (personality_name, anomaly_type, description, severity, 
                     metric_name, metric_value, expected_range, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (personality_name, anomaly_type, description, severity, 
                     metric_name, metric_value, expected_range, timestamp)
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to record personality anomaly: {str(e)}")
    
    def get_usage_statistics(
        self, 
        days: Optional[int] = None,
        personality_name: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get usage statistics for personalities.
        
        Args:
            days: Number of days to include (None for all time)
            personality_name: Filter to a specific personality
            
        Returns:
            Dictionary mapping personality names to usage statistics
        """
        results = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = """
                SELECT 
                    personality_name,
                    COUNT(*) as usage_count,
                    MAX(timestamp) as last_used,
                    AVG(duration_ms) as avg_duration,
                    SUM(token_count) as total_tokens,
                    COUNT(DISTINCT context) as context_count
                FROM personality_usage
                """
                
                params = []
                where_clauses = []
                
                if days is not None:
                    cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
                    where_clauses.append("timestamp >= ?")
                    params.append(cutoff_date)
                
                if personality_name is not None:
                    where_clauses.append("personality_name = ?")
                    params.append(personality_name)
                
                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)
                
                query += " GROUP BY personality_name"
                
                cursor.execute(query, params)
                
                for row in cursor.fetchall():
                    results[row['personality_name']] = {
                        "usage_count": row['usage_count'],
                        "last_used": row['last_used'],
                        "avg_duration": row['avg_duration'],
                        "total_tokens": row['total_tokens'],
                        "context_count": row['context_count']
                    }
                    
                # Get context distribution for each personality
                for personality in results.keys():
                    context_query = """
                    SELECT context, COUNT(*) as count
                    FROM personality_usage
                    WHERE personality_name = ?
                    """
                    
                    context_params = [personality]
                    
                    if days is not None:
                        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
                        context_query += " AND timestamp >= ?"
                        context_params.append(cutoff_date)
                    
                    context_query += " GROUP BY context"
                    
                    cursor.execute(context_query, context_params)
                    
                    results[personality]['contexts'] = {
                        row['context'] or 'unknown': row['count'] 
                        for row in cursor.fetchall()
                    }
                    
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve usage statistics: {str(e)}")
        
        return results
    
    def get_performance_metrics(
        self,
        personality_name: Optional[str] = None,
        metric_name: Optional[str] = None,
        domain: Optional[str] = None,
        days: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Get aggregated performance metrics.
        
        Args:
            personality_name: Filter to a specific personality
            metric_name: Filter to a specific metric
            domain: Filter to a specific domain
            days: Number of days to include
            
        Returns:
            Dictionary of performance metrics
        """
        results = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = """
                SELECT 
                    personality_name,
                    metric_name,
                    AVG(metric_value) as avg_value,
                    MIN(metric_value) as min_value,
                    MAX(metric_value) as max_value,
                    COUNT(*) as count
                FROM personality_performance
                """
                
                params = []
                where_clauses = []
                
                if personality_name is not None:
                    where_clauses.append("personality_name = ?")
                    params.append(personality_name)
                
                if metric_name is not None:
                    where_clauses.append("metric_name = ?")
                    params.append(metric_name)
                
                if domain is not None:
                    where_clauses.append("domain = ?")
                    params.append(domain)
                
                if days is not None:
                    cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
                    where_clauses.append("timestamp >= ?")
                    params.append(cutoff_date)
                
                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)
                
                query += " GROUP BY personality_name, metric_name"
                
                cursor.execute(query, params)
                
                for row in cursor.fetchall():
                    personality = row['personality_name']
                    metric = row['metric_name']
                    
                    if personality not in results:
                        results[personality] = {}
                    
                    results[personality][metric] = {
                        "avg": row['avg_value'],
                        "min": row['min_value'],
                        "max": row['max_value'],
                        "count": row['count']
                    }
                    
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve performance metrics: {str(e)}")
        
        return results
    
    def get_historical_performance(
        self,
        personality_name: str,
        metric_name: str,
        domain: Optional[str] = None,
        days: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Get historical performance data for a specific metric.
        
        Args:
            personality_name: Name of the personality
            metric_name: Name of the metric
            domain: Filter to a specific domain
            days: Number of days to include
            
        Returns:
            List of (timestamp, value) tuples
        """
        results = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = """
                SELECT timestamp, metric_value
                FROM personality_performance
                WHERE personality_name = ? AND metric_name = ?
                """
                
                params = [personality_name, metric_name]
                
                if domain is not None:
                    query += " AND domain = ?"
                    params.append(domain)
                
                if days is not None:
                    cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
                    query += " AND timestamp >= ?"
                    params.append(cutoff_date)
                
                query += " ORDER BY timestamp ASC"
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve historical performance: {str(e)}")
        
        return results
    
    def get_anomalies(
        self,
        personality_name: Optional[str] = None,
        severity: Optional[str] = None,
        days: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recorded anomalies.
        
        Args:
            personality_name: Filter to a specific personality
            severity: Filter to a specific severity
            days: Number of days to include
            
        Returns:
            List of anomaly records
        """
        results = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = """
                SELECT *
                FROM personality_anomalies
                """
                
                params = []
                where_clauses = []
                
                if personality_name is not None:
                    where_clauses.append("personality_name = ?")
                    params.append(personality_name)
                
                if severity is not None:
                    where_clauses.append("severity = ?")
                    params.append(severity)
                
                if days is not None:
                    cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
                    where_clauses.append("timestamp >= ?")
                    params.append(cutoff_date)
                
                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)
                
                query += " ORDER BY timestamp DESC"
                
                cursor.execute(query, params)
                
                for row in cursor.fetchall():
                    results.append(dict(row))
                    
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve anomalies: {str(e)}")
        
        return results
    
    def generate_performance_report(
        self,
        days: int = 30,
        personality_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Args:
            days: Number of days to include
            personality_name: Filter to a specific personality
            
        Returns:
            Report data dictionary
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "period_days": days,
            "personalities": {},
            "anomalies": [],
            "overall_statistics": {}
        }
        
        # Get usage statistics
        usage_stats = self.get_usage_statistics(days=days, personality_name=personality_name)
        
        # Get performance metrics
        metrics = self.get_performance_metrics(days=days, personality_name=personality_name)
        
        # Get anomalies
        anomalies = self.get_anomalies(days=days, personality_name=personality_name)
        report["anomalies"] = anomalies
        
        # Combine data for each personality
        for name in set(list(usage_stats.keys()) + list(metrics.keys())):
            report["personalities"][name] = {
                "usage": usage_stats.get(name, {}),
                "metrics": metrics.get(name, {})
            }
        
        # Calculate overall statistics
        total_usage = sum(stats.get("usage_count", 0) for stats in usage_stats.values())
        avg_metrics = {}
        
        # Combine all metrics across personalities
        for personality_metrics in metrics.values():
            for metric_name, metric_data in personality_metrics.items():
                if metric_name not in avg_metrics:
                    avg_metrics[metric_name] = {
                        "values": [],
                        "counts": []
                    }
                
                avg_metrics[metric_name]["values"].append(metric_data["avg"])
                avg_metrics[metric_name]["counts"].append(metric_data["count"])
        
        # Calculate weighted averages
        overall_metrics = {}
        for metric_name, data in avg_metrics.items():
            values = data["values"]
            counts = data["counts"]
            
            if sum(counts) > 0:
                weighted_avg = sum(v * c for v, c in zip(values, counts)) / sum(counts)
                overall_metrics[metric_name] = weighted_avg
        
        report["overall_statistics"] = {
            "total_usage": total_usage,
            "unique_personalities": len(usage_stats),
            "overall_metrics": overall_metrics,
            "anomaly_count": len(anomalies)
        }
        
        return report
    
    def export_data(self, export_path: str) -> bool:
        """
        Export telemetry data to a JSON file.
        
        Args:
            export_path: Path to export the data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Export usage data
                usage_cursor = conn.cursor()
                usage_cursor.execute("SELECT * FROM personality_usage ORDER BY timestamp")
                usage_data = [dict(row) for row in usage_cursor.fetchall()]
                
                # Export performance data
                perf_cursor = conn.cursor()
                perf_cursor.execute("SELECT * FROM personality_performance ORDER BY timestamp")
                performance_data = [dict(row) for row in perf_cursor.fetchall()]
                
                # Export anomaly data
                anomaly_cursor = conn.cursor()
                anomaly_cursor.execute("SELECT * FROM personality_anomalies ORDER BY timestamp")
                anomaly_data = [dict(row) for row in anomaly_cursor.fetchall()]
                
                # Combine into export data
                export_data = {
                    "exported_at": datetime.now().isoformat(),
                    "usage_data": usage_data,
                    "performance_data": performance_data,
                    "anomaly_data": anomaly_data
                }
                
                # Write to file
                with open(export_path, "w") as f:
                    json.dump(export_data, f, indent=2)
                
                return True
                
        except (sqlite3.Error, IOError) as e:
            logger.error(f"Failed to export telemetry data: {str(e)}")
            return False


# Convenience functions for module-level access
_tracker = None

def get_telemetry_tracker() -> PersonalityTelemetryTracker:
    """Get the singleton telemetry tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = PersonalityTelemetryTracker()
    return _tracker

def record_personality_usage(
    personality_name: str,
    context: Optional[str] = None,
    duration_ms: Optional[int] = None,
    template_name: Optional[str] = None,
    token_count: Optional[int] = None,
    session_id: Optional[str] = None
) -> None:
    """Record personality usage."""
    tracker = get_telemetry_tracker()
    tracker.record_usage(
        personality_name=personality_name,
        context=context,
        duration_ms=duration_ms,
        template_name=template_name,
        token_count=token_count,
        session_id=session_id
    )

def record_personality_performance(
    personality_name: str,
    metric_name: str,
    metric_value: float,
    domain: Optional[str] = None,
    question_id: Optional[str] = None
) -> None:
    """Record personality performance metric."""
    tracker = get_telemetry_tracker()
    tracker.record_performance(
        personality_name=personality_name,
        metric_name=metric_name,
        metric_value=metric_value,
        domain=domain,
        question_id=question_id
    )

def get_personality_report(days: int = 30, personality_name: Optional[str] = None) -> Dict[str, Any]:
    """Generate a personality performance report."""
    tracker = get_telemetry_tracker()
    return tracker.generate_performance_report(days=days, personality_name=personality_name) 