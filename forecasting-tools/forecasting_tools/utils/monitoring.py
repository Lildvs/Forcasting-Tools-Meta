"""
Monitoring System

This module provides comprehensive monitoring for the forecasting tools system,
tracking system performance, API usage, error rates, and other key metrics.
"""

import time
import logging
import threading
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Set
from pathlib import Path
import contextlib
import statistics
import dataclasses
from enum import Enum
import platform
import psutil
import prometheus_client
from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry

# Define metric types
class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclasses.dataclass
class MetricDefinition:
    """Definition of a metric to be tracked."""
    name: str
    description: str
    type: MetricType
    labels: List[str] = dataclasses.field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms

class MonitoringManager:
    """
    Central manager for monitoring and metrics collection.
    
    Provides both Prometheus-compatible metrics and internal tracking.
    """
    
    def __init__(
        self, 
        service_name: str = "forecasting_tools",
        enable_prometheus: bool = False, 
        metrics_port: int = 9090,
        log_metrics: bool = True,
        metrics_log_interval: int = 60
    ):
        self.service_name = service_name
        self.enable_prometheus = enable_prometheus
        self.metrics_port = metrics_port
        self.log_metrics = log_metrics
        self.metrics_log_interval = metrics_log_interval
        
        # Internal tracking
        self.start_time = time.time()
        self.metrics: Dict[str, Any] = {}
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        
        # Prometheus registry
        self.registry = CollectorRegistry()
        self.prometheus_metrics: Dict[str, Any] = {}
        
        # Logging
        self.logger = logging.getLogger("monitoring")
        
        # Initialize default metrics
        self._initialize_default_metrics()
        
        # Start metrics logging if enabled
        if self.log_metrics:
            self._start_metrics_logging()
        
        # Start Prometheus server if enabled
        if self.enable_prometheus:
            self._start_prometheus_server()
    
    def _initialize_default_metrics(self) -> None:
        """Initialize default system and application metrics."""
        # System metrics
        self.register_metric(
            MetricDefinition(
                name="system_cpu_usage",
                description="CPU usage percentage",
                type=MetricType.GAUGE
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="system_memory_usage",
                description="Memory usage in bytes",
                type=MetricType.GAUGE,
                labels=["type"]  # total, available, used, etc.
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="system_disk_usage",
                description="Disk usage in bytes",
                type=MetricType.GAUGE,
                labels=["type", "mount_point"]
            )
        )
        
        # API metrics
        self.register_metric(
            MetricDefinition(
                name="api_requests_total",
                description="Total API requests",
                type=MetricType.COUNTER,
                labels=["endpoint", "method", "status"]
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="api_request_duration_seconds",
                description="API request duration in seconds",
                type=MetricType.HISTOGRAM,
                labels=["endpoint", "method"],
                buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]
            )
        )
        
        # LLM metrics
        self.register_metric(
            MetricDefinition(
                name="llm_requests_total",
                description="Total LLM API requests",
                type=MetricType.COUNTER,
                labels=["model", "provider", "status"]
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="llm_tokens_total",
                description="Total tokens used in LLM requests",
                type=MetricType.COUNTER,
                labels=["model", "provider", "type"]  # type: prompt, completion
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="llm_request_duration_seconds",
                description="LLM request duration in seconds",
                type=MetricType.HISTOGRAM,
                labels=["model", "provider"],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="llm_request_cost_usd",
                description="LLM request cost in USD",
                type=MetricType.COUNTER,
                labels=["model", "provider"]
            )
        )
        
        # Forecasting metrics
        self.register_metric(
            MetricDefinition(
                name="forecasts_generated_total",
                description="Total forecasts generated",
                type=MetricType.COUNTER,
                labels=["forecast_type", "status"]
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="forecast_generation_duration_seconds",
                description="Forecast generation duration in seconds",
                type=MetricType.HISTOGRAM,
                labels=["forecast_type"],
                buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
            )
        )
        
        # Error metrics
        self.register_metric(
            MetricDefinition(
                name="errors_total",
                description="Total errors",
                type=MetricType.COUNTER,
                labels=["component", "error_type"]
            )
        )
        
        # Job queue metrics
        self.register_metric(
            MetricDefinition(
                name="job_queue_size",
                description="Current job queue size",
                type=MetricType.GAUGE,
                labels=["job_type"]
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="job_processing_duration_seconds",
                description="Job processing duration in seconds",
                type=MetricType.HISTOGRAM,
                labels=["job_type", "status"],
                buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0]
            )
        )
        
        # Database metrics
        self.register_metric(
            MetricDefinition(
                name="database_operations_total",
                description="Total database operations",
                type=MetricType.COUNTER,
                labels=["operation", "model"]
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="database_operation_duration_seconds",
                description="Database operation duration in seconds",
                type=MetricType.HISTOGRAM,
                labels=["operation", "model"],
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
            )
        )
        
        # Cache metrics
        self.register_metric(
            MetricDefinition(
                name="cache_operations_total",
                description="Total cache operations",
                type=MetricType.COUNTER,
                labels=["operation", "cache_type", "status"]
            )
        )
        
        self.register_metric(
            MetricDefinition(
                name="cache_hit_ratio",
                description="Cache hit ratio",
                type=MetricType.GAUGE,
                labels=["cache_type"]
            )
        )
    
    def register_metric(self, definition: MetricDefinition) -> None:
        """
        Register a new metric for tracking.
        
        Args:
            definition: Metric definition
        """
        metric_name = f"{self.service_name}_{definition.name}"
        self.metric_definitions[definition.name] = definition
        
        # Create internal tracking data structure
        if definition.type in (MetricType.COUNTER, MetricType.GAUGE):
            self.metrics[definition.name] = {}
        elif definition.type == MetricType.HISTOGRAM:
            self.metrics[definition.name] = {
                "values": [],
                "labels": {}
            }
        elif definition.type == MetricType.SUMMARY:
            self.metrics[definition.name] = {
                "values": [],
                "count": 0,
                "sum": 0,
                "labels": {}
            }
        
        # Create Prometheus metric if enabled
        if self.enable_prometheus:
            if definition.type == MetricType.COUNTER:
                self.prometheus_metrics[definition.name] = Counter(
                    metric_name,
                    definition.description,
                    definition.labels,
                    registry=self.registry
                )
            elif definition.type == MetricType.GAUGE:
                self.prometheus_metrics[definition.name] = Gauge(
                    metric_name,
                    definition.description,
                    definition.labels,
                    registry=self.registry
                )
            elif definition.type == MetricType.HISTOGRAM:
                self.prometheus_metrics[definition.name] = Histogram(
                    metric_name,
                    definition.description,
                    definition.labels,
                    buckets=definition.buckets,
                    registry=self.registry
                )
            elif definition.type == MetricType.SUMMARY:
                self.prometheus_metrics[definition.name] = Summary(
                    metric_name,
                    definition.description,
                    definition.labels,
                    registry=self.registry
                )
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Metric name
            value: Value to increment by
            labels: Labels to apply to the metric
        """
        if name not in self.metric_definitions:
            self.logger.warning(f"Attempted to increment unknown metric: {name}")
            return
        
        definition = self.metric_definitions[name]
        if definition.type != MetricType.COUNTER:
            self.logger.warning(f"Attempted to increment non-counter metric: {name}")
            return
        
        # Update internal metrics
        labels_tuple = self._labels_dict_to_tuple(labels, definition.labels)
        if labels_tuple not in self.metrics[name]:
            self.metrics[name][labels_tuple] = 0
        
        self.metrics[name][labels_tuple] += value
        
        # Update Prometheus metrics if enabled
        if self.enable_prometheus:
            if labels:
                self.prometheus_metrics[name].labels(**labels).inc(value)
            else:
                self.prometheus_metrics[name].inc(value)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Set a gauge metric.
        
        Args:
            name: Metric name
            value: Value to set
            labels: Labels to apply to the metric
        """
        if name not in self.metric_definitions:
            self.logger.warning(f"Attempted to set unknown metric: {name}")
            return
        
        definition = self.metric_definitions[name]
        if definition.type != MetricType.GAUGE:
            self.logger.warning(f"Attempted to set non-gauge metric: {name}")
            return
        
        # Update internal metrics
        labels_tuple = self._labels_dict_to_tuple(labels, definition.labels)
        self.metrics[name][labels_tuple] = value
        
        # Update Prometheus metrics if enabled
        if self.enable_prometheus:
            if labels:
                self.prometheus_metrics[name].labels(**labels).set(value)
            else:
                self.prometheus_metrics[name].set(value)
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Observe a value for a histogram metric.
        
        Args:
            name: Metric name
            value: Value to observe
            labels: Labels to apply to the metric
        """
        if name not in self.metric_definitions:
            self.logger.warning(f"Attempted to observe unknown metric: {name}")
            return
        
        definition = self.metric_definitions[name]
        if definition.type != MetricType.HISTOGRAM:
            self.logger.warning(f"Attempted to observe non-histogram metric: {name}")
            return
        
        # Update internal metrics
        self.metrics[name]["values"].append(value)
        
        labels_tuple = self._labels_dict_to_tuple(labels, definition.labels)
        if labels_tuple not in self.metrics[name]["labels"]:
            self.metrics[name]["labels"][labels_tuple] = []
        
        self.metrics[name]["labels"][labels_tuple].append(value)
        
        # Update Prometheus metrics if enabled
        if self.enable_prometheus:
            if labels:
                self.prometheus_metrics[name].labels(**labels).observe(value)
            else:
                self.prometheus_metrics[name].observe(value)
    
    def observe_summary(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Observe a value for a summary metric.
        
        Args:
            name: Metric name
            value: Value to observe
            labels: Labels to apply to the metric
        """
        if name not in self.metric_definitions:
            self.logger.warning(f"Attempted to observe unknown metric: {name}")
            return
        
        definition = self.metric_definitions[name]
        if definition.type != MetricType.SUMMARY:
            self.logger.warning(f"Attempted to observe non-summary metric: {name}")
            return
        
        # Update internal metrics
        self.metrics[name]["values"].append(value)
        self.metrics[name]["count"] += 1
        self.metrics[name]["sum"] += value
        
        labels_tuple = self._labels_dict_to_tuple(labels, definition.labels)
        if labels_tuple not in self.metrics[name]["labels"]:
            self.metrics[name]["labels"][labels_tuple] = {
                "values": [],
                "count": 0,
                "sum": 0
            }
        
        self.metrics[name]["labels"][labels_tuple]["values"].append(value)
        self.metrics[name]["labels"][labels_tuple]["count"] += 1
        self.metrics[name]["labels"][labels_tuple]["sum"] += value
        
        # Update Prometheus metrics if enabled
        if self.enable_prometheus:
            if labels:
                self.prometheus_metrics[name].labels(**labels).observe(value)
            else:
                self.prometheus_metrics[name].observe(value)
    
    def track_api_request(
        self, 
        endpoint: str, 
        method: str, 
        duration: float, 
        status: str = "success"
    ) -> None:
        """
        Track an API request.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            duration: Request duration in seconds
            status: Request status (success, error, etc.)
        """
        self.increment_counter(
            "api_requests_total", 
            labels={"endpoint": endpoint, "method": method, "status": status}
        )
        
        self.observe_histogram(
            "api_request_duration_seconds",
            duration,
            labels={"endpoint": endpoint, "method": method}
        )
    
    def track_llm_request(
        self,
        model: str,
        provider: str,
        duration: float,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        status: str = "success"
    ) -> None:
        """
        Track an LLM API request.
        
        Args:
            model: LLM model name
            provider: LLM provider (OpenAI, Anthropic, etc.)
            duration: Request duration in seconds
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            cost: Request cost in USD
            status: Request status (success, error, etc.)
        """
        self.increment_counter(
            "llm_requests_total",
            labels={"model": model, "provider": provider, "status": status}
        )
        
        self.increment_counter(
            "llm_tokens_total",
            value=prompt_tokens,
            labels={"model": model, "provider": provider, "type": "prompt"}
        )
        
        self.increment_counter(
            "llm_tokens_total",
            value=completion_tokens,
            labels={"model": model, "provider": provider, "type": "completion"}
        )
        
        self.observe_histogram(
            "llm_request_duration_seconds",
            duration,
            labels={"model": model, "provider": provider}
        )
        
        self.increment_counter(
            "llm_request_cost_usd",
            value=cost,
            labels={"model": model, "provider": provider}
        )
    
    def track_forecast_generation(
        self,
        forecast_type: str,
        duration: float,
        status: str = "success"
    ) -> None:
        """
        Track a forecast generation.
        
        Args:
            forecast_type: Type of forecast (binary, numeric, etc.)
            duration: Generation duration in seconds
            status: Generation status (success, error, etc.)
        """
        self.increment_counter(
            "forecasts_generated_total",
            labels={"forecast_type": forecast_type, "status": status}
        )
        
        self.observe_histogram(
            "forecast_generation_duration_seconds",
            duration,
            labels={"forecast_type": forecast_type}
        )
    
    def track_error(self, component: str, error_type: str) -> None:
        """
        Track an error.
        
        Args:
            component: Component where the error occurred
            error_type: Type of error
        """
        self.increment_counter(
            "errors_total",
            labels={"component": component, "error_type": error_type}
        )
    
    def track_job(self, job_type: str, duration: float, status: str = "success") -> None:
        """
        Track a job execution.
        
        Args:
            job_type: Type of job
            duration: Job duration in seconds
            status: Job status (success, error, etc.)
        """
        self.observe_histogram(
            "job_processing_duration_seconds",
            duration,
            labels={"job_type": job_type, "status": status}
        )
    
    def set_job_queue_size(self, job_type: str, size: int) -> None:
        """
        Set the job queue size.
        
        Args:
            job_type: Type of job
            size: Queue size
        """
        self.set_gauge(
            "job_queue_size",
            size,
            labels={"job_type": job_type}
        )
    
    def track_database_operation(
        self,
        operation: str,
        model: str,
        duration: float
    ) -> None:
        """
        Track a database operation.
        
        Args:
            operation: Operation type (query, insert, update, etc.)
            model: Database model
            duration: Operation duration in seconds
        """
        self.increment_counter(
            "database_operations_total",
            labels={"operation": operation, "model": model}
        )
        
        self.observe_histogram(
            "database_operation_duration_seconds",
            duration,
            labels={"operation": operation, "model": model}
        )
    
    def track_cache_operation(
        self,
        operation: str,
        cache_type: str,
        status: str = "hit"
    ) -> None:
        """
        Track a cache operation.
        
        Args:
            operation: Operation type (get, set, etc.)
            cache_type: Type of cache (memory, disk, redis, etc.)
            status: Operation status (hit, miss, error, etc.)
        """
        self.increment_counter(
            "cache_operations_total",
            labels={"operation": operation, "cache_type": cache_type, "status": status}
        )
        
        # Update cache hit ratio if this is a get operation
        if operation == "get":
            hits = self.metrics["cache_operations_total"].get(
                self._labels_dict_to_tuple(
                    {"operation": "get", "cache_type": cache_type, "status": "hit"},
                    self.metric_definitions["cache_operations_total"].labels
                ), 0
            )
            
            misses = self.metrics["cache_operations_total"].get(
                self._labels_dict_to_tuple(
                    {"operation": "get", "cache_type": cache_type, "status": "miss"},
                    self.metric_definitions["cache_operations_total"].labels
                ), 0
            )
            
            total = hits + misses
            hit_ratio = hits / total if total > 0 else 0
            
            self.set_gauge(
                "cache_hit_ratio",
                hit_ratio,
                labels={"cache_type": cache_type}
            )
    
    def update_system_metrics(self) -> None:
        """Update system metrics (CPU, memory, disk)."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent()
        self.set_gauge("system_cpu_usage", cpu_percent)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.set_gauge("system_memory_usage", memory.total, labels={"type": "total"})
        self.set_gauge("system_memory_usage", memory.available, labels={"type": "available"})
        self.set_gauge("system_memory_usage", memory.used, labels={"type": "used"})
        self.set_gauge("system_memory_usage", memory.free, labels={"type": "free"})
        
        # Disk metrics
        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
                self.set_gauge(
                    "system_disk_usage", 
                    usage.total, 
                    labels={"type": "total", "mount_point": part.mountpoint}
                )
                self.set_gauge(
                    "system_disk_usage", 
                    usage.used, 
                    labels={"type": "used", "mount_point": part.mountpoint}
                )
                self.set_gauge(
                    "system_disk_usage", 
                    usage.free, 
                    labels={"type": "free", "mount_point": part.mountpoint}
                )
            except (PermissionError, FileNotFoundError):
                pass
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Get a summary of all metrics.
        
        Returns:
            Dictionary of metric summaries
        """
        summary = {}
        
        for name, definition in self.metric_definitions.items():
            if definition.type == MetricType.COUNTER:
                summary[name] = {
                    "type": "counter",
                    "description": definition.description,
                    "value": sum(self.metrics[name].values()),
                    "labels": {
                        self._labels_tuple_to_dict(labels_tuple, definition.labels): value
                        for labels_tuple, value in self.metrics[name].items()
                    }
                }
            elif definition.type == MetricType.GAUGE:
                # Find the most recent value for each label combination
                label_values = {}
                for labels_tuple, value in self.metrics[name].items():
                    label_values[self._labels_tuple_to_dict(labels_tuple, definition.labels)] = value
                
                summary[name] = {
                    "type": "gauge",
                    "description": definition.description,
                    "values": label_values
                }
            elif definition.type == MetricType.HISTOGRAM:
                values = self.metrics[name]["values"]
                
                if not values:
                    summary[name] = {
                        "type": "histogram",
                        "description": definition.description,
                        "count": 0,
                        "labels": {}
                    }
                    continue
                
                label_stats = {}
                for labels_tuple, label_values in self.metrics[name]["labels"].items():
                    if not label_values:
                        continue
                    
                    label_stats[self._labels_tuple_to_dict(labels_tuple, definition.labels)] = {
                        "count": len(label_values),
                        "min": min(label_values),
                        "max": max(label_values),
                        "mean": statistics.mean(label_values),
                        "median": statistics.median(label_values),
                        "p95": statistics.quantiles(label_values, n=20)[18] if len(label_values) >= 20 else None,
                        "p99": statistics.quantiles(label_values, n=100)[98] if len(label_values) >= 100 else None
                    }
                
                summary[name] = {
                    "type": "histogram",
                    "description": definition.description,
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "p95": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else None,
                    "p99": statistics.quantiles(values, n=100)[98] if len(values) >= 100 else None,
                    "labels": label_stats
                }
            elif definition.type == MetricType.SUMMARY:
                values = self.metrics[name]["values"]
                count = self.metrics[name]["count"]
                
                if count == 0:
                    summary[name] = {
                        "type": "summary",
                        "description": definition.description,
                        "count": 0,
                        "labels": {}
                    }
                    continue
                
                label_stats = {}
                for labels_tuple, label_data in self.metrics[name]["labels"].items():
                    label_values = label_data["values"]
                    label_count = label_data["count"]
                    
                    if label_count == 0:
                        continue
                    
                    label_stats[self._labels_tuple_to_dict(labels_tuple, definition.labels)] = {
                        "count": label_count,
                        "sum": label_data["sum"],
                        "mean": label_data["sum"] / label_count,
                        "min": min(label_values),
                        "max": max(label_values)
                    }
                
                summary[name] = {
                    "type": "summary",
                    "description": definition.description,
                    "count": count,
                    "sum": self.metrics[name]["sum"],
                    "mean": self.metrics[name]["sum"] / count,
                    "labels": label_stats
                }
        
        return summary
    
    def _labels_dict_to_tuple(self, labels_dict: Optional[Dict[str, str]], label_names: List[str]) -> tuple:
        """
        Convert a labels dictionary to a tuple for internal storage.
        
        Args:
            labels_dict: Labels dictionary
            label_names: Expected label names
        
        Returns:
            Tuple representation of labels
        """
        if not labels_dict:
            return tuple()
        
        # Ensure only valid labels are used
        filtered_dict = {k: v for k, v in labels_dict.items() if k in label_names}
        
        # Sort by label name to ensure consistent ordering
        return tuple(sorted(filtered_dict.items()))
    
    def _labels_tuple_to_dict(self, labels_tuple: tuple, label_names: List[str]) -> Dict[str, str]:
        """
        Convert a labels tuple to a dictionary.
        
        Args:
            labels_tuple: Labels tuple
            label_names: Expected label names
        
        Returns:
            Dictionary representation of labels
        """
        return dict(labels_tuple)
    
    def _start_prometheus_server(self) -> None:
        """Start the Prometheus metrics server."""
        try:
            prometheus_client.start_http_server(self.metrics_port, registry=self.registry)
            self.logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
        except Exception as e:
            self.logger.error(f"Failed to start Prometheus metrics server: {e}")
    
    def _start_metrics_logging(self) -> None:
        """Start periodic metrics logging."""
        def log_metrics():
            while True:
                try:
                    # Update system metrics
                    self.update_system_metrics()
                    
                    # Log summary metrics
                    summary = self.get_summary_metrics()
                    self.logger.info(f"Metrics summary: {json.dumps(summary, default=str)}")
                except Exception as e:
                    self.logger.error(f"Error logging metrics: {e}")
                
                time.sleep(self.metrics_log_interval)
        
        # Start metrics logging thread
        thread = threading.Thread(target=log_metrics, daemon=True)
        thread.start()
    
    def export_metrics(self, file_path: str) -> None:
        """
        Export metrics to a JSON file.
        
        Args:
            file_path: Path to export file
        """
        try:
            summary = self.get_summary_metrics()
            
            with open(file_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"Metrics exported to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")


@contextlib.contextmanager
def timed_operation(
    manager: MonitoringManager,
    metric_name: str,
    labels: Optional[Dict[str, str]] = None
) -> None:
    """
    Context manager for timing operations.
    
    Args:
        manager: Monitoring manager
        metric_name: Metric name to observe
        labels: Labels to apply to the metric
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        
        if metric_name.endswith("_duration_seconds"):
            manager.observe_histogram(metric_name, duration, labels)
        else:
            manager.observe_histogram(f"{metric_name}_duration_seconds", duration, labels)


# Global monitoring manager instance
_monitoring_manager: Optional[MonitoringManager] = None

def get_monitoring_manager() -> MonitoringManager:
    """
    Get the global monitoring manager instance.
    
    Returns:
        Global monitoring manager
    """
    global _monitoring_manager
    
    if _monitoring_manager is None:
        # Get configuration from environment or config
        from forecasting_tools.config import SystemConfig
        
        enable_prometheus = SystemConfig.ENABLE_METRICS
        metrics_port = SystemConfig.METRICS_PORT
        
        _monitoring_manager = MonitoringManager(
            enable_prometheus=enable_prometheus,
            metrics_port=metrics_port
        )
    
    return _monitoring_manager 