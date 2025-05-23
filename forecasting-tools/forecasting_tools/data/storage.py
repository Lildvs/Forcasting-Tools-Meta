"""
Data Persistence Layer

This module provides a data persistence layer for storing forecasts, research results,
and user interactions. It supports multiple backends including SQLite, PostgreSQL,
and in-memory storage for testing.
"""

import json
import sqlite3
import datetime
import os
from typing import Any, Dict, List, Optional, Union, Tuple, TypeVar, Generic, Callable
from contextlib import contextmanager
from pathlib import Path
import logging
from abc import ABC, abstractmethod
import threading
import time

T = TypeVar('T')

class PersistenceError(Exception):
    """Base exception for persistence errors."""
    pass

class StorageAdapter(ABC):
    """Abstract base class for storage adapters."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the storage backend."""
        pass
    
    @abstractmethod
    def save(self, collection: str, data: Dict[str, Any], key: Optional[str] = None) -> str:
        """
        Save data to the storage backend.
        
        Args:
            collection: Collection name
            data: Data to save
            key: Optional key for the data
            
        Returns:
            Key of the saved data
        """
        pass
    
    @abstractmethod
    def get(self, collection: str, key: str) -> Optional[Dict[str, Any]]:
        """
        Get data from the storage backend.
        
        Args:
            collection: Collection name
            key: Key of the data
            
        Returns:
            Data or None if not found
        """
        pass
    
    @abstractmethod
    def update(self, collection: str, key: str, data: Dict[str, Any], upsert: bool = False) -> bool:
        """
        Update data in the storage backend.
        
        Args:
            collection: Collection name
            key: Key of the data
            data: Data to update
            upsert: Whether to insert if not exists
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def delete(self, collection: str, key: str) -> bool:
        """
        Delete data from the storage backend.
        
        Args:
            collection: Collection name
            key: Key of the data
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def query(self, collection: str, query: Dict[str, Any], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query data from the storage backend.
        
        Args:
            collection: Collection name
            query: Query parameters
            limit: Maximum number of results
            
        Returns:
            List of matching data
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the storage backend connection."""
        pass


class SqliteStorage(StorageAdapter):
    """SQLite storage adapter."""
    
    def __init__(self, db_path: Union[str, Path], pool_size: int = 5):
        self.db_path = Path(db_path)
        self.pool_size = pool_size
        self.connection_pool = []
        self.pool_lock = threading.Lock()
        self.initialized = False
    
    def initialize(self) -> None:
        """Initialize the SQLite database and create connection pool."""
        # Create directory if not exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize connection pool
        with self.pool_lock:
            for _ in range(self.pool_size):
                conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
                conn.row_factory = sqlite3.Row
                self.connection_pool.append(conn)
        
        # Create tables if not exist
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS storage_collections (
                    name TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS storage_data (
                    collection TEXT,
                    key TEXT,
                    data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (collection, key),
                    FOREIGN KEY (collection) REFERENCES storage_collections(name)
                )
            """)
            
            conn.commit()
        
        self.initialized = True
    
    @contextmanager
    def _get_connection(self):
        """Get a connection from the pool."""
        if not self.initialized:
            self.initialize()
        
        conn = None
        try:
            with self.pool_lock:
                if self.connection_pool:
                    conn = self.connection_pool.pop()
                else:
                    conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
                    conn.row_factory = sqlite3.Row
            
            yield conn
            
            # Reset connection state and return to pool
            if conn.in_transaction:
                conn.rollback()
            
            with self.pool_lock:
                self.connection_pool.append(conn)
                conn = None
        finally:
            # If connection wasn't returned to pool, close it
            if conn is not None:
                conn.close()
    
    def _ensure_collection(self, collection: str) -> None:
        """Ensure collection exists in the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO storage_collections (name) VALUES (?)",
                (collection,)
            )
            conn.commit()
    
    def save(self, collection: str, data: Dict[str, Any], key: Optional[str] = None) -> str:
        """Save data to SQLite."""
        self._ensure_collection(collection)
        
        # Generate key if not provided
        if key is None:
            key = f"{int(time.time() * 1000)}_{os.urandom(4).hex()}"
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO storage_data (collection, key, data)
                VALUES (?, ?, ?)
                ON CONFLICT (collection, key) DO UPDATE SET
                    data = excluded.data,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (collection, key, json.dumps(data))
            )
            conn.commit()
        
        return key
    
    def get(self, collection: str, key: str) -> Optional[Dict[str, Any]]:
        """Get data from SQLite."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT data FROM storage_data WHERE collection = ? AND key = ?",
                (collection, key)
            )
            result = cursor.fetchone()
        
        if result:
            return json.loads(result[0])
        return None
    
    def update(self, collection: str, key: str, data: Dict[str, Any], upsert: bool = False) -> bool:
        """Update data in SQLite."""
        self._ensure_collection(collection)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if upsert:
                cursor.execute(
                    """
                    INSERT INTO storage_data (collection, key, data)
                    VALUES (?, ?, ?)
                    ON CONFLICT (collection, key) DO UPDATE SET
                        data = excluded.data,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (collection, key, json.dumps(data))
                )
            else:
                cursor.execute(
                    """
                    UPDATE storage_data SET
                        data = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE collection = ? AND key = ?
                    """,
                    (json.dumps(data), collection, key)
                )
            
            conn.commit()
            return cursor.rowcount > 0
    
    def delete(self, collection: str, key: str) -> bool:
        """Delete data from SQLite."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM storage_data WHERE collection = ? AND key = ?",
                (collection, key)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def query(self, collection: str, query: Dict[str, Any], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Query data from SQLite."""
        # Basic implementation - in a real app would use SQL query builder
        # For now, we'll load all records and filter in Python
        with self._get_connection() as conn:
            cursor = conn.cursor()
            sql = "SELECT data FROM storage_data WHERE collection = ?"
            params = [collection]
            
            if limit is not None:
                sql += f" LIMIT {limit}"
            
            cursor.execute(sql, params)
            results = cursor.fetchall()
        
        # Parse and filter results
        items = []
        for row in results:
            item = json.loads(row[0])
            
            # Check if item matches query
            matches = True
            for key, value in query.items():
                if key not in item or item[key] != value:
                    matches = False
                    break
            
            if matches:
                items.append(item)
                if limit is not None and len(items) >= limit:
                    break
        
        return items
    
    def close(self) -> None:
        """Close all connections in the pool."""
        with self.pool_lock:
            for conn in self.connection_pool:
                conn.close()
            self.connection_pool = []
        self.initialized = False


class MemoryStorage(StorageAdapter):
    """In-memory storage adapter for testing."""
    
    def __init__(self):
        self.collections: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.initialized = False
    
    def initialize(self) -> None:
        """Initialize the in-memory storage."""
        self.collections = {}
        self.initialized = True
    
    def save(self, collection: str, data: Dict[str, Any], key: Optional[str] = None) -> str:
        """Save data to memory."""
        if not self.initialized:
            self.initialize()
        
        # Create collection if not exists
        if collection not in self.collections:
            self.collections[collection] = {}
        
        # Generate key if not provided
        if key is None:
            key = f"{int(time.time() * 1000)}_{os.urandom(4).hex()}"
        
        # Add timestamps
        if 'created_at' not in data:
            data['created_at'] = datetime.datetime.now().isoformat()
        data['updated_at'] = datetime.datetime.now().isoformat()
        
        # Save data
        self.collections[collection][key] = data.copy()
        
        return key
    
    def get(self, collection: str, key: str) -> Optional[Dict[str, Any]]:
        """Get data from memory."""
        if not self.initialized:
            self.initialize()
        
        if collection not in self.collections or key not in self.collections[collection]:
            return None
        
        return self.collections[collection][key].copy()
    
    def update(self, collection: str, key: str, data: Dict[str, Any], upsert: bool = False) -> bool:
        """Update data in memory."""
        if not self.initialized:
            self.initialize()
        
        # Check if exists
        if collection not in self.collections or key not in self.collections[collection]:
            if not upsert:
                return False
            
            # Create collection if not exists
            if collection not in self.collections:
                self.collections[collection] = {}
            
            # Add created_at timestamp
            if 'created_at' not in data:
                data['created_at'] = datetime.datetime.now().isoformat()
        else:
            # Preserve created_at timestamp
            data['created_at'] = self.collections[collection][key].get('created_at')
        
        # Update timestamp
        data['updated_at'] = datetime.datetime.now().isoformat()
        
        # Save data
        self.collections[collection][key] = data.copy()
        
        return True
    
    def delete(self, collection: str, key: str) -> bool:
        """Delete data from memory."""
        if not self.initialized:
            self.initialize()
        
        if collection not in self.collections or key not in self.collections[collection]:
            return False
        
        del self.collections[collection][key]
        return True
    
    def query(self, collection: str, query: Dict[str, Any], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Query data from memory."""
        if not self.initialized:
            self.initialize()
        
        if collection not in self.collections:
            return []
        
        results = []
        
        for item in self.collections[collection].values():
            # Check if item matches query
            matches = True
            for key, value in query.items():
                if key not in item or item[key] != value:
                    matches = False
                    break
            
            if matches:
                results.append(item.copy())
                if limit is not None and len(results) >= limit:
                    break
        
        return results
    
    def close(self) -> None:
        """Close the memory storage."""
        self.collections = {}
        self.initialized = False


class StorageManager:
    """
    Manager for handling different storage backends.
    
    Provides a unified interface for storage operations across different backends.
    """
    
    def __init__(self, adapter: StorageAdapter):
        self.adapter = adapter
        self.adapter.initialize()
    
    def save_forecast(self, forecast_data: Dict[str, Any], forecast_id: Optional[str] = None) -> str:
        """
        Save a forecast to storage.
        
        Args:
            forecast_data: Forecast data to save
            forecast_id: Optional forecast ID
            
        Returns:
            Forecast ID
        """
        return self.adapter.save("forecasts", forecast_data, forecast_id)
    
    def get_forecast(self, forecast_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a forecast from storage.
        
        Args:
            forecast_id: Forecast ID
            
        Returns:
            Forecast data or None if not found
        """
        return self.adapter.get("forecasts", forecast_id)
    
    def save_research(self, research_data: Dict[str, Any], research_id: Optional[str] = None) -> str:
        """
        Save research results to storage.
        
        Args:
            research_data: Research data to save
            research_id: Optional research ID
            
        Returns:
            Research ID
        """
        return self.adapter.save("research", research_data, research_id)
    
    def get_research(self, research_id: str) -> Optional[Dict[str, Any]]:
        """
        Get research results from storage.
        
        Args:
            research_id: Research ID
            
        Returns:
            Research data or None if not found
        """
        return self.adapter.get("research", research_id)
    
    def save_user_interaction(self, interaction_data: Dict[str, Any], interaction_id: Optional[str] = None) -> str:
        """
        Save user interaction to storage.
        
        Args:
            interaction_data: Interaction data to save
            interaction_id: Optional interaction ID
            
        Returns:
            Interaction ID
        """
        return self.adapter.save("user_interactions", interaction_data, interaction_id)
    
    def query_forecasts(self, query: Dict[str, Any], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query forecasts from storage.
        
        Args:
            query: Query parameters
            limit: Maximum number of results
            
        Returns:
            List of matching forecasts
        """
        return self.adapter.query("forecasts", query, limit)
    
    def query_research(self, query: Dict[str, Any], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query research results from storage.
        
        Args:
            query: Query parameters
            limit: Maximum number of results
            
        Returns:
            List of matching research results
        """
        return self.adapter.query("research", query, limit)
    
    def close(self) -> None:
        """Close the storage manager."""
        self.adapter.close() 