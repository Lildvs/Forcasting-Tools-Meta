import unittest
import logging
import json
import os
import re
import tempfile
import threading
from io import StringIO

from forecasting_tools.utils.logging_config import (
    setup_logging, SensitiveDataFilter, ContextFilter, JsonFormatter,
    LogLevel, LogFormat, generate_request_id, set_request_id,
    get_request_id, request_context, log_execution_time, get_logger
)


class TestLoggingConfig(unittest.TestCase):
    """Tests for the logging_config module."""

    def setUp(self):
        """Set up test environment."""
        # Reset logging to default configuration
        logging.root.handlers = []
        logging.root.setLevel(logging.WARNING)
        
        # Create a stream handler for capturing logs
        self.log_output = StringIO()
        self.stream_handler = logging.StreamHandler(self.log_output)
        self.stream_handler.setLevel(logging.DEBUG)
        
        # Create a test logger
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        self.logger.addHandler(self.stream_handler)
        self.logger.propagate = False
    
    def tearDown(self):
        """Clean up after tests."""
        self.log_output.close()
    
    def test_sensitive_data_filter(self):
        """Test that sensitive data filter redacts sensitive information."""
        # Create the filter
        sensitive_filter = SensitiveDataFilter()
        
        # Add the filter to our logger
        self.stream_handler.addFilter(sensitive_filter)
        
        # Log messages with sensitive data
        self.logger.info("API key is api_key='secret-api-key-12345'")
        self.logger.info("Password is password='super-secret-password'")
        self.logger.info("This is a token='{}'".format("abc123token456"))
        
        # Check output
        output = self.log_output.getvalue()
        
        # Sensitive data should be redacted
        self.assertIn("api_key=\"***REDACTED***\"", output)
        self.assertIn("password=\"***REDACTED***\"", output)
        self.assertNotIn("secret-api-key-12345", output)
        self.assertNotIn("super-secret-password", output)
    
    def test_context_filter(self):
        """Test that context filter adds context information."""
        # Create the filter with custom context
        context_filter = ContextFilter(
            app_name="test-app",
            context={"environment": "testing", "version": "1.0.0"}
        )
        
        # Create a custom formatter that includes context fields
        formatter = logging.Formatter(
            "%(hostname)s %(app_name)s %(environment)s %(version)s - %(message)s"
        )
        self.stream_handler.setFormatter(formatter)
        
        # Add the filter to our logger
        self.stream_handler.addFilter(context_filter)
        
        # Log a message
        self.logger.info("This is a test message")
        
        # Check output
        output = self.log_output.getvalue()
        
        # Context information should be included
        self.assertIn("test-app", output)
        self.assertIn("testing", output)
        self.assertIn("1.0.0", output)
    
    def test_json_formatter(self):
        """Test that JSON formatter produces valid JSON."""
        # Create the formatter
        json_formatter = JsonFormatter()
        self.stream_handler.setFormatter(json_formatter)
        
        # Add context filter to provide extra fields
        context_filter = ContextFilter(app_name="test-app")
        self.stream_handler.addFilter(context_filter)
        
        # Log a message
        self.logger.info("This is a test message")
        
        # Check output
        output = self.log_output.getvalue()
        
        # Should be valid JSON
        try:
            log_data = json.loads(output)
            self.assertIsInstance(log_data, dict)
            self.assertEqual(log_data["level"], "INFO")
            self.assertEqual(log_data["name"], "test_logger")
            self.assertIn("This is a test message", log_data["message"])
            self.assertEqual(log_data["app_name"], "test-app")
        except json.JSONDecodeError:
            self.fail("Output is not valid JSON")
    
    def test_setup_logging_text_format(self):
        """Test setup_logging with text format."""
        # Set up logging with text format
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            setup_logging(
                level=LogLevel.DEBUG,
                log_format=LogFormat.TEXT,
                log_file=temp_path,
                app_name="test-app"
            )
            
            # Get a logger
            logger = logging.getLogger("test_setup")
            
            # Log a message
            logger.debug("Debug message")
            logger.info("Info message")
            
            # Check log file
            with open(temp_path, "r") as f:
                log_content = f.read()
            
            # Should contain our messages in text format
            self.assertIn("Debug message", log_content)
            self.assertIn("Info message", log_content)
            self.assertIn("[DEBUG]", log_content)
            self.assertIn("[INFO]", log_content)
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_setup_logging_json_format(self):
        """Test setup_logging with JSON format."""
        # Set up logging with JSON format
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            setup_logging(
                level=LogLevel.DEBUG,
                log_format=LogFormat.JSON,
                log_file=temp_path,
                app_name="test-app"
            )
            
            # Get a logger
            logger = logging.getLogger("test_setup_json")
            
            # Log a message
            logger.info("Info message")
            
            # Check log file
            with open(temp_path, "r") as f:
                log_content = f.read()
            
            # Should be valid JSON
            try:
                # Find the JSON object in the log file (there may be multiple lines)
                for line in log_content.splitlines():
                    if "Info message" in line:
                        log_data = json.loads(line)
                        self.assertIsInstance(log_data, dict)
                        self.assertEqual(log_data["level"], "INFO")
                        self.assertEqual(log_data["name"], "test_setup_json")
                        self.assertIn("Info message", log_data["message"])
                        self.assertEqual(log_data["app_name"], "test-app")
                        break
                else:
                    self.fail("Log message not found in file")
            except json.JSONDecodeError:
                self.fail("Output is not valid JSON")
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_request_id_functions(self):
        """Test request ID functions."""
        # Generate a request ID
        request_id = generate_request_id()
        self.assertIsInstance(request_id, str)
        self.assertGreater(len(request_id), 10)  # Should be a substantial string
        
        # Set and get request ID
        set_request_id(request_id)
        retrieved_id = get_request_id()
        self.assertEqual(retrieved_id, request_id)
        
        # Should be thread-specific
        def thread_func():
            # Should be None in a different thread
            self.assertIsNone(get_request_id())
            
            # Set a different ID
            thread_id = "thread-specific-id"
            set_request_id(thread_id)
            self.assertEqual(get_request_id(), thread_id)
            
            # Shouldn't affect the main thread
            self.assertNotEqual(get_request_id(), request_id)
        
        thread = threading.Thread(target=thread_func)
        thread.start()
        thread.join()
        
        # Main thread should still have the original ID
        self.assertEqual(get_request_id(), request_id)
    
    def test_request_context(self):
        """Test request_context context manager."""
        # Set initial request ID
        initial_id = "initial-id"
        set_request_id(initial_id)
        
        # Use context manager
        with request_context() as ctx_id:
            # Should have a new ID
            self.assertNotEqual(get_request_id(), initial_id)
            self.assertEqual(get_request_id(), ctx_id)
            
            # Nested context
            with request_context("nested-id") as nested_id:
                self.assertEqual(get_request_id(), "nested-id")
                self.assertEqual(nested_id, "nested-id")
            
            # Should restore previous ID
            self.assertEqual(get_request_id(), ctx_id)
        
        # Should restore initial ID
        self.assertEqual(get_request_id(), initial_id)
    
    def test_log_execution_time(self):
        """Test log_execution_time decorator."""
        # Set up logger with a string IO handler
        logger = logging.getLogger("timing_test")
        logger.setLevel(logging.DEBUG)
        log_output = StringIO()
        handler = logging.StreamHandler(log_output)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.propagate = False
        
        # Create a function with the decorator
        @log_execution_time(logger=logger)
        def slow_function():
            import time
            time.sleep(0.1)
            return "result"
        
        # Call the function
        result = slow_function()
        
        # Check result
        self.assertEqual(result, "result")
        
        # Check log output
        log_content = log_output.getvalue()
        self.assertIn("Function slow_function took", log_content)
        self.assertRegex(log_content, r"took \d+\.\d+ seconds")
    
    def test_get_logger(self):
        """Test get_logger function."""
        # Get a logger
        logger = get_logger("test_custom_logger")
        
        # Capture its output
        log_output = StringIO()
        handler = logging.StreamHandler(log_output)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        
        # Log with sensitive information
        logger.info("API key is api_key='secret-key-12345'")
        
        # Check that filters are applied
        log_content = log_output.getvalue()
        self.assertIn("***REDACTED***", log_content)
        self.assertNotIn("secret-key-12345", log_content)


if __name__ == "__main__":
    unittest.main() 