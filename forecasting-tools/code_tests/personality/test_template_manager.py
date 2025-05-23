"""
Unit tests for the template manager module.
"""

import unittest
import os
import tempfile
import json
import shutil
from typing import Dict, Any

from forecasting_tools.personality_management.template_manager import TemplateManager


class TestTemplateManager(unittest.TestCase):
    """Test cases for TemplateManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test templates
        self.test_dir = tempfile.mkdtemp()
        
        # Create some test templates
        self.test_templates = {
            "test_template1": {
                "content": "This is template 1 with {{variable1}}",
                "variables": {
                    "variable1": "default value"
                }
            },
            "test_template2": {
                "content": "This is template 2 with {{variable2}}",
                "variables": {
                    "variable2": "another value"
                }
            },
            "conditional_template": {
                "content": "Start <!-- IF condition == true -->show this<!-- ENDIF --> end",
                "variables": {
                    "condition": "true"
                }
            }
        }
        
        # Write test templates to files
        for name, content in self.test_templates.items():
            with open(os.path.join(self.test_dir, f"{name}.json"), "w") as f:
                json.dump(content, f)
        
        # Create template manager instance and add test directory
        self.template_manager = TemplateManager()
        self.template_manager.add_template_directory(self.test_dir)
        self.template_manager.invalidate_all_templates()  # Clear any existing templates

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_discover_templates(self):
        """Test discovering templates in a directory."""
        templates = self.template_manager.discover_templates()
        
        # Check if all test templates are discovered
        for name in self.test_templates.keys():
            self.assertIn(name, templates)

    def test_get_template(self):
        """Test getting a template."""
        # Test getting a template
        template = self.template_manager.get_template("test_template1")
        
        self.assertIsNotNone(template)
        self.assertEqual(template["content"], "This is template 1 with {{variable1}}")
        self.assertEqual(template["variables"]["variable1"], "default value")

    def test_get_nonexistent_template(self):
        """Test getting a non-existent template."""
        template = self.template_manager.get_template("nonexistent_template")
        self.assertIsNone(template)

    def test_force_reload(self):
        """Test forcing a template reload."""
        # First load the template
        template = self.template_manager.get_template("test_template1")
        self.assertIsNotNone(template)
        
        # Modify the template file
        modified_template = {
            "content": "This is modified template 1 with {{variable1}}",
            "variables": {
                "variable1": "modified value"
            }
        }
        
        with open(os.path.join(self.test_dir, "test_template1.json"), "w") as f:
            json.dump(modified_template, f)
        
        # Without force_reload, should get cached version
        template = self.template_manager.get_template("test_template1")
        self.assertEqual(template["content"], "This is template 1 with {{variable1}}")
        
        # With force_reload, should get updated version
        template = self.template_manager.get_template("test_template1", force_reload=True)
        self.assertEqual(template["content"], "This is modified template 1 with {{variable1}}")

    def test_get_template_field(self):
        """Test getting a specific field from a template."""
        content = self.template_manager.get_template_field("test_template1", "content")
        self.assertEqual(content, "This is template 1 with {{variable1}}")
        
        variables = self.template_manager.get_template_field("test_template1", "variables")
        self.assertEqual(variables["variable1"], "default value")
        
        # Test nonexistent field
        nonexistent = self.template_manager.get_template_field("test_template1", "nonexistent")
        self.assertIsNone(nonexistent)

    def test_combine_templates(self):
        """Test combining multiple templates."""
        # Create additional test templates
        extension_template = {
            "content": "Extended content",
            "variables": {
                "new_var": "new value",
                "variable1": "overridden value"  # This should override base template
            },
            "additional_field": "additional value"
        }
        
        with open(os.path.join(self.test_dir, "extension_template.json"), "w") as f:
            json.dump(extension_template, f)
        
        # Combine templates
        combined = self.template_manager.combine_templates(
            "test_template1", 
            ["extension_template"]
        )
        
        # Check that base fields are preserved
        self.assertEqual(combined["content"], "This is template 1 with {{variable1}}")
        
        # Check that extension fields are added
        self.assertEqual(combined["additional_field"], "additional value")
        
        # Check that variables are merged with extension taking precedence
        self.assertEqual(combined["variables"]["variable1"], "overridden value")
        self.assertEqual(combined["variables"]["new_var"], "new value")

    def test_invalidate_template(self):
        """Test invalidating a template in the cache."""
        # First load the template
        template = self.template_manager.get_template("test_template1")
        self.assertIsNotNone(template)
        
        # Invalidate the template
        self.template_manager.invalidate_template("test_template1")
        
        # Modify the template file
        modified_template = {
            "content": "This is modified template 1 with {{variable1}}",
            "variables": {
                "variable1": "modified value"
            }
        }
        
        with open(os.path.join(self.test_dir, "test_template1.json"), "w") as f:
            json.dump(modified_template, f)
        
        # Load the template again, should get updated version
        template = self.template_manager.get_template("test_template1")
        self.assertEqual(template["content"], "This is modified template 1 with {{variable1}}")

    def test_get_template_variables(self):
        """Test extracting variables from a template."""
        variables = self.template_manager.get_template_variables("test_template1")
        self.assertEqual(variables["variable1"], "default value")
        
        # Test template without variables
        template_without_vars = {
            "content": "Template without variables"
        }
        
        with open(os.path.join(self.test_dir, "no_vars_template.json"), "w") as f:
            json.dump(template_without_vars, f)
        
        variables = self.template_manager.get_template_variables("no_vars_template")
        self.assertEqual(variables, {})

    def test_render_template(self):
        """Test rendering a template with variables."""
        rendered = self.template_manager.render_template("test_template1", {"variable1": "custom value"})
        self.assertEqual(rendered, "This is template 1 with custom value")
        
        # Test with multiple variables
        with open(os.path.join(self.test_dir, "multi_var_template.json"), "w") as f:
            json.dump({
                "content": "Template with {{var1}} and {{var2}} and {{var3}}",
                "variables": {}
            }, f)
        
        rendered = self.template_manager.render_template(
            "multi_var_template", 
            {"var1": "first", "var2": "second", "var3": "third"}
        )
        self.assertEqual(rendered, "Template with first and second and third")

    def test_render_nonexistent_template(self):
        """Test rendering a non-existent template."""
        rendered = self.template_manager.render_template("nonexistent_template", {})
        self.assertIsNone(rendered)

    def test_render_template_without_content(self):
        """Test rendering a template without content field."""
        with open(os.path.join(self.test_dir, "no_content_template.json"), "w") as f:
            json.dump({
                "variables": {"test": "value"}
            }, f)
        
        rendered = self.template_manager.render_template("no_content_template", {})
        self.assertIsNone(rendered)

    def test_add_template_directory(self):
        """Test adding a template directory."""
        # Create a new temporary directory
        new_dir = tempfile.mkdtemp()
        
        try:
            # Create a new template in the new directory
            with open(os.path.join(new_dir, "new_template.json"), "w") as f:
                json.dump({
                    "content": "New template content",
                    "variables": {}
                }, f)
            
            # Add the new directory
            self.template_manager.add_template_directory(new_dir)
            
            # Try to get the new template
            template = self.template_manager.get_template("new_template")
            self.assertIsNotNone(template)
            self.assertEqual(template["content"], "New template content")
        finally:
            # Clean up
            shutil.rmtree(new_dir)


if __name__ == "__main__":
    unittest.main() 