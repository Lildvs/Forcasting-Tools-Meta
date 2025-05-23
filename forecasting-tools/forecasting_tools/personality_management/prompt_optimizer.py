"""
Prompt Generation Pipeline Optimizer

This module provides optimized prompt generation for personalities,
including compression, caching, and streamlined processing.
"""

import logging
import hashlib
import re
from typing import Dict, Any, Optional, List, Tuple

from forecasting_tools.personality_management.config import PersonalityConfig
from forecasting_tools.personality_management.template_manager import TemplateManager
from forecasting_tools.personality_management.cache import PersonalityCache

logger = logging.getLogger(__name__)

class PromptOptimizer:
    """
    Optimizer for prompt generation pipelines.
    
    This class provides efficient prompt generation with:
    - Prompt compression
    - Caching of generated prompts
    - Template fragment reuse
    - Performance optimizations
    """
    
    _instance = None
    _prompt_cache: Dict[str, str] = {}
    _max_cache_size = 100
    
    def __new__(cls):
        """Implement singleton pattern for the optimizer."""
        if cls._instance is None:
            cls._instance = super(PromptOptimizer, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Initialize the prompt optimizer."""
        self._template_manager = TemplateManager()
        self._personality_cache = PersonalityCache()
        logger.debug("Prompt optimizer initialized")
    
    def generate_prompt(
        self,
        prompt_template: str,
        personality_name: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        compress: bool = False,
        use_cache: bool = True
    ) -> str:
        """
        Generate an optimized prompt using a template and personality.
        
        Args:
            prompt_template: Template name or template content
            personality_name: Optional personality to apply
            variables: Optional variables to inject
            compress: Whether to compress the prompt
            use_cache: Whether to use cached prompts
            
        Returns:
            The generated prompt
        """
        variables = variables or {}
        
        # Generate cache key if using cache
        cache_key = None
        if use_cache:
            cache_key = self._generate_cache_key(prompt_template, personality_name, variables, compress)
            if cache_key in self._prompt_cache:
                logger.debug(f"Prompt cache hit for {personality_name or 'default'}")
                return self._prompt_cache[cache_key]
        
        # Load personality if specified
        personality = None
        if personality_name:
            try:
                personality = self._personality_cache.get(personality_name)
                if not personality:
                    # Not in cache, attempt to load it
                    from forecasting_tools.personality_management import PersonalityManager
                    manager = PersonalityManager()
                    personality = manager.load_personality(personality_name)
                    # Add to cache
                    if personality:
                        self._personality_cache.put(personality_name, personality)
            except Exception as e:
                logger.warning(f"Error loading personality {personality_name}: {str(e)}")
        
        # Apply personality variables
        if personality:
            # Add standard personality traits as variables
            variables["thinking_style"] = personality.thinking_style.value
            variables["uncertainty_approach"] = personality.uncertainty_approach.value
            variables["reasoning_depth"] = personality.reasoning_depth.value
            variables["temperature"] = getattr(personality, "temperature", 0.7)
            
            # Add personality template variables
            variables.update(personality.template_variables)
            
            # Add custom traits
            for trait_name, trait in personality.traits.items():
                variables[f"trait_{trait_name}"] = trait.value
        
        # Check if prompt_template is a template name or content
        if "\n" not in prompt_template and len(prompt_template) < 100:
            # Likely a template name
            template_content = self._template_manager.render_template(prompt_template, variables)
            if not template_content:
                logger.warning(f"Template not found: {prompt_template}, treating as raw content")
                template_content = prompt_template
        else:
            # Raw template content
            template_content = prompt_template
        
        # Fill template with variables
        prompt = self._fill_template(template_content, variables)
        
        # Apply compression if requested
        if compress:
            prompt = self._compress_prompt(prompt)
        
        # Cache the result
        if use_cache and cache_key:
            self._cache_prompt(cache_key, prompt)
        
        return prompt
    
    def _generate_cache_key(
        self,
        prompt_template: str,
        personality_name: Optional[str],
        variables: Dict[str, Any],
        compress: bool
    ) -> str:
        """
        Generate a cache key for prompt caching.
        
        Args:
            prompt_template: Template name or content
            personality_name: Personality name
            variables: Variables dictionary
            compress: Whether compression was applied
            
        Returns:
            Cache key string
        """
        # Create a deterministic representation of inputs
        key_parts = [
            prompt_template,
            personality_name or "no_personality",
            str(sorted([(k, str(v)) for k, v in variables.items()])),
            "compressed" if compress else "not_compressed"
        ]
        
        # Generate hash
        combined = "||".join(key_parts)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _cache_prompt(self, cache_key: str, prompt: str) -> None:
        """
        Cache a generated prompt.
        
        Args:
            cache_key: The cache key
            prompt: The prompt to cache
        """
        # Implement LRU-like behavior by removing oldest entry if at capacity
        if len(self._prompt_cache) >= self._max_cache_size:
            # Remove first item (oldest)
            oldest_key = next(iter(self._prompt_cache))
            del self._prompt_cache[oldest_key]
        
        # Add to cache
        self._prompt_cache[cache_key] = prompt
    
    def _fill_template(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Fill a template with variables.
        
        Args:
            template: Template string
            variables: Variables to inject
            
        Returns:
            Filled template
        """
        result = template
        
        # Process conditional sections first
        result = self._process_conditionals(result, variables)
        
        # Replace variable placeholders
        for var_name, var_value in variables.items():
            placeholder = "{{" + var_name + "}}"
            result = result.replace(placeholder, str(var_value))
        
        return result
    
    def _process_conditionals(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Process conditional sections in a template.
        
        Args:
            template: Template string
            variables: Variables dictionary
            
        Returns:
            Processed template
        """
        # Process IF conditions
        if_pattern = r'<!-- IF ([\w\.]+) == ([\w\.]+) -->(.*?)<!-- ENDIF -->'
        
        def replace_conditional(match):
            var_name = match.group(1)
            expected_value = match.group(2)
            content = match.group(3)
            
            # Get actual value
            actual_value = str(variables.get(var_name, ""))
            
            # Compare and return appropriate content
            if actual_value.lower() == expected_value.lower():
                return content
            return ""
        
        # Apply replacements
        return re.sub(if_pattern, replace_conditional, template, flags=re.DOTALL)
    
    def _compress_prompt(self, prompt: str) -> str:
        """
        Compress a prompt to reduce token usage.
        
        Args:
            prompt: The prompt to compress
            
        Returns:
            Compressed prompt
        """
        # Remove unnecessary whitespace
        result = re.sub(r'\s+', ' ', prompt)
        
        # Remove comments
        result = re.sub(r'<!--.*?-->', '', result)
        
        # Remove duplicate newlines
        result = re.sub(r'\n\s*\n', '\n', result)
        
        # Reduce bullet point indentation
        result = re.sub(r'  +- ', '- ', result)
        
        return result.strip()
    
    def optimize_prompt_pipeline(
        self,
        personality_name: str,
        template_name: str,
        variables: Dict[str, Any],
        context_size: Optional[int] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Optimize the full prompt generation pipeline.
        
        Args:
            personality_name: Name of the personality to use
            template_name: Name of the template to use
            variables: Variables to inject
            context_size: Optional maximum context size
            
        Returns:
            Tuple of (optimized prompt, metadata)
        """
        # Generate initial prompt
        prompt = self.generate_prompt(template_name, personality_name, variables)
        
        # Apply additional optimizations if context size is specified
        if context_size and len(prompt) > context_size:
            logger.warning(f"Prompt exceeds context size ({len(prompt)} > {context_size}), applying compression")
            prompt = self._compress_prompt(prompt)
        
        # Calculate token estimation (rough approximation)
        estimated_tokens = len(prompt.split())
        
        # Generate metadata
        metadata = {
            "personality": personality_name,
            "template": template_name,
            "estimated_tokens": estimated_tokens,
            "compressed": context_size is not None,
            "variables_used": list(variables.keys())
        }
        
        return prompt, metadata
    
    def set_max_cache_size(self, size: int) -> None:
        """
        Set the maximum prompt cache size.
        
        Args:
            size: Maximum number of prompts to cache
        """
        self._max_cache_size = max(10, size)
        
        # If current cache is larger than new size, trim it
        if len(self._prompt_cache) > self._max_cache_size:
            # Keep only the most recent entries (last N items)
            items = list(self._prompt_cache.items())
            self._prompt_cache = dict(items[-self._max_cache_size:])
    
    def clear_cache(self) -> None:
        """Clear the prompt cache."""
        self._prompt_cache.clear()
        logger.debug("Prompt cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        return {
            "size": len(self._prompt_cache),
            "max_size": self._max_cache_size
        } 