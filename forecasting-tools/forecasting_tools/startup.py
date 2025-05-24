"""
Startup module to apply patches and fixes on application startup.
"""

def apply_startup_fixes():
    """Apply all necessary fixes and patches on startup."""
    try:
        # Apply compatibility patches
        from forecasting_tools.compat.compatibility import BinaryForecastCompat
        BinaryForecastCompat.apply_patches()
        
        # Fix any other startup issues here
        
        return True
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error applying startup fixes: {e}")
        return False

# Run startup fixes
apply_startup_fixes()
