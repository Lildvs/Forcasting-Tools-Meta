"""
Compatibility layer for handling API changes between versions.
"""

class BinaryForecastCompat:
    """
    Monkey patch to add backward compatibility to BinaryForecast class.
    """
    
    @classmethod
    def apply_patches(cls):
        """Apply all compatibility patches."""
        cls.patch_binary_forecast()
    
    @classmethod
    def patch_binary_forecast(cls):
        """
        Add compatibility for BinaryForecast objects.
        Adds prediction property that returns probability.
        """
        try:
            from forecasting_tools.data_models.binary_report import BinaryForecast
            
            # Only add the property if it doesn't already exist
            if not hasattr(BinaryForecast, "prediction"):
                # Add prediction property that returns probability
                BinaryForecast.prediction = property(lambda self: self.probability)
                print("  Successfully added prediction property to BinaryForecast")
            else:
                print("  BinaryForecast already has prediction property")
                
        except ImportError as e:
            print(f"  Error importing BinaryForecast: {e}")
            return False
        except Exception as e:
            print(f"  Error patching BinaryForecast: {e}")
            return False
        
        return True

# Create empty __init__.py to make it a proper package
init_file = __file__.replace("compatibility.py", "__init__.py")
if not os.path.exists(init_file):
    with open(init_file, "w") as f:
        f.write("# Compatibility layer for handling API changes between versions.")
