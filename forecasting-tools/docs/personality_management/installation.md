# Installing the Personality Management System

This guide provides instructions for installing and configuring the personality management system.

## Prerequisites

- Python 3.10 or higher
- pip or poetry for package management

## Installation Methods

### Method 1: Installing with pip

Install the full forecasting-tools package with personality management support:

```bash
# Install the base package with personality management extras
pip install forecasting-tools[personality]
```

For development or to access all features:

```bash
# Install with all extras
pip install forecasting-tools[full]
```

### Method 2: Installing from Source

Clone the repository and install from source:

```bash
# Clone the repository
git clone https://github.com/Metaculus/forecasting-tools.git
cd forecasting-tools

# Install with personality extras
pip install -e .[personality]
```

## Verifying Installation

You can verify that the personality management system is correctly installed by running:

```bash
# Run the health check utility
personality-health-check --level minimal
```

If the installation is successful, you should see a summary of the personality system health status.

## Configuration

### Environment Variables

The personality system can be configured using environment variables:

```bash
# Enable/disable the entire system
export FORECASTING_PERSONALITY_PERSONALITY_SYSTEM_ENABLED=true

# Control caching behavior
export FORECASTING_PERSONALITY_USE_CACHING=true

# Enable debug mode for detailed logging
export FORECASTING_PERSONALITY_DEBUG_MODE=false

# Set rollout percentage for gradual deployment
export FORECASTING_PERSONALITY_ROLLOUT_PERCENTAGE=100
```

### Configuration File

Alternatively, you can create a configuration file at `~/.forecasting-tools/personality_flags.json`:

```json
{
  "feature_flags": {
    "personality_system_enabled": true,
    "use_caching": true,
    "debug_mode": false,
    "rollout_percentage": 100
  }
}
```

## Directory Structure

The personality management system looks for personalities and templates in these directories (in order of precedence):

1. **Project-specific**: 
   - `./personalities/`
   - `./templates/`

2. **User-specific**:
   - `~/.forecasting-tools/personalities/`
   - `~/.forecasting-tools/templates/`

3. **System-wide**:
   - `/etc/forecasting-tools/personalities/`
   - `/etc/forecasting-tools/templates/`

You can create these directories and add custom personalities/templates as needed:

```bash
# Create user-specific directories
mkdir -p ~/.forecasting-tools/personalities
mkdir -p ~/.forecasting-tools/templates
```

## Migrating Existing Forecasts

If you have existing forecasts and want to migrate them to use the personality system:

```bash
# Run the migration script
personality-migration --input-dir=/path/to/forecasts --dry-run
```

The `--dry-run` flag allows you to see what changes would be made without actually modifying files. Remove this flag to apply the changes.

## Troubleshooting

### Common Issues

1. **Missing dependencies**:
   
   If you encounter import errors, ensure you installed with personality extras:
   ```bash
   pip install forecasting-tools[personality]
   ```

2. **Personalities not found**:
   
   Check that the personality directories exist and contain properly formatted JSON files:
   ```bash
   # List built-in personalities
   python -c "from forecasting_tools.personality_management import PersonalityManager; print(PersonalityManager().list_available_personalities())"
   ```

3. **Feature flags not working**:
   
   Verify your environment variables or configuration file:
   ```bash
   # Check current feature flags
   python -c "from forecasting_tools.personality_management.feature_flags import get_feature_flags; print(get_feature_flags().get_all_flags())"
   ```

### Getting Help

For more detailed diagnostics, run the comprehensive health check:

```bash
personality-health-check --level comprehensive --output health_report.json --verbose
```

This will generate a detailed report of your system's status that can help identify issues.

## Next Steps

Once installed, you can proceed to:
- [Personality System Guide](../personality_system.md): Overview of the system
- [Personality Customization Tutorial](../tutorials/personality_customization.md): Learn to create custom personalities
- [API Reference](../api/personality_api.md): Detailed API documentation 