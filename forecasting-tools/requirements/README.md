# Dependency Management

This directory contains environment-specific requirements files generated from Poetry.

## File Structure

- `requirements-dev.txt`: Development environment requirements
- `requirements-prod.txt`: Production environment requirements with exact version pinning
- `requirements-streamlit.txt`: Streamlit Cloud deployment requirements
- `requirements-monitoring.txt`: Monitoring-specific requirements

## Generation

These files are generated from Poetry's lock file using the script:

```bash
python scripts/generate_requirements.py
```

## Usage

### Development Environment

```bash
pip install -r requirements/requirements-dev.txt
```

### Production Environment

```bash
pip install -r requirements/requirements-prod.txt
```

### Streamlit Cloud

For Streamlit Cloud deployment, use the requirements-streamlit.txt file in your deployment settings.

## Version Pinning Strategy

We use the following versioning strategy:

1. In `pyproject.toml`: Use caret (`^`) version constraints for flexibility during development
2. In generated requirements files: Use exact (`==`) version pinning for reproducibility in deployments

## Dependencies Management Workflow

1. Add new dependencies to `pyproject.toml` using Poetry:
   ```bash
   poetry add package_name
   ```

2. For dev dependencies:
   ```bash
   poetry add --group dev package_name
   ```

3. Regenerate requirements files:
   ```bash
   python scripts/generate_requirements.py
   ```

4. Commit both `pyproject.toml`, `poetry.lock`, and the generated requirements files 