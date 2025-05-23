#!/usr/bin/env python

from setuptools import setup, find_packages

# Read requirements
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Read long description
with open("README.md") as f:
    long_description = f.read()

setup(
    name="forecasting-tools",
    version="0.2.39",
    description="AI forecasting and research tools to help humans reason about and forecast the future",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Benjamin Wilson",
    author_email="mokoresearch@gmail.com",
    url="https://github.com/Metaculus/forecasting-tools",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black",
            "flake8",
            "isort",
            "mypy",
            "pytest",
            "pytest-cov",
        ],
        "personality": [
            "pydantic>=2.0.0",
            "jinja2>=3.1.2",
            "sqlitedict>=2.0.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "matplotlib>=3.7.0",
            "pytest-benchmark>=4.0.0"
        ],
        "full": [
            "black",
            "flake8",
            "isort",
            "mypy",
            "pytest",
            "pytest-cov",
            "pydantic>=2.0.0",
            "jinja2>=3.1.2",
            "sqlitedict>=2.0.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "matplotlib>=3.7.0",
            "pytest-benchmark>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "forecasting-tools=forecasting_tools.cli:main",
            "personality-health-check=forecasting_tools.personality_management.health_check:main",
            "personality-migration=forecasting_tools.scripts.personality_migration:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "ai",
        "artificial intelligence",
        "forecasting",
        "research",
        "metaculus",
        "prediction",
        "future",
        "market",
        "personality",
    ],
) 