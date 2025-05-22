from setuptools import setup, find_packages

setup(
    name="forecasting-tools-demo",
    version="0.1.0",
    description="Demo of the Metaculus forecasting-tools package",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "forecasting-tools",
        "python-dotenv",
    ],
    python_requires=">=3.9",
) 