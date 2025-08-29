"""Setup script for Legal Summarization package."""

from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="legal-summarization",
    version="1.0.0",
    description="Advanced agentic workflow for legal document summarization",
    long_description=open("legal_summarization/README.md").read(),
    long_description_content_type="text/markdown",
    author="Legal AI Team",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Legal Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="legal, summarization, nlp, langchain, langgraph, gemini",
)