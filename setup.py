"""
MINDI 1.5 Vision-Coder — Package Setup

Allows installing the project as a Python package:
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="mindi-vision-coder",
    version="1.5.0",
    author="Faaz",
    author_email="faaz@mindigenous.ai",
    description="Multimodal agentic AI code generator by MINDIGENOUS.AI",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://huggingface.co/Mindigenous/MINDI-1.5-Vision-Coder",
    packages=find_packages(),
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
