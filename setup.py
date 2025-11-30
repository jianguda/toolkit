"""Setup script for LM Lens (LM透镜) - LLM Transparency and Repair Tool."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="lm-lens",
    version="1.0.0",
    description="LM Lens (LM透镜) - LM Transparency and Model Repair Visualization Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LM Lens Contributors",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.28.0",
        "streamlit-extras",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "transformer-lens",
        "networkx",
        "plotly",
        "pandas",
        "numpy",
        "datasets",
        "evaluate",
        "einops",
        "fancy-einsum",
        "jaxtyping==0.2.25",
        "tokenizers",
        "pyinstrument",
        # Repair module dependencies
        "loguru>=0.7.0",
        "captum>=0.7.0",
        "scikit-learn>=1.5.0",
        "scipy>=1.14.0",
        "sentence-transformers>=3.4.0",
        "sentencepiece>=0.2.0",
        "tqdm>=4.67.0",
        "nltk>=3.9.0",
        "rouge-score>=0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "lm-lens=lm_lens.server.app:main",
        ],
    },
)

