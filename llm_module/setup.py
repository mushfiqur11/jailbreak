from setuptools import setup, find_packages

setup(
    name="llm_module",
    version="1.0.0",
    description="LLM Module for Jailbreak Framework",
    author="Jailbreak Framework Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "openai>=1.3.0",
        "huggingface_hub>=0.19.0",
        "bitsandbytes>=0.41.0",
        "peft>=0.7.0",
        "psutil>=5.9.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
        "scipy>=1.11.0",
        "tokenizers>=0.15.0",
        "safetensors>=0.4.0",
        "pyyaml>=6.0.1",
        "pandas>=2.0.0",
        "jsonlines>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
        ],
        "optional": [
            "nvidia-ml-py3>=7.352.0",  # For GPU monitoring
            "flash-attn>=2.3.0",       # Flash attention (optional)
            "scikit-learn>=1.3.0",     # For evaluation metrics
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    keywords="llm language-model jailbreaking ai-security pytorch transformers",
    project_urls={
        "Source": "https://github.com/jailbreak-framework/llm_module",
        "Documentation": "https://github.com/jailbreak-framework/llm_module/blob/main/README.md",
    },
)
