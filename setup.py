from setuptools import setup, find_packages

setup(
    name="jailbreak",
    version="1.0.0", 
    description="Jailbreak Framework - Complete toolkit for AI security research and jailbreaking experiments",
    long_description=open("README.md", "r", encoding="utf-8").read() if open("README.md", "r", encoding="utf-8") else "",
    long_description_content_type="text/markdown",
    author="Md Mushfiqur Rahman",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        # Core ML Libraries
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "tokenizers>=0.15.0",
        "safetensors>=0.4.0",
        
        # API Integrations
        "openai>=1.3.0",
        "huggingface_hub>=0.19.0",
        
        # Quantization and Optimization
        "bitsandbytes>=0.41.0",
        "scipy>=1.11.0",
        
        # Fine-tuning (LoRA)
        "peft>=0.7.0",
        
        # System and Utilities
        "psutil>=5.9.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
        
        # Data Processing
        "pandas>=2.0.0",
        "jsonlines>=4.0.0",
        
        # Configuration Management
        "pyyaml>=6.0.1",
        "toml>=0.10.2",

        "typing",
        "omegaconf>=2.3.0",
        "datasets>=2.14.0",
        "loguru>=0.7.0",
        "Pillow>=10.0.0",
    ],
    entry_points={
        "console_scripts": [
            "jailbreak=jailbreak.cli:main",  # Future CLI entry point
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
        "Operating System :: OS Independent",
    ],
    keywords="jailbreaking ai-security llm language-model pytorch transformers agents research",
    project_urls={
        "Homepage": "https://github.com/mushfiqur11/jailbreak",
        "Source": "https://github.com/mushfiqur11/jailbreak",
        "Documentation": "https://github.com/mushfiqur11/jailbreak/blob/main/README.md",
        "Bug Reports": "https://github.com/mushfiqur11/jailbreak/issues",
    },
    include_package_data=True,
    package_data={
        "jailbreak": ["**/*"]
    },
)
