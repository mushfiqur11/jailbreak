from setuptools import setup, find_packages

setup(
    name="agentic_jailbreak",
    version="0.1.0",
    description="Agentic Jailbreak Framework (Under Development)",
    author="Jailbreak Framework Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "llm_module>=1.0.0",
        # TODO: Add specific dependencies when framework is complete
        # Placeholder dependencies based on existing agents
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
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
    keywords="jailbreaking ai-security agents research",
    project_urls={
        "Source": "https://github.com/jailbreak-framework/agentic_jailbreak",
        "Documentation": "https://github.com/jailbreak-framework/agentic_jailbreak/blob/main/README.md",
    },
)
