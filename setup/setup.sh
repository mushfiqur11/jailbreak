#!/bin/bash

# Jailbreak Framework Setup Script
# This script sets up both llm_module and agentic_jailbreak packages

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print formatted messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}\n"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check CUDA availability
check_cuda() {
    if command_exists nvidia-smi; then
        print_info "NVIDIA GPU detected"
        nvidia-smi
        return 0
    else
        print_warning "No NVIDIA GPU detected or nvidia-smi not available"
        return 1
    fi
}

# Function to check Python version
check_python() {
    module load python
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2)
        print_info "Python version: $PYTHON_VERSION"
        
        # Check if Python version is 3.9 or higher
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
            print_success "Python version is compatible"
            return 0
        else
            print_error "Python 3.8 or higher is required"
            return 1
        fi
    else
        print_error "Python3 is not installed"
        return 1
    fi
}

# # Function to create and activate virtual environment
setup_venv() {
    local venv_name="venv_jailbreak"
    
    if [ -d "$venv_name" ]; then
        print_info "Virtual environment '$venv_name' already exists"
        read -p "Do you want to remove it and create a new one? (y/N): " -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing virtual environment..."
            rm -rf "$venv_name"
        else
            print_info "Using existing virtual environment"
        fi
    fi
    
    if [ ! -d "$venv_name" ]; then
        print_info "Creating virtual environment: $venv_name"
        cd ../../
        python3 -m venv "$venv_name"
    fi
    
    print_info "Activating virtual environment..."
    source "$venv_name/bin/activate"
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip
    
    print_success "Virtual environment setup complete"
}

# Function to install PyTorch with appropriate CUDA support
install_pytorch() {
    print_info "Installing PyTorch..."
    
    if check_cuda; then
        # Install PyTorch with CUDA support
        print_info "Installing PyTorch with CUDA support..."
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    else
        # Install CPU-only PyTorch
        print_info "Installing PyTorch CPU-only version..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    print_success "PyTorch installation complete"
}

# Function to install requirements
install_requirements() {
    local requirements_file="requirements.txt"
    
    if [ ! -f "$requirements_file" ]; then
        print_error "Requirements file not found: $requirements_file"
        print_info "Make sure you're running this script from the setup directory"
        exit 1
    fi
    
    print_info "Installing requirements from $requirements_file..."
    
    # Install requirements, but skip torch since we handle it separately
    grep -v "^torch" "$requirements_file" | pip install -r /dev/stdin
    
    print_success "Requirements installation complete"
}

# Function to create setup.py for llm_module if it doesn't exist
create_llm_module_setup() {
    local setup_file="../llm_module/setup.py"
    
    if [ ! -f "$setup_file" ]; then
        print_info "Creating setup.py for llm_module..."
        
        cat > "$setup_file" << 'EOF'
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
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
EOF
        
        print_success "setup.py created for llm_module"
    else
        print_info "setup.py already exists for llm_module"
    fi
}

# Function to install llm_module package
install_llm_module() {
    print_info "Installing llm_module package in editable mode..."
    
    # Go to the llm_module directory
    cd ../llm_module
    
    # Install in editable mode
    pip install -e .
    
    print_success "llm_module package installed successfully"
    
    # Return to setup directory
    cd ../setup
}

# Function to create placeholder setup.py for agentic_jailbreak
create_agentic_jailbreak_placeholder() {
    local setup_file="../agentic_jailbreak/setup.py"
    local init_file="../agentic_jailbreak/__init__.py"
    
    # Create __init__.py if it doesn't exist
    if [ ! -f "$init_file" ]; then
        print_info "Creating __init__.py for agentic_jailbreak..."
        cat > "$init_file" << 'EOF'
"""
Agentic Jailbreak Framework

A framework for conducting automated jailbreaking research on language models.
Currently under development.
"""

__version__ = "0.1.0"
__author__ = "Jailbreak Framework Team"

# TODO: Add main framework components when ready
print("Agentic Jailbreak Framework - Under Development")
EOF
    fi
    
    if [ ! -f "$setup_file" ]; then
        print_info "Creating placeholder setup.py for agentic_jailbreak..."
        
        cat > "$setup_file" << 'EOF'
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
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
EOF
        
        print_success "Placeholder setup.py created for agentic_jailbreak"
    else
        print_info "setup.py already exists for agentic_jailbreak"
    fi
}

# Function to install agentic_jailbreak package (placeholder)
install_agentic_jailbreak() {
    print_warning "Installing agentic_jailbreak package (placeholder - under development)..."
    
    # Go to the agentic_jailbreak directory
    cd ../agentic_jailbreak
    
    # Install in editable mode
    pip install -e . || {
        print_warning "agentic_jailbreak installation failed (expected - under development)"
        print_info "Continuing with setup..."
    }
    
    print_info "agentic_jailbreak placeholder installed"
    
    # Return to setup directory
    cd ../setup
}

# Function to verify installation
verify_installation() {
    print_info "Verifying installation..."
    
    # Test core dependencies
    python3 -c "
import torch
import transformers
import openai
import numpy as np
print('✓ All core dependencies imported successfully')

# Check CUDA availability
if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'✓ CUDA version: {torch.version.cuda}')
else:
    print('✓ CPU-only mode (CUDA not available)')

print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ Transformers version: {transformers.__version__}')
"
    
    # Test llm_module import
    print_info "Testing llm_module imports..."
    python3 -c "
try:
    import llm_module
    from llm_module.models import Llama, GPT, Phi, Qwen, DeepSeek, Aya
    from llm_module.config import ModelConfigs, ConfigManager
    from llm_module.utils import ConversationManager, ModelUtils
    print('✓ llm_module imported successfully')
except Exception as e:
    print(f'✗ llm_module import error: {e}')
    exit(1)
"
    
    # Test agentic_jailbreak import (may fail - that's ok)
    print_info "Testing agentic_jailbreak import..."
    python3 -c "
try:
    import agentic_jailbreak
    print('✓ agentic_jailbreak imported successfully (placeholder)')
except Exception as e:
    print(f'⚠ agentic_jailbreak import warning: {e}')
    print('  This is expected - agentic_jailbreak is under development')
"
    
    if [ $? -eq 0 ]; then
        print_success "Installation verification completed!"
    else
        print_warning "Installation verification completed with warnings"
    fi
}

# Function to setup API keys
setup_api_keys() {
    print_info "Setting up API key directories..."
    
    # Create tokens directory if it doesn't exist
    mkdir -p ../tokens
    
    print_info "API key setup:"
    print_info "1. For HuggingFace: Place your token in '../tokens/hf_token.txt'"
    print_info "2. For OpenAI: Place your API key in '../tokens/openai_key.txt'"
    print_info "3. You can also set environment variables:"
    print_info "   export HUGGINGFACE_HUB_TOKEN='your_token_here'"
    print_info "   export OPENAI_API_KEY='your_key_here'"
    
    print_warning "Remember to keep your API keys secure and never commit them to git!"
}

# Function to run tests
run_tests() {
    print_info "Running basic functionality tests..."
    
    # Test llm_module examples
    if [ -f "../llm_module/examples/basic_usage.py" ]; then
        print_info "Running llm_module examples..."
        cd ..
        python llm_module/examples/basic_usage.py || {
            print_warning "Examples ran with some expected failures (no API keys/models)"
        }
        cd setup
    else
        print_warning "llm_module examples not found"
    fi
}

# Main installation function
main() {
    print_header "Jailbreak Framework Setup"
    
    # Check if we're in the right directory
    if [ ! -f "requirements.txt" ]; then
        print_error "Please run this script from the setup directory"
        exit 1
    fi
    
    # Parse command line arguments
    SKIP_VENV=false
    SKIP_OPTIONAL=false
    SKIP_TESTS=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-venv)
                SKIP_VENV=true
                shift
                ;;
            --skip-optional)
                SKIP_OPTIONAL=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --skip-venv     Skip virtual environment creation"
                echo "  --skip-optional Skip optional dependencies"
                echo "  --skip-tests    Skip functionality tests"
                echo "  --help, -h      Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Step 1: Check system requirements
    print_header "Checking System Requirements"
    check_python || exit 1
    check_cuda
    
    # Step 2: Setup virtual environment (optional)
    if [ "$SKIP_VENV" = false ]; then
        print_header "Setting up Virtual Environment"
        setup_venv
    else
        print_info "Skipping virtual environment setup"
    fi
    
    # Step 3: Install PyTorch
    print_header "Installing PyTorch"
    install_pytorch
    
    # Step 4: Install requirements
    print_header "Installing Requirements"
    install_requirements
    
    # Step 5: Setup and install llm_module
    print_header "Installing LLM Module Package"
    create_llm_module_setup
    install_llm_module
    
    # Step 6: Setup agentic_jailbreak (placeholder)
    print_header "Setting up Agentic Jailbreak (Placeholder)"
    create_agentic_jailbreak_placeholder
    install_agentic_jailbreak
    
    # Step 7: Verify installation
    print_header "Verifying Installation"
    verify_installation
    
    # Step 8: Setup API keys
    print_header "API Key Setup"
    setup_api_keys
    
    # Step 9: Run tests
    if [ "$SKIP_TESTS" = false ]; then
        print_header "Running Tests"
        run_tests
    else
        print_info "Skipping tests"
    fi
    
    # Final success message
    print_header "Setup Complete!"
    print_success "Jailbreak Framework has been successfully set up!"
    print_info ""
    print_info "Installed packages:"
    print_info "✓ llm_module - Complete LLM management system"
    print_info "⚠ agentic_jailbreak - Placeholder (under development)"
    print_info ""
    print_info "Next steps:"
    print_info "1. Add your API keys to the tokens directory"
    print_info "2. Test llm_module: python -c 'import llm_module; print(\"Success!\")"
    print_info "3. Run examples: python llm_module/examples/basic_usage.py"
    print_info "4. Check documentation in llm_module/README.md"
    print_info ""
    print_success "Happy jailbreaking research! 🚀"
}

# Run main function with all arguments
main "$@"
