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
    cd ../../
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
        python3 -m venv "$venv_name"
    fi
    
    print_info "Activating virtual environment..."
    source "$venv_name/bin/activate"
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip
    
    print_success "Virtual environment setup complete"
    cd jailbreak-pkg/setup
}

# Function to install PyTorch with appropriate CUDA support
install_pytorch() {
    print_info "Installing PyTorch..."
    
    if check_cuda; then
        # Install PyTorch with CUDA support
        print_info "Installing PyTorch with CUDA support..."
        module unload python
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        print_info "ERROR: No NVIDIA GPU detected or nvidia-smi not available"
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

# Function to install the unified jailbreak package
install_jailbreak_package() {
    print_info "Installing jailbreak framework package in editable mode..."
    
    # Go to the jailbreak root directory (parent of setup)
    cd ..
    
    # Check that setup.py exists
    if [ ! -f "setup.py" ]; then
        print_error "setup.py not found in jailbreak directory"
        print_info "Please ensure the setup.py file exists at the root level"
        exit 1
    fi
    
    # Install in editable mode
    pip install -e .
    
    print_success "jailbreak package installed successfully"
    
    # Return to setup directory
    cd setup
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
    
    # Test jailbreak package imports
    print_info "Testing jailbreak package imports..."
    python3 -c "
try:
    import jailbreak
    from jailbreak.llm_module.models import Llama, GPT, Phi, Qwen, DeepSeek, Aya
    from jailbreak.llm_module.config import ModelConfigs, ConfigManager
    from jailbreak.llm_module.utils import ConversationManager, ModelUtils
    print('✓ jailbreak.llm_module imported successfully')
except Exception as e:
    print(f'✗ jailbreak.llm_module import error: {e}')
    exit(1)
"
    
    # Test agentic_jailbreak subpackage import (may fail - that's ok)
    print_info "Testing jailbreak.agentic_jailbreak import..."
    python3 -c "
try:
    import jailbreak.agentic_jailbreak
    print('✓ jailbreak.agentic_jailbreak imported successfully (placeholder)')
except Exception as e:
    print(f'⚠ jailbreak.agentic_jailbreak import warning: {e}')
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
    
    # Step 5: Install unified jailbreak package
    print_header "Installing Jailbreak Framework Package"
    install_jailbreak_package
    
    # Step 6: Verify installation
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
    print_info "2. Test jailbreak package: python -c 'import jailbreak; print(\"Success!\")'"
    print_info "3. Run examples: python jailbreak/llm_module/examples/basic_usage.py"
    print_info "4. Check documentation in jailbreak/llm_module/README.md"
    print_info ""
    print_success "Happy jailbreaking research! 🚀"
}

# Run main function with all arguments
main "$@"
