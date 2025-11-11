# Jailbreak Framework Setup Guide

This directory contains setup files for the complete jailbreak framework, including both the llm_module and agentic_jailbreak packages.

## Files

- **`requirements.txt`**: Complete list of Python dependencies
- **`setup.sh`**: Automated setup script for both packages (Linux/Mac)
- **`README.md`**: This setup guide

## Quick Setup

### Option 1: Automated Setup (Linux/Mac)

```bash
cd setup
chmod +x setup.sh
./setup.sh
```

**Script Options:**
- `--skip-venv`: Skip virtual environment creation
- `--skip-optional`: Skip optional dependencies
- `--skip-tests`: Skip functionality tests
- `--help`: Show help message

### Option 2: Manual Setup (All Platforms)

#### Step 1: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv jailbreak_env

# On Linux/Mac:
source jailbreak_env/bin/activate

# On Windows:
jailbreak_env\Scripts\activate
```

#### Step 2: Install PyTorch

**For GPU (CUDA) support:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU-only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Step 3: Install Requirements

```bash
pip install -r requirements.txt
```

#### Step 4: Install Packages

**Install llm_module:**
```bash
cd ../llm_module
pip install -e .
cd ../setup
```

**Install agentic_jailbreak (placeholder):**
```bash
cd ../agentic_jailbreak
pip install -e .
cd ../setup
```

#### Step 5: Setup API Keys

Create a `tokens` directory in the parent folder:

```bash
mkdir ../tokens
```

Add your API keys:
- **HuggingFace**: Save token in `../tokens/hf_token.txt`
- **OpenAI**: Save API key in `../tokens/openai_key.txt`

Or set environment variables:
```bash
export HUGGINGFACE_HUB_TOKEN='your_token_here'
export OPENAI_API_KEY='your_key_here'
```

#### Step 6: Verify Installation

```bash
cd ..
python -c "import llm_module; print('llm_module: Success!')"
python -c "import agentic_jailbreak; print('agentic_jailbreak: Success!')"
python llm_module/examples/basic_usage.py
```

## What Gets Installed

### llm_module (Complete)
A comprehensive language model management system with:
- **Unified API** for HuggingFace and OpenAI models
- **Model Support**: Llama, GPT, Phi, Qwen, DeepSeek, Aya
- **Quantization** for memory efficiency
- **Conversation Management** with analytics
- **Configuration System** with presets
- **Future LoRA** fine-tuning framework

### agentic_jailbreak (Placeholder)
Currently under development:
- **Placeholder Package** that depends on llm_module
- **Agent Framework** (agents/ directory exists but needs integration)
- **Research Methods** (methods/ directory for future algorithms)

## Dependencies Overview

### Core Dependencies
- **PyTorch** (≥2.0.0): Deep learning framework
- **Transformers** (≥4.35.0): HuggingFace model library
- **OpenAI** (≥1.3.0): OpenAI API client
- **Accelerate** (≥0.24.0): Model acceleration library

### Quantization & Performance
- **bitsandbytes** (≥0.41.0): 4-bit/8-bit quantization
- **PEFT** (≥0.7.0): LoRA fine-tuning support
- **flash-attn** (≥2.3.0): Optimized attention (optional, GPU only)

### Utilities
- **psutil**: System monitoring
- **pandas**: Data processing
- **jsonlines**: Training data export
- **pyyaml**: Configuration management

## Troubleshooting

### Common Issues

1. **CUDA not available**: 
   - Install CPU version of PyTorch
   - Check GPU drivers and CUDA installation

2. **Module not found errors**:
   - Ensure packages installed with `-e` flag
   - Check virtual environment is activated

3. **API key errors**:
   - Verify token files exist and contain valid keys
   - Check file permissions

4. **Memory issues**:
   - Use quantization for large models
   - Consider smaller model variants

### Installation Verification

Test individual components:
```bash
# Test core dependencies
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Test llm_module
python -c "from llm_module.models import Llama, GPT; print('Models imported')"
python -c "from llm_module.config import ModelConfigs; print('Config imported')"

# Test imports work
python -c "import llm_module; import agentic_jailbreak; print('All packages imported')"
```

## Development Notes

### llm_module Structure
```
llm_module/
├── base/          # Abstract base classes
├── models/        # Model implementations
├── config/        # Configuration management
├── utils/         # Utilities and helpers
├── quantization/  # Memory optimization
├── finetuning/    # LoRA framework
└── examples/      # Usage examples
```

### agentic_jailbreak Structure
```
agentic_jailbreak/
├── agents/        # Existing agent implementations
├── methods/       # Future jailbreaking algorithms
└── scripts/       # Utility scripts
```

## Usage Examples

### Quick llm_module Test
```python
from llm_module.models import Llama
from llm_module.config import ModelConfigs

# Get a pre-configured setup
config = ModelConfigs.get_config("llama-3b")

# Initialize model (will show placeholder behavior without actual model)
llm = Llama(config)
llm.set_system_prompt("You are a helpful assistant.")
response = llm.get_system_prompt()
print(f"System prompt: {response}")
```

### Package Integration
```python
import llm_module
import agentic_jailbreak

print("Both packages loaded successfully!")
print(f"llm_module version: {llm_module.__version__}")
print(f"agentic_jailbreak version: {agentic_jailbreak.__version__}")
```

## Next Steps

1. **Complete agentic_jailbreak**: Integrate existing agents with llm_module
2. **Add Methods**: Implement jailbreaking algorithms in methods/ directory
3. **Create Examples**: Add comprehensive usage examples
4. **Documentation**: Expand documentation for research workflows

Happy jailbreaking research! 🚀
