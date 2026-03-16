# LLM Module for Jailbreaking Research

A comprehensive language model management system designed specifically for AI security research and jailbreaking experiments.

## Overview

The `llm_module` provides a unified interface for working with both HuggingFace models (local) and OpenAI API models, with specialized features for jailbreaking research including conversation management, memory optimization, and future LoRA fine-tuning capabilities.

## Quick Start

### Installation

```python
# The module uses the existing project structure
# Ensure you have the required dependencies:
# - transformers
# - torch  
# - openai
# - bitsandbytes (for quantization)
```

### Basic Usage

```python
from llm_module.models import Llama, GPT, Phi, Qwen, DeepSeek, Aya
from llm_module.config import ModelConfigs

# Initialize a model
config = {
    "model_id": "meta-llama/Llama-3.2-3B-Instruct",
    "quantization": "auto",
    "temperature": 0.7,
    "max_new_tokens": 512,
    "hf_token_path": "./tokens/hf_token.txt"
}

llm = Llama(config)

# Set system prompt
llm.set_system_prompt("You are a helpful AI assistant specialized in cybersecurity.")

# Add messages and generate responses
llm.add_message("user", "What are common web vulnerabilities?")
result = llm.forward()
response = result['response']  # Extract response text
# result also contains: input_tokens, output_tokens, generation_time

# One-shot generation without affecting conversation state
standalone_result = llm.forward([{"role": "user", "content": "Explain XSS"}])
standalone_response = standalone_result['response']
print(f"Generated in {standalone_result['generation_time']:.2f}s using "
      f"{standalone_result['input_tokens']} input + {standalone_result['output_tokens']} output tokens")
```

## Features

### 🎯 Unified Model Interface
- **HuggingFace Models**: Llama, Phi, Qwen, DeepSeek, Aya
- **OpenAI Models**: GPT-3.5, GPT-4o, o3-mini
- Consistent API across all model types

### 🧠 Memory Optimization
- Automatic quantization (4-bit, 8-bit)
- Memory usage estimation
- GPU memory management

### 💬 Advanced Conversation Management
- Full conversation history tracking
- Export to training formats (Alpaca, ShareGPT, JSONL)
- Conversation analytics and statistics

### ⚙️ Configuration Management
- Pre-defined configurations for all models
- Task-specific optimization (jailbreaking, reasoning, creative)
- Memory-efficient configurations

### 🔧 Future LoRA Fine-tuning
- Complete LoRA configuration system
- Task-optimized parameters
- Ready for PEFT integration

## Supported Models

### HuggingFace Models
- **Llama**: 3.1-8B, 3.2-1B/3B, 3.3-70B variants
- **Phi**: Phi-3-mini, Phi-4
- **Qwen**: Qwen3-0.6B/4B/8B/32B, Qwen2.5-VL
- **DeepSeek**: R1-Distill series (1.5B, 14B, 32B, 70B)
- **Aya**: Expanse-8B/32B (multilingual)

### OpenAI Models
- GPT-3.5-turbo
- GPT-4o / GPT-4o-2024-08-06
- o3-mini / o3-mini-2025-01-31
- chatgpt-4o-latest

## Configuration Examples

### Pre-defined Configurations

```python
from llm_module.config import ModelConfigs

# List available configurations
configs = ModelConfigs.list_configs()
print(configs)

# Get specific configuration
llama_config = ModelConfigs.get_config("llama-3b")
gpt_config = ModelConfigs.get_config("gpt-4o")

# Memory-optimized configuration
memory_config = ModelConfigs.get_memory_efficient_config("llama-70b", available_memory_gb=16)
```

### Task-Specific Configurations

```python
# Get configuration optimized for jailbreaking research
jailbreak_config = ModelConfigs.get_recommended_config_for_task("jailbreaking", "llama")

# Other task optimizations
reasoning_config = ModelConfigs.get_recommended_config_for_task("reasoning", "deepseek")
creative_config = ModelConfigs.get_recommended_config_for_task("creative", "llama")
```

## Advanced Features

### Conversation Management

```python
from llm_module.utils import ConversationManager

# Create conversation manager
conv = ConversationManager()
conv.set_system_prompt("You are a security researcher.")
conv.add_message("user", "What is a buffer overflow?")
conv.add_message("assistant", "A buffer overflow occurs when...")

# Get conversation statistics
stats = conv.get_conversation_stats()
print(f"Total messages: {stats['total_messages']}")
print(f"Estimated tokens: {stats['estimated_tokens']}")

# Export for training
training_data = conv.export_for_training("alpaca")

# Save/load conversations
conv.save_to_file("conversation.json")
loaded_conv = ConversationManager.load_from_file("conversation.json")
```

### Model Utilities

```python
from llm_module.utils import ModelUtils

# Validate model IDs
validation = ModelUtils.validate_model_id("meta-llama/Llama-3.2-3B-Instruct")
print(validation)

# Extract model information
family = ModelUtils.extract_model_family("microsoft/Phi-3-mini-4k-instruct")
size = ModelUtils.extract_model_size("meta-llama/Llama-3.2-3B-Instruct")

# Get system information
system_info = ModelUtils.get_system_info()
print(f"GPU available: {system_info['cuda_available']}")
```

### Quantization

```python
from llm_module.quantization import QuantizationConfig

# Auto-select quantization based on model size
mode = QuantizationConfig.auto_select_quantization("meta-llama/Llama-3.3-70B-Instruct")
print(f"Recommended quantization: {mode}")

# Get memory usage information
memory_info = QuantizationConfig.get_memory_usage_info("meta-llama/Llama-3.2-7B-Instruct", "4bit")
print(f"Estimated memory: {memory_info['estimated_memory_gb']} GB")
```

## LoRA Fine-tuning (Future)

```python
from llm_module.finetuning import LoRAConfig, LoRATrainer

# Get task-optimized LoRA configuration
lora_config = LoRAConfig.get_task_optimized_config("jailbreaking", "llama")

# Create trainer for specific task
trainer = LoRATrainer.create_trainer_for_task(model, "jailbreaking", "llama")

# Get training information
training_info = trainer.get_training_info()
print(f"Estimated trainable parameters: {training_info['parameter_info']['estimated_trainable_params']}")
```

## Architecture

The module is organized into several key components:

### Base Classes
- `BaseLLM`: Abstract base class defining the common interface
- `HuggingFaceBase`: Base for local HuggingFace models
- `OpenAIBase`: Base for OpenAI API models

### Model Implementations
Each model family has its own optimized implementation:
- `Llama`: Meta's Llama models with Llama-specific optimizations
- `GPT`: OpenAI models with API-specific features
- `Phi`: Microsoft Phi models with efficiency optimizations
- `Qwen`: Alibaba Qwen models with multilingual support
- `DeepSeek`: DeepSeek models optimized for reasoning
- `Aya`: Cohere Aya models with cultural awareness

### Core Systems
- **Quantization**: Memory optimization with 4-bit/8-bit support
- **Configuration**: Comprehensive config management
- **Utilities**: Text processing, model validation, system info
- **Conversation**: Advanced conversation tracking and analytics
- **Fine-tuning**: Future-ready LoRA framework

## Integration with Jailbreaking Agents

The module integrates seamlessly with your existing jailbreaking framework:

```python
from llm_module.models import Llama, GPT
from llm_module.config import ModelConfigs

# Initialize target model for jailbreaking
target_config = ModelConfigs.get_recommended_config_for_task("jailbreaking", "llama")
target_model = Llama(target_config)

# Initialize judge model
judge_config = ModelConfigs.get_config("gpt-4o")
judge_model = GPT(judge_config)

# Use with your existing agents
# Your attacker, target, and judge agents can now use these unified model interfaces
```

## Performance Considerations

### Memory Usage
- Use quantization for large models (70B parameters)
- `auto` quantization selects optimal settings based on available memory
- Memory usage is estimated and reported before model loading

### Model Selection
- **Small models (1B-3B)**: Fast inference, good for development/testing
- **Medium models (7B-14B)**: Balanced performance and resource usage
- **Large models (32B-70B)**: Best quality, requires quantization or multiple GPUs

### API vs Local
- **OpenAI API**: No local compute required, pay-per-use, consistent performance
- **HuggingFace Local**: Full control, privacy, one-time setup cost

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Use quantization or smaller model variants
2. **Missing API Keys**: Ensure tokens are properly configured in token files
3. **Model Download Issues**: Check HuggingFace token permissions
4. **Import Errors**: Verify all required dependencies are installed

### Getting Help

Check the examples in `llm_module/examples/basic_usage.py` for complete usage demonstrations.

## Future Roadmap

- [ ] Full LoRA fine-tuning implementation with PEFT
- [ ] Additional model families (Gemma, Claude via API)
- [ ] Advanced quantization techniques (GPTQ, AWQ)
- [ ] Distributed inference support
- [ ] Integration with evaluation frameworks

## Contributing

The module is designed to be easily extensible. To add a new model family:

1. Create a new file in `models/` directory
2. Inherit from `HuggingFaceBase` or `OpenAIBase`
3. Implement model-specific optimizations
4. Add configuration presets in `config/model_configs.py`
5. Update the main `__init__.py` file

The architecture supports rapid addition of new models and features while maintaining backward compatibility.
