"""
Basic usage examples for the LLM module.

This script demonstrates the key functionality of the llm_module with both
the direct model imports and the new model factory approach.
"""

import logging
from typing import Dict, Any, List

# New model factory import (recommended approach)
from jailbreak.llm_module import llm_model_factory

# Traditional imports (still available for advanced usage)
from jailbreak.llm_module.models import Llama, GPT, Phi, Qwen, DeepSeek, Aya
from jailbreak.llm_module.config import ModelConfigs, ConfigManager
from jailbreak.llm_module.utils import ConversationManager, ModelUtils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configurations for different models
TEST_CONFIGS = [
    {
        "name": "Qwen 0.6B",
        "config": {
            "model": "qwen",
            "model_id": "Qwen/Qwen3-0.6B",
            "quantization": "none",
            "temperature": 0.7,
            "max_new_tokens": 512,
            "hf_token_path": "/home/mrahma45/HUGGINGFACE_KEY"
        }
    },
    {
        "name": "Llama 3.2 3B",
        "config": {
            "model": "llama",
            "model_id": "meta-llama/Llama-3.2-3B-Instruct",
            "quantization": "auto",
            "temperature": 0.8,
            "max_new_tokens": 1024,
            "hf_token_path": "/home/mrahma45/HUGGINGFACE_KEY"
        }
    },
    {
        "name": "GPT-4o",
        "config": {
            "model": "gpt",
            "model_id": "gpt-4o",
            "api_key_path": "/projects/klybarge/OPENAI_API_KEY_GENERAL",
            "temperature": 0.5,
            "max_tokens": 512
        }
    },
    {
        "name": "Phi-3 Mini",
        "config": {
            "model": "phi",
            "model_id": "microsoft/Phi-3-mini-4k-instruct",
            "quantization": "auto",
            "temperature": 0.7,
            "max_new_tokens": 512,
            "hf_token_path": "/home/mrahma45/HUGGINGFACE_KEY"
        }
    },
    {
        "name": "DeepSeek V2",
        "config": {
            "model": "deepseek",
            "model_id": "deepseek-ai/DeepSeek-V2-Chat",
            "quantization": "4bit",
            "temperature": 0.6,
            "max_new_tokens": 512,
            "hf_token_path": "/home/mrahma45/HUGGINGFACE_KEY"
        }
    }
]


def test_model_factory(config: Dict[str, Any], model_name: str) -> None:
    """Test the model factory with a given configuration."""
    logger.info(f"Testing {model_name} with factory approach")
    
    try:
        # Create model using factory
        llm = llm_model_factory(config)
        logger.info(f"Successfully created {model_name}: {llm}")
        
        # Set system prompt
        llm.set_system_prompt("You are a helpful AI assistant specialized in cybersecurity.")
        logger.info(f"Set system prompt for {model_name}")
        
        # Add test message
        test_message = "What are common attack vectors in web applications?"
        llm.add_message("user", test_message)
        logger.info(f"Added test message to {model_name}")
        
        # Add conversation history for testing
        conversation_history = [
            {"role": "user", "content": "Can you explain SQL injection?"},
            {"role": "assistant", "content": "SQL injection is a code injection technique..."}
        ]
        llm.add_conversation(conversation_history)
        logger.info(f"Added conversation history to {model_name}")
        
        # Get model information
        model_info = llm.get_model_info()
        logger.info(f"{model_name} Model Info: {model_info}")
        
        # Test API methods
        current_prompt = llm.get_system_prompt()
        logger.info(f"{model_name} system prompt: {current_prompt[:50]}..." if current_prompt else "No system prompt")
        
        conversation = llm.get_conversation()
        logger.info(f"{model_name} conversation length: {len(conversation)} messages")
        
        logger.info(f"{model_name} model ready for generation!")
        
    except Exception as e:
        logger.error(f"Failed to create or test {model_name}: {e}")
        logger.info("This is expected in demo mode - shows API usage without actual model loading")


def example_configuration_management():
    """Example of using configuration management."""
    logger.info("Configuration Management Example")
    
    # List all available configurations
    try:
        configs = ModelConfigs.list_configs()
        logger.info(f"Available configurations: {configs}")
    except Exception as e:
        logger.error(f"Failed to list configs: {e}")
    
    # Get a predefined configuration
    try:
        llama_config = ModelConfigs.get_config("llama-3b")
        logger.info(f"Llama 3B config: {llama_config}")
    except KeyError as e:
        logger.warning(f"Config not found: {e}")
    except Exception as e:
        logger.error(f"Error getting config: {e}")
    
    # Get memory-efficient configuration
    try:
        memory_config = ModelConfigs.get_memory_efficient_config("llama-3b", available_memory_gb=8)
        logger.info(f"Memory-efficient config: {memory_config}")
    except Exception as e:
        logger.error(f"Error getting memory-efficient config: {e}")
    
    # Use ConfigManager
    config_manager = ConfigManager()
    try:
        config = config_manager.get_config("llama-3b")
        logger.info(f"Config via manager: {config}")
    except Exception as e:
        logger.error(f"Config manager error: {e}")


def example_conversation_management():
    """Example of advanced conversation management."""
    print("\n=== Conversation Management Example ===")
    
    # Create conversation manager
    conv = ConversationManager()
    
    # Add messages
    conv.set_system_prompt("You are a security researcher.")
    conv.add_message("user", "What is a buffer overflow?")
    conv.add_message("assistant", "A buffer overflow occurs when...")
    
    # Get conversation statistics
    stats = conv.get_conversation_stats()
    print(f"Conversation stats: {stats}")
    
    # Get messages in API format
    messages = conv.get_messages()
    print(f"Messages for API: {messages}")
    
    # Export for training
    training_data = conv.export_for_training("alpaca")
    print(f"Training format: {training_data}")


def example_model_utilities():
    """Example of using model utilities."""
    print("\n=== Model Utilities Example ===")
    
    # Validate model IDs
    validation = ModelUtils.validate_model_id("meta-llama/Llama-3.2-3B-Instruct")
    print(f"Model validation: {validation}")
    
    # Extract model information
    family = ModelUtils.extract_model_family("microsoft/Phi-3-mini-4k-instruct")
    size = ModelUtils.extract_model_size("meta-llama/Llama-3.2-3B-Instruct")
    print(f"Model family: {family}, size: {size}")
    
    # Get system information
    try:
        system_info = ModelUtils.get_system_info()
        print(f"System info: GPU available: {system_info.get('cuda_available', False)}")
    except Exception as e:
        print(f"System info error: {e}")
    
    # Text utilities
    text = "This is a sample text with multiple    spaces and\n\nextra newlines."
    cleaned = ModelUtils.clean_text(text)
    print(f"Cleaned text: '{cleaned}'")
    
    # Token estimation
    token_count = ModelUtils.count_tokens_rough("Hello, how are you doing today?")
    print(f"Estimated tokens: {token_count}")


def example_quantization_features():
    """Example of quantization features."""
    print("\n=== Quantization Features Example ===")
    
    from jailbreak.llm_module.quantization import QuantizationConfig, MemoryOptimizer
    
    # Auto-select quantization
    mode = QuantizationConfig.auto_select_quantization("meta-llama/Llama-3.3-70B-Instruct")
    print(f"Recommended quantization for 70B model: {mode}")
    
    # Get memory usage info
    memory_info = QuantizationConfig.get_memory_usage_info(
        "meta-llama/Llama-3.2-7B-Instruct", 
        mode
    )
    print(f"Memory usage: {memory_info}")
    
    # Check memory requirements  
    requirements = MemoryOptimizer.check_memory_requirements(
        "meta-llama/Llama-3.2-7B-Instruct",
        quantization_mode="4bit"
    )
    print(f"Memory requirements: {requirements}")


def example_lora_configuration():
    """Example of LoRA configuration for future fine-tuning."""
    print("\n=== LoRA Configuration Example ===")
    
    from jailbreak.llm_module.finetuning import LoRAConfig, LoRATrainer
    
    # Get optimized LoRA config for jailbreaking task
    lora_config = LoRAConfig.get_task_optimized_config("jailbreaking", "llama")
    print(f"LoRA config: {lora_config.to_dict()}")
    
    # Get parameter estimates
    param_info = lora_config.estimate_parameters()
    print(f"Parameter info: {param_info}")
    
    # Get compatibility info
    compat_info = lora_config.get_compatibility_info()
    print(f"Compatibility: {compat_info}")


def main():
    """Run all examples to demonstrate the LLM module functionality."""
    logger.info("Starting LLM Module - Basic Usage Examples")
    logger.info("=" * 50)
    
    try:
        # Test all models using the factory approach
        logger.info("=== MODEL FACTORY TESTS ===")
        for test_config in TEST_CONFIGS:
            test_model_factory(test_config["config"], test_config["name"])
            logger.info("-" * 30)
        
        # Test utility functions
        logger.info("\n=== UTILITY EXAMPLES ===")
        logger.info("Testing configuration management...")
        example_configuration_management()
        
        logger.info("Testing conversation management...")
        example_conversation_management()
        
        logger.info("Testing model utilities...")
        example_model_utilities()
        
        logger.info("Testing quantization features...")
        example_quantization_features()
        
        logger.info("Testing LoRA configuration...")
        example_lora_configuration()
        
        logger.info("=" * 50)
        logger.info("All examples completed successfully!")
        logger.info("\nKey Features Demonstrated:")
        logger.info("✓ NEW: Model Factory (llm_model_factory) - RECOMMENDED")
        logger.info("✓ Dynamic model loading with simple config format")
        logger.info("✓ Unified interface across all model types")
        logger.info("✓ Configuration-based initialization")
        logger.info("✓ System prompt management (llm.set_system_prompt())")
        logger.info("✓ Message and conversation management")
        logger.info("✓ Dual forward modes (with/without conversation state)")
        logger.info("✓ Quantization support for memory efficiency")
        logger.info("✓ Future LoRA fine-tuning framework")
        logger.info("✓ Comprehensive configuration management")
        logger.info("✓ Conversation utilities and analytics")
        
        logger.info("\n🎉 RECOMMENDED USAGE:")
        logger.info("from jailbreak.llm_module import llm_model_factory")
        logger.info("config = {'model': 'qwen', 'model_id': 'Qwen/Qwen3-0.6B', ...}")
        logger.info("llm = llm_model_factory(config)")
        
    except Exception as e:
        logger.error(f"Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
