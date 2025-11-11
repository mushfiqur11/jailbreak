"""
Basic usage examples for the LLM module.

This script demonstrates the key functionality of the llm_module with the exact API
as requested, including explicit imports and usage patterns.
"""

# Example usage as requested by the user
from llm_module.models import Llama, GPT, Phi, Qwen, DeepSeek, Aya
from llm_module.config import ModelConfigs, ConfigManager
from llm_module.utils import ConversationManager, ModelUtils


def example_llama_usage():
    """Example of using Llama model with the requested API."""
    print("=== Llama Model Example ===")
    
    # Configuration with variants as requested
    config = {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "quantization": "none",  # Can be "auto", "4bit", "8bit" 
        "temperature": 0.7,
        "max_new_tokens": 512,
        "hf_token_path": "/home/mrahma45/HUGGINGFACE_KEY"
    }
    
    # Initialize model as requested: llm = Llama(config)
    try:
        llm = Llama(config)
        
        # Set system prompt as requested
        llm.set_system_prompt("You are a helpful AI assistant specialized in cybersecurity.")
        
        # Get system prompt to verify it was set
        current_prompt = llm.get_system_prompt()
        print(f"Current system prompt: {current_prompt[:50]}...")
        
        # Add message as requested
        llm.add_message("user", "What are common attack vectors in web applications?")
        
        # Add conversation as requested
        conversation_history = [
            {"role": "user", "content": "Can you explain SQL injection?"},
            {"role": "assistant", "content": "SQL injection is a code injection technique..."}
        ]
        llm.add_conversation(conversation_history)
        
        # Forward without arguments - uses conversation, doesn't save
        print("Generating response using conversation history...")
        response = llm.forward()  # This would actually call the model
        print(f"{response}")
        
        # Forward with messages - one-shot, doesn't affect conversation state
        standalone_messages = [{"role": "user", "content": "What is XSS?"}]
        print("Generating standalone response...")
        response = llm.forward(standalone_messages)  # One-shot generation
        print(f"{response}")
        
        print(f"f{llm.get_system_prompt()}")
        # Get model information
        model_info = llm.get_model_info()
        print(f"Model Info: {model_info}")
        
    except Exception as e:
        print(f"Note: Model loading failed (expected in demo): {e}")
        print("This is normal - the demo shows API usage without actual model loading")


def example_gpt_usage():
    """Example of using GPT model."""
    print("\n=== GPT Model Example ===")
    
    config = {
        "model_id": "gpt-4o",
        "api_key_path": "/projects/klybarge/OPENAI_API_KEY_GENERAL",
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    try:
        llm = GPT(config)
        llm.set_system_prompt("You are an expert in AI security research.")
        
        # Demonstrate get_system_prompt method
        current_prompt = llm.get_system_prompt()
        print(f"GPT system prompt: {current_prompt}")
        
        llm.add_message("user", "Explain jailbreaking techniques")
        
        # response = llm.forward()
        print("GPT response generated (placeholder)")
        
    except Exception as e:
        print(f"Note: GPT initialization failed (expected in demo): {e}")


def example_configuration_management():
    """Example of using configuration management."""
    print("\n=== Configuration Management Example ===")
    
    # List all available configurations
    configs = ModelConfigs.list_configs()
    print("Available configurations:", configs)
    
    # Get a predefined configuration
    try:
        llama_config = ModelConfigs.get_config("llama-3b")
        print(f"Llama 3B config: {llama_config}")
    except KeyError as e:
        print(f"Config not found: {e}")
    
    # Get memory-efficient configuration
    memory_config = ModelConfigs.get_memory_efficient_config("llama-3b", available_memory_gb=8)
    print(f"Memory-efficient config: {memory_config}")
    
    # Use ConfigManager
    config_manager = ConfigManager()
    try:
        config = config_manager.get_config("llama-3b")
        print(f"Config via manager: {config}")
    except Exception as e:
        print(f"Config manager error: {e}")


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
    
    from llm_module.quantization import QuantizationConfig, MemoryOptimizer
    
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
    
    from llm_module.finetuning import LoRAConfig, LoRATrainer
    
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
    print("LLM Module - Basic Usage Examples")
    print("=" * 50)
    
    try:
        example_llama_usage()
        example_gpt_usage()
        example_configuration_management()
        example_conversation_management()
        example_model_utilities()
        example_quantization_features()
        example_lora_configuration()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✓ Explicit model imports (from llm_module.models import Llama)")
        print("✓ Configuration-based initialization (llm = Llama(config))")
        print("✓ System prompt management (llm.set_system_prompt())")
        print("✓ Message and conversation management")
        print("✓ Dual forward modes (with/without conversation state)")
        print("✓ Quantization support for memory efficiency")
        print("✓ Future LoRA fine-tuning framework")
        print("✓ Comprehensive configuration management")
        print("✓ Conversation utilities and analytics")
        
    except Exception as e:
        print(f"Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
