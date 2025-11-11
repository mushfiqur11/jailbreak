#!/usr/bin/env python3
"""
Test script to verify that Qwen model configurations are fixed and working.

This script tests that Qwen models no longer cause VL model loading errors.
"""

import sys
from jailbreak.llm_module.config import ModelConfigs


def test_qwen_configurations():
    """Test that all Qwen configurations are properly set up."""
    print("Testing Qwen Model Configurations")
    print("=" * 50)
    
    # Get all Qwen configurations
    qwen_configs = ModelConfigs.list_configs()["qwen"]
    print(f"Available Qwen models: {qwen_configs}")
    
    # Test each Qwen configuration
    for qwen_model in qwen_configs:
        print(f"\nTesting: {qwen_model}")
        try:
            config = ModelConfigs.get_config(qwen_model)
            model_id = config["model_id"]
            
            # Check if it's a text-only model (not VL)
            is_vl_model = config.get("supports_vision", False) or "VL" in model_id
            is_qwen25_model = "Qwen2.5" in model_id
            
            print(f"  Model ID: {model_id}")
            print(f"  Is VL model: {is_vl_model}")
            print(f"  Is Qwen2.5 model: {is_qwen25_model}")
            print(f"  Quantization: {config['quantization']}")
            
            if qwen_model == "qwen-vl":
                # VL model should have special handling note
                if is_vl_model:
                    print(f"  ✅ VL model correctly identified")
                else:
                    print(f"  ❌ VL model not properly marked")
                    return False
            else:
                # Regular models should be text-only Qwen2.5
                if is_qwen25_model and not is_vl_model:
                    print(f"  ✅ Text-only Qwen2.5 model - should work with AutoModelForCausalLM")
                else:
                    print(f"  ❌ Not a proper text-only model")
                    return False
            
        except Exception as e:
            print(f"  ❌ Error getting config: {e}")
            return False
    
    return True


def test_old_problematic_models():
    """Verify that the old problematic model IDs are no longer used."""
    print("\n" + "=" * 50)
    print("Testing for Old Problematic Models")
    
    problematic_models = [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-4B", 
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-32B"
    ]
    
    all_configs = ModelConfigs.ALL_CONFIGS
    
    for config_name, config in all_configs.items():
        model_id = config.get("model_id", "")
        
        if model_id in problematic_models:
            print(f"❌ FOUND PROBLEMATIC MODEL: {config_name} uses {model_id}")
            return False
    
    print("✅ No problematic Qwen3 models found in configurations")
    return True


def main():
    """Run all tests."""
    print("Qwen Model Configuration Fix Test")
    print("=" * 50)
    
    try:
        # Test 1: Qwen configurations are properly set up
        test1_passed = test_qwen_configurations()
        
        # Test 2: Old problematic models are removed
        test2_passed = test_old_problematic_models()
        
        # Summary
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("-" * 20)
        print(f"Qwen configurations: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
        print(f"No problematic models: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
        
        if test1_passed and test2_passed:
            print("\n🎉 ALL TESTS PASSED!")
            print("Qwen models should now work without VL model loading errors!")
            return 0
        else:
            print("\n❌ SOME TESTS FAILED!")
            return 1
            
    except Exception as e:
        print(f"\n💥 ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
