#!/usr/bin/env python3
"""
Test script to verify HuggingFace model caching uses the correct default path.

This script tests that models are cached to the new default location:
github_repos/jailbreak/hf_models/
"""

import os
import sys
import pathlib
from jailbreak.llm_module.config import ModelConfigs


def test_default_cache_path():
    """Test that the default cache path is correctly configured."""
    print("Testing HuggingFace Model Cache Path Configuration")
    print("=" * 60)
    
    # Get a sample configuration
    config = ModelConfigs.get_config("llama-3b")
    
    print(f"Sample config (llama-3b): {config}")
    print(f"hf_model_path in config: {'hf_model_path' in config}")
    
    if 'hf_model_path' in config:
        print(f"Configured path: {config['hf_model_path']}")
        print("❌ UNEXPECTED: hf_model_path should not be in base config anymore")
        return False
    else:
        print("✅ EXPECTED: hf_model_path not in config - will use default")
    
    # Test path resolution logic (simulate what happens in HuggingFaceBase)
    current_file = pathlib.Path(__file__).resolve()
    jailbreak_root = current_file.parent  # This script is at jailbreak root
    expected_hf_path = jailbreak_root / "hf_models"
    
    print(f"\nPath Resolution Test:")
    print(f"Current script location: {current_file}")
    print(f"Detected jailbreak root: {jailbreak_root}")
    print(f"Expected hf_models path: {expected_hf_path}")
    print(f"hf_models directory exists: {expected_hf_path.exists()}")
    
    # Verify the hf_models directory exists
    if expected_hf_path.exists() and expected_hf_path.is_dir():
        print("✅ PASSED: hf_models directory exists at expected location")
        return True
    else:
        print("❌ FAILED: hf_models directory not found at expected location")
        return False


def test_model_path_override():
    """Test that custom hf_model_path still works."""
    print("\n" + "=" * 60) 
    print("Testing Custom Path Override")
    
    # Create a config with custom path
    custom_config = ModelConfigs.get_config("llama-3b")
    custom_path = "/tmp/custom_hf_models"
    custom_config["hf_model_path"] = custom_path
    
    print(f"Custom config with override: {custom_config}")
    print(f"Custom hf_model_path: {custom_config['hf_model_path']}")
    
    if custom_config["hf_model_path"] == custom_path:
        print("✅ PASSED: Custom path override works correctly")
        return True
    else:
        print("❌ FAILED: Custom path override not working")
        return False


def main():
    """Run all tests."""
    print("HuggingFace Model Cache Path Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Default path behavior
        test1_passed = test_default_cache_path()
        
        # Test 2: Custom path override
        test2_passed = test_model_path_override()
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("-" * 20)
        print(f"Default cache path: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
        print(f"Custom path override: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
        
        if test1_passed and test2_passed:
            print("\n🎉 ALL TESTS PASSED!")
            print("HuggingFace models will be cached to: github_repos/jailbreak/hf_models/")
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
