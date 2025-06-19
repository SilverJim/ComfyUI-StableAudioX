#!/usr/bin/env python3
"""
Debug script to check ComfyUI node loading issues.
This simulates how ComfyUI loads the nodes.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def simulate_comfyui_loading():
    """Simulate how ComfyUI loads the AudioX nodes."""
    print("Simulating ComfyUI node loading process...")
    
    try:
        print("1. Testing __init__.py import...")
        
        # This simulates how ComfyUI imports the extension
        import __init__ as audiox_init
        
        print("   ✅ __init__.py imported successfully")
        
        # Check if NODE_CLASS_MAPPINGS exists
        if hasattr(audiox_init, 'NODE_CLASS_MAPPINGS'):
            mappings = audiox_init.NODE_CLASS_MAPPINGS
            print(f"   ✅ NODE_CLASS_MAPPINGS found with {len(mappings)} nodes")
            
            # Check for our volume control nodes
            basic_node = mappings.get("AudioXVolumeControl")
            advanced_node = mappings.get("AudioXAdvancedVolumeControl")
            
            if basic_node:
                print("   ✅ AudioXVolumeControl found in mappings")
                try:
                    # Test instantiation
                    instance = basic_node()
                    print("   ✅ AudioXVolumeControl can be instantiated")
                    
                    # Test INPUT_TYPES
                    input_types = instance.INPUT_TYPES()
                    print(f"   ✅ AudioXVolumeControl INPUT_TYPES: {list(input_types.keys())}")
                except Exception as e:
                    print(f"   ❌ AudioXVolumeControl instantiation failed: {e}")
            else:
                print("   ❌ AudioXVolumeControl NOT found in mappings")
            
            if advanced_node:
                print("   ✅ AudioXAdvancedVolumeControl found in mappings")
                try:
                    # Test instantiation
                    instance = advanced_node()
                    print("   ✅ AudioXAdvancedVolumeControl can be instantiated")
                    
                    # Test INPUT_TYPES
                    input_types = instance.INPUT_TYPES()
                    required = input_types.get("required", {})
                    optional = input_types.get("optional", {})
                    print(f"   ✅ AudioXAdvancedVolumeControl required: {list(required.keys())}")
                    print(f"   ✅ AudioXAdvancedVolumeControl optional: {list(optional.keys())}")
                except Exception as e:
                    print(f"   ❌ AudioXAdvancedVolumeControl instantiation failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("   ❌ AudioXAdvancedVolumeControl NOT found in mappings")
            
            # List all available nodes
            print(f"\n   📋 All available nodes:")
            for node_name in sorted(mappings.keys()):
                print(f"      - {node_name}")
                
        else:
            print("   ❌ NODE_CLASS_MAPPINGS not found in __init__.py")
            
    except Exception as e:
        print(f"   ❌ Failed to import __init__.py: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ ComfyUI loading simulation completed!")
    return True

def check_node_class_directly():
    """Try to import the node class directly from the module."""
    print("\n" + "="*60)
    print("Testing direct node class import...")
    
    try:
        # Mock the relative imports to make them work
        import audiox_utils
        sys.modules['ComfyUI-AudioX.audiox_utils'] = audiox_utils
        
        # Now try to import the nodes module
        import nodes as audiox_nodes
        
        print("   ✅ nodes.py imported successfully")
        
        # Check if the classes exist
        if hasattr(audiox_nodes, 'AudioXVolumeControl'):
            print("   ✅ AudioXVolumeControl class found")
        else:
            print("   ❌ AudioXVolumeControl class NOT found")
            
        if hasattr(audiox_nodes, 'AudioXAdvancedVolumeControl'):
            print("   ✅ AudioXAdvancedVolumeControl class found")
        else:
            print("   ❌ AudioXAdvancedVolumeControl class NOT found")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Direct import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 AudioX Node Loading Debugger")
    print("="*60)
    
    success1 = simulate_comfyui_loading()
    success2 = check_node_class_directly()
    
    if success1 and success2:
        print("\n🎉 All tests passed! The nodes should be loading correctly in ComfyUI.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        print("\nPossible solutions:")
        print("1. Restart ComfyUI completely")
        print("2. Check ComfyUI console for error messages")
        print("3. Verify all dependencies are installed")
    
    sys.exit(0 if (success1 and success2) else 1)
