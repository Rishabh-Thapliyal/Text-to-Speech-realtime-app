#!/usr/bin/env python3
"""
Simple test script to verify Kokoro TTS works in isolation
"""

def test_kokoro_basic():
    """Test basic Kokoro functionality"""
    print("🧪 Testing Kokoro TTS - Basic Functionality")
    print("=" * 50)
    
    # Test 1: Import
    try:
        print("1. Testing import...")
        import kokoro
        print(f"   ✅ Kokoro imported from: {kokoro.__file__}")
        print(f"   Version: {getattr(kokoro, '__version__', 'Unknown')}")
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: KPipeline class
    try:
        print("\n2. Testing KPipeline class...")
        from kokoro import KPipeline
        print("   ✅ KPipeline class imported")
    except ImportError as e:
        print(f"   ❌ KPipeline import failed: {e}")
        return False
    
    # Test 3: Instantiation
    try:
        print("\n3. Testing KPipeline instantiation...")
        pipeline = KPipeline(lang_code='a')
        print(f"   ✅ KPipeline instantiated: {type(pipeline)}")
    except Exception as e:
        print(f"   ❌ Instantiation failed: {e}")
        print(f"   Error type: {type(e)}")
        return False
    
    # Test 4: Check if callable
    try:
        print("\n4. Testing if KPipeline is callable...")
        if hasattr(pipeline, '__call__'):
            print("   ✅ KPipeline is callable")
        else:
            print("   ⚠️  KPipeline may not be callable")
            print("   Available methods:", [attr for attr in dir(pipeline) if not attr.startswith('_')])
    except Exception as e:
        print(f"   ❌ Callable check failed: {e}")
    
    # Test 5: Try to use the pipeline
    try:
        print("\n5. Testing basic usage...")
        text = "Hello world"
        print(f"   Testing with text: '{text}'")
        
        # Check what methods are available
        methods = [attr for attr in dir(pipeline) if not attr.startswith('_')]
        print(f"   Available methods: {methods}")
        
        # Try to find a method that looks like it generates audio
        if hasattr(pipeline, 'generate'):
            print("   Found 'generate' method, testing...")
            result = pipeline.generate(text)
            print(f"   ✅ Generate method works: {type(result)}")
        elif hasattr(pipeline, '__call__'):
            print("   Pipeline is callable, testing...")
            result = pipeline(text)
            print(f"   ✅ Call method works: {type(result)}")
        else:
            print("   ⚠️  No obvious audio generation method found")
            print("   This might be the issue!")
            
    except Exception as e:
        print(f"   ❌ Usage test failed: {e}")
        print(f"   Error type: {type(e)}")
        return False
    
    print("\n✅ Basic Kokoro tests completed!")
    return True

def test_kokoro_advanced():
    """Test more advanced Kokoro functionality"""
    print("\n🧪 Testing Kokoro TTS - Advanced Functionality")
    print("=" * 50)
    
    try:
        from kokoro import KPipeline
        
        # Test different language codes
        print("1. Testing different language codes...")
        lang_codes = ['a', 'en', 'ja']
        
        for lang in lang_codes:
            try:
                pipeline = KPipeline(lang_code=lang)
                print(f"   ✅ Language '{lang}': {type(pipeline)}")
            except Exception as e:
                print(f"   ❌ Language '{lang}': {e}")
        
        # Test voice options
        print("\n2. Testing voice options...")
        try:
            pipeline = KPipeline(lang_code='a')
            
            # Check if pipeline has voice-related attributes
            if hasattr(pipeline, 'voices'):
                print(f"   Available voices: {pipeline.voices}")
            elif hasattr(pipeline, 'voice'):
                print(f"   Current voice: {pipeline.voice}")
            else:
                print("   No voice information available")
                
        except Exception as e:
            print(f"   ❌ Voice test failed: {e}")
        
        # Test configuration
        print("\n3. Testing configuration...")
        try:
            pipeline = KPipeline(lang_code='a')
            config = {attr: getattr(pipeline, attr) for attr in dir(pipeline) 
                     if not attr.startswith('_') and not callable(getattr(pipeline, attr))}
            print(f"   Pipeline attributes: {config}")
            
        except Exception as e:
            print(f"   ❌ Configuration test failed: {e}")
            
    except Exception as e:
        print(f"❌ Advanced tests failed: {e}")
        return False
    
    print("\n✅ Advanced Kokoro tests completed!")
    return True

def main():
    """Main test function"""
    print("🎭 Kokoro TTS - Complete Test Suite")
    print("=" * 60)
    
    # Basic tests
    basic_ok = test_kokoro_basic()
    
    # Advanced tests
    advanced_ok = test_kokoro_advanced()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 40)
    print(f"   Basic tests: {'✅' if basic_ok else '❌'}")
    print(f"   Advanced tests: {'✅' if advanced_ok else '❌'}")
    
    if basic_ok and advanced_ok:
        print("\n🎉 All tests passed! Kokoro TTS is working correctly.")
        print("   The issue might be in the server integration.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for issues.")
        print("   You may need to reinstall or fix Kokoro TTS.")
    
    print("\n💡 Next steps:")
    print("   1. If tests pass: Check server logs for integration issues")
    print("   2. If tests fail: Fix Kokoro installation first")
    print("   3. Run debug script: python debug_kokoro_issue.py")

if __name__ == "__main__":
    main()
