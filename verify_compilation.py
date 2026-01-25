"""Quick verification script to check if LanguageDetectProcessor compiles."""

import sys
import asyncio

async def verify():
    """Verify the LanguageDetectProcessor can be imported and instantiated."""
    try:
        # Test import
        from genai_processors.contrib import LanguageDetectProcessor
        print("✓ Import successful")
        
        # Test instantiation
        processor = LanguageDetectProcessor()
        print("✓ Instantiation successful")
        
        # Test with custom parameters
        processor_custom = LanguageDetectProcessor(
            metadata_key="lang",
            unknown_label="not_detected",
            min_text_length=5
        )
        print("✓ Custom parameters successful")
        
        # Test basic functionality
        from genai_processors.core import ProcessorPart
        
        async def part_gen():
            yield ProcessorPart(
                text="This is a test.",
                mimetype="text/plain",
                metadata={}
            )
        
        result = [p async for p in processor(part_gen())]
        if result and "language" in result[0].metadata:
            print(f"✓ Basic processing successful (detected: {result[0].metadata['language']})")
        else:
            print("✗ Basic processing failed")
            return False
        
        print("\n✅ All compilation checks passed!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\nMake sure to install with: pip install .[contrib]")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(verify())
    sys.exit(0 if success else 1)
