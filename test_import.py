"""Quick import test for LanguageDetectProcessor."""

import sys

def test_imports():
    """Test that all imports work correctly."""
    try:
        print("Testing imports...")
        
        # Test core imports
        from genai_processors.core.part_processor import PartProcessor
        print("✓ PartProcessor imported")
        
        from genai_processors.core.processor_part import ProcessorPart
        print("✓ ProcessorPart imported")
        
        from genai_processors.core.utils import is_text
        print("✓ is_text imported")
        
        # Test langdetect
        try:
            from langdetect import detect
            print("✓ langdetect imported")
        except ImportError:
            print("✗ langdetect not installed - run: pip install langdetect")
            return False
        
        # Test LanguageDetectProcessor
        from genai_processors.contrib.language_detect_processor import LanguageDetectProcessor
        print("✓ LanguageDetectProcessor imported")
        
        # Test instantiation
        processor = LanguageDetectProcessor()
        print("✓ LanguageDetectProcessor instantiated")
        
        # Test from contrib module
        from genai_processors.contrib import LanguageDetectProcessor as LDP
        print("✓ LanguageDetectProcessor imported from contrib")
        
        print("\n✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
