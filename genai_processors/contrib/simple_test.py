#!/usr/bin/env python3
"""Simple test script for Code Documentation Processor.

Run this from the contrib/ directory:
cd contrib/
python simple_test.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path to import genai_processors
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

try:
    from genai_processors import content_api
    from code_documentation_processor import CodeDocumentationProcessor
    
    # Use the same pattern as the existing codebase
    ProcessorPart = content_api.ProcessorPart
    ProcessorContent = content_api.ProcessorContent
    
    print("✅ Successfully imported modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the contrib/ directory")
    print("And that genai_processors is installed")
    sys.exit(1)


async def simple_test():
    """Simple test of the code documentation processor."""
    
    # Check for API key
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        print("❌ Please set GOOGLE_AI_API_KEY environment variable")
        print("Example: export GOOGLE_AI_API_KEY='your-api-key'")
        return
    
    print("🚀 Testing Code Documentation Processor")
    print("=" * 50)
    
    # Simple test code
    test_code = '''
def add(a, b):
    """Add two numbers together."""
    return a + b

def multiply(a, b):
    """Multiply two numbers."""
    return a * b

class Calculator:
    def __init__(self):
        self.value = 0
    
    def calculate(self, operation, a, b):
        if operation == "add":
            return add(a, b)
        elif operation == "multiply":
            return multiply(a, b)
        return None
'''
    
    try:
        # Create processor
        print("📝 Creating processor...")
        processor = CodeDocumentationProcessor(
            api_key=api_key,
            documentation_format="markdown",
            include_examples=True
        )
        
        # Create content
        code_part = ProcessorPart(
            test_code,
            metadata={'filename': 'test_calculator.py'}
        )
        
        print("🔄 Processing code...")
        
        # Generate documentation (call the PartProcessor directly on the part)
        async for doc_part in processor.call(code_part):
            print("✅ Generated documentation:")
            print("-" * 50)
            print(doc_part.text)  # Use .text instead of .data
            print("-" * 50)
            print(f"Language detected: {doc_part.metadata.get('language', 'unknown')}")
            print(f"Format: {doc_part.metadata.get('documentation_format', 'unknown')}")
            
        print("\n🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(simple_test())