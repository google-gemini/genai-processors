#!/usr/bin/env python3
"""Example usage of the Code Documentation Processor.

This script demonstrates how to use the CodeDocumentationProcessor to 
automatically generate documentation for code files.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to Python path so we can import the processor
sys.path.insert(0, str(Path(__file__).parent.parent))

from genai_processors import content_api
from genai_processors import streams

# Use the same pattern as the existing codebase
ProcessorPart = content_api.ProcessorPart
ProcessorContent = content_api.ProcessorContent

# Import the processor from the same directory
from code_documentation_processor import (
    CodeDocumentationProcessor,
    document_code_file,
    create_code_documentation_processor
)


async def example_single_file():
    """Example: Document a single Python function."""
    print("🔍 Example 1: Documenting a simple Python function")
    
    # Sample Python code
    sample_code = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def get_history(self):
        return self.history.copy()
'''
    
    # Get API key (you'll need to set this)
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        print("❌ Please set GOOGLE_AI_API_KEY environment variable")
        return
    
    # Create processor
    processor = CodeDocumentationProcessor(
        api_key=api_key,
        documentation_format="markdown",
        include_examples=True
    )
    
    # Create ProcessorPart with metadata
    code_part = ProcessorPart(
        sample_code,
        metadata={
            'filename': 'fibonacci_calculator.py',
            'description': 'Simple math functions and calculator class'
        }
    )
    
    # Process and generate documentation
    print("📝 Generating documentation...")
    async for doc_part in processor.call(code_part):
        print("✅ Generated documentation:")
        print("=" * 50)
        print(doc_part.text)  # Use .text instead of .data
        print("=" * 50)
        print(f"Metadata: {doc_part.metadata}")


async def example_multiple_files():
    """Example: Document multiple code files."""
    print("\n🔍 Example 2: Documenting multiple files")
    
    # Sample files
    files = {
        'utils.py': '''
import math
from typing import List, Optional

def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def find_closest_point(target: tuple, points: List[tuple]) -> Optional[tuple]:
    """Find the closest point to target from a list of points."""
    if not points:
        return None
    
    closest = points[0]
    min_distance = calculate_distance(target[0], target[1], closest[0], closest[1])
    
    for point in points[1:]:
        distance = calculate_distance(target[0], target[1], point[0], point[1])
        if distance < min_distance:
            min_distance = distance
            closest = point
    
    return closest
''',
        'main.js': '''
// Simple JavaScript utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

class EventManager {
    constructor() {
        this.events = new Map();
    }
    
    on(event, callback) {
        if (!this.events.has(event)) {
            this.events.set(event, []);
        }
        this.events.get(event).push(callback);
    }
    
    emit(event, ...args) {
        if (this.events.has(event)) {
            this.events.get(event).forEach(callback => callback(...args));
        }
    }
}
'''
    }
    
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        print("❌ Please set GOOGLE_AI_API_KEY environment variable")
        return
    
    # Create processor with different settings
    part_processor = create_code_documentation_processor(
        api_key=api_key,
        documentation_format="markdown",
        docstring_style="google",
        include_examples=True,
        include_type_hints=True
    )
    
    # Convert to regular Processor for handling multiple files
    processor = part_processor.to_processor()
    
    # Create ProcessorParts for each file
    parts = []
    for filename, code in files.items():
        part = ProcessorPart(
            code,
            metadata={'filename': filename}
        )
        parts.append(part)
    
    # Process all files
    content_stream = streams.stream_content(parts)
    
    print("📝 Generating documentation for multiple files...")
    file_count = 0
    async for doc_part in processor(content_stream):
        file_count += 1
        original_file = doc_part.metadata.get('original_filename', f'file_{file_count}')
        print(f"\n✅ Documentation for {original_file}:")
        print("-" * 40)
        print(doc_part.text[:500] + "..." if len(doc_part.text) > 500 else doc_part.text)  # Use .text
        print("-" * 40)


async def example_file_from_disk():
    """Example: Document a real file from disk."""
    print("\n🔍 Example 3: Documenting a file from disk")
    
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        print("❌ Please set GOOGLE_AI_API_KEY environment variable")
        return
    
    # You can replace this with any Python file you want to document
    current_file = __file__
    
    try:
        documentation = await document_code_file(
            file_path=current_file,
            api_key=api_key,
            output_format="markdown"
        )
        
        print(f"✅ Documentation for {current_file}:")
        print("=" * 60)
        print(documentation[:800] + "..." if len(documentation) > 800 else documentation)
        print("=" * 60)
        
    except FileNotFoundError:
        print(f"❌ File {current_file} not found")
    except Exception as e:
        print(f"❌ Error processing file: {e}")


async def example_batch_processing():
    """Example: Batch process multiple files from a directory."""
    print("\n🔍 Example 4: Batch processing directory")
    
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        print("❌ Please set GOOGLE_AI_API_KEY environment variable")
        return
    
    # Create processor
    part_processor = CodeDocumentationProcessor(
        api_key=api_key,
        documentation_format="markdown"
    )
    
    # Convert to regular Processor for batch processing
    processor = part_processor.to_processor()
    
    # Find Python files in current directory (you can change this path)
    directory = Path(".")
    python_files = list(directory.glob("*.py"))[:3]  # Limit to first 3 files
    
    if not python_files:
        print("❌ No Python files found in current directory")
        return
    
    print(f"📁 Found {len(python_files)} Python files to process")
    
    # Create parts for each file
    parts = []
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            part = ProcessorPart(
                content,
                metadata={
                    'filename': str(file_path),
                    'file_size': len(content),
                    'lines': content.count('\n') + 1
                }
            )
            parts.append(part)
            
        except Exception as e:
            print(f"⚠️ Could not read {file_path}: {e}")
    
    if not parts:
        print("❌ No files could be processed")
        return
    
    # Process all files
    print("📝 Processing files...")
    content_stream = streams.stream_content(parts)
    async for doc_part in processor(content_stream):
        filename = doc_part.metadata.get('original_filename', 'unknown')
        print(f"\n✅ Processed: {Path(filename).name}")
        print(f"📊 Documentation length: {len(doc_part.text)} characters")  # Use .text


async def main():
    """Run all examples."""
    print("🚀 Code Documentation Processor Examples")
    print("=" * 50)
    
    # Check for API key
    if not os.getenv("GOOGLE_AI_API_KEY"):
        print("❌ Missing GOOGLE_AI_API_KEY environment variable")
        print("Please set it with: export GOOGLE_AI_API_KEY='your-api-key'")
        return
    
    try:
        await example_single_file()
        await example_multiple_files()
        await example_file_from_disk()
        await example_batch_processing()
        
        print("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())