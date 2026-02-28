# Code Documentation Processor

A GenAI Processors contribution that automatically generates comprehensive documentation for code files using Gemini models.

## Overview

The `CodeDocumentationProcessor` analyzes code structure and generates high-quality documentation in various formats. It supports multiple programming languages and follows established documentation conventions.

## Features

- 🔍 **Multi-language Support**: Detects and processes Python, JavaScript, TypeScript, Java, C++, and more
- 📝 **Multiple Output Formats**: Markdown, reStructuredText, HTML documentation
- 🎯 **Configurable Docstring Styles**: Google, Sphinx, NumPy conventions
- 🏗️ **Structure Analysis**: Extracts functions, classes, methods, and imports
- 💡 **Smart Examples**: Generates usage examples automatically
- 🔧 **Type Hint Analysis**: Processes and documents type annotations
- ⚡ **Batch Processing**: Handle multiple files efficiently

## Installation

1. Ensure you have the GenAI Processors library installed
2. Place `code_documentation_processor.py` in the `contrib/` directory
3. Set your Google AI API key: `export GOOGLE_AI_API_KEY='your-api-key'`

## Quick Start

```python
import asyncio
from contrib.code_documentation_processor import CodeDocumentationProcessor
from genai_processors import content_api

async def main():
    # Initialize processor
    processor = CodeDocumentationProcessor(
        api_key="your-google-ai-api-key",
        documentation_format="markdown",
        include_examples=True
    )
    
    # Your code to document
    code = '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
    '''
    
    # Create processor part
    code_part = content_api.ProcessorPart(code, metadata={'filename': 'math_utils.py'})
    
    # Generate documentation (single file)
    async for doc_part in processor.call(code_part):
        print(doc_part.text)

asyncio.run(main())
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `"gemini-2.0-flash-exp"` | Gemini model to use |
| `documentation_format` | `"markdown"` | Output format (`markdown`, `rst`, `html`) |
| `docstring_style` | `"google"` | Docstring convention (`google`, `sphinx`, `numpy`) |
| `include_examples` | `True` | Generate usage examples |
| `include_type_hints` | `True` | Analyze type annotations |

## Usage Examples

### Single File Documentation

```python
from contrib.code_documentation_processor import document_code_file

# Document a file from disk (uses the utility function)
docs = await document_code_file(
    file_path="my_module.py",
    api_key="your-api-key",
    output_format="markdown"
)
print(docs)
```

### Batch Processing

```python
from genai_processors import streams

# Process multiple files
files = ["utils.py", "models.py", "views.py"]
parts = []

for filepath in files:
    with open(filepath, 'r') as f:
        content = f.read()
    parts.append(ProcessorPart(content, metadata={'filename': filepath}))

# Convert PartProcessor to Processor for batch processing
full_processor = processor.to_processor()
content_stream = streams.stream_content(parts)

async for doc in full_processor(content_stream):
    print(f"Documentation for {doc.metadata['original_filename']}:")
    print(doc.text)
```

### Custom Configuration

```python
processor = CodeDocumentationProcessor(
    api_key="your-api-key",
    documentation_format="rst",
    docstring_style="sphinx",
    include_examples=True,
    include_type_hints=True
)

# For single file
async for doc in processor.call(code_part):
    print(doc.text)

# For multiple files
full_processor = processor.to_processor()
async for doc in full_processor(streams.stream_content(parts)):
    print(doc.text)
```

## Supported Languages

- **Python** (.py) - Full AST analysis, docstring generation
- **JavaScript** (.js) - Function and class extraction
- **TypeScript** (.ts) - Type-aware documentation
- **Java** (.java) - Object-oriented structure analysis
- **C++** (.cpp, .cc) - Function and class documentation
- **C** (.c) - Function documentation
- **Go** (.go) - Package and function docs
- **Rust** (.rs) - Module and function analysis
- **And more...** - Extensible language detection

## Output Examples

### Generated Markdown Documentation

```markdown
# fibonacci_calculator.py

## Overview
This module contains mathematical utility functions and a Calculator class for basic arithmetic operations with history tracking.

## Functions

### `fibonacci(n)`
Calculates the nth Fibonacci number using recursive approach.

**Parameters:**
- `n` (int): The position in the Fibonacci sequence

**Returns:**
- `int`: The nth Fibonacci number

**Example:**
```python
result = fibonacci(5)  # Returns 5
```

### Classes

#### `Calculator`
A simple calculator with operation history.

**Methods:**
- `add(a, b)`: Adds two numbers and records the operation
- `get_history()`: Returns a copy of the operation history
```

## Integration with GenAI Processors

This processor follows the standard GenAI Processors patterns:

- ✅ Inherits from `processor.PartProcessor` (ideal for independent file processing)
- ✅ Uses `ProcessorPart` and `ProcessorContent` correctly
- ✅ Implements async `call(part)` method for PartProcessors
- ✅ Can be converted to `Processor` using `.to_processor()` for batch processing
- ✅ Proper metadata handling and error handling
- ✅ Uses `streams.stream_content()` for proper async iteration
- ✅ Configurable system instructions via prompts

**Usage Patterns:**
```python
# Single file (PartProcessor)
async for doc in processor.call(part):
    print(doc.text)

# Multiple files (convert to Processor)
full_processor = processor.to_processor()
stream = streams.stream_content(parts)
async for doc in full_processor(stream):
    print(doc.text)
```

## Testing

Run the example script to test the processor:

```bash
python contrib/code_documentation_processor_example.py
```

The example demonstrates:
1. Single file documentation
2. Multiple file processing
3. File-from-disk documentation
4. Batch directory processing

## Contributing

This processor can be extended to support:
- Additional programming languages
- Custom documentation templates
- Integration with documentation sites (Sphinx, GitBook)
- Code quality analysis
- API documentation generation
- Interactive documentation with examples

## Why This Contribution?

1. **Practical Value**: Every developer needs better documentation
2. **Fits Library Philosophy**: Natural extension of GenAI processing pipelines
3. **Reusable**: Can be combined with other processors
4. **Extensible**: Easy to add new languages and formats
5. **Standards Compliant**: Follows established documentation conventions

## License

Licensed under the Apache License, Version 2.0, same as the GenAI Processors library.