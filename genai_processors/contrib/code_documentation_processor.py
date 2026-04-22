# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Code Documentation Processor for GenAI Processors library.

This processor automatically generates comprehensive documentation for code files.
It analyzes code structure, extracts key components, and creates well-formatted
documentation following standard conventions.

Usage:
  processor = CodeDocumentationProcessor(api_key="your_api_key")
  
  # Process a single code file
  code_part = ProcessorPart("def hello_world():\n    print('Hello!')")
  async for doc in processor.call(code_part):
      print(doc.text)
  
  # Process multiple files
  processor_full = processor.to_processor()
  stream = streams.stream_content([code_part1, code_part2])
  async for doc in processor_full(stream):
      print(doc.text)
"""

from typing import AsyncIterable
import re
import ast
from pathlib import Path

from genai_processors import processor
from genai_processors import content_api
from genai_processors import streams
from genai_processors.core import genai_model
from google.genai import types as genai_types

ProcessorPart = content_api.ProcessorPart
ProcessorContent = content_api.ProcessorContent


class CodeDocumentationProcessor(processor.PartProcessor):
  """Generates comprehensive documentation for code files.
  
  This processor analyzes code structure, extracts functions, classes, and
  modules, and generates documentation in various formats (Markdown, docstrings,
  API docs, etc.).
  
  Features:
  - Detects programming language automatically
  - Extracts function signatures and class definitions
  - Generates docstrings in standard formats (Google, Sphinx, NumPy)
  - Creates markdown documentation
  - Supports multiple programming languages
  - Preserves code context and relationships
  """

  def __init__(
      self,
      api_key: str,
      model_name: str = "gemini-2.0-flash-exp",
      documentation_format: str = "markdown",
      docstring_style: str = "google",
      include_examples: bool = True,
      include_type_hints: bool = True,
  ) -> None:
    """Initialize the Code Documentation Processor.
    
    Args:
      api_key: Google AI API key for Gemini model access
      model_name: Name of the Gemini model to use
      documentation_format: Output format ("markdown", "rst", "html")
      docstring_style: Docstring format ("google", "sphinx", "numpy")
      include_examples: Whether to generate usage examples
      include_type_hints: Whether to include type hint analysis
    """
    self._model = genai_model.GenaiModel(
        model_name=model_name,
        api_key=api_key,
    )
    self._documentation_format = documentation_format
    self._docstring_style = docstring_style
    self._include_examples = include_examples
    self._include_type_hints = include_type_hints

  def _detect_language(self, code: str, filename: str = "") -> str:
    """Detect programming language from code content and filename."""
    # File extension mapping
    extension_map = {
        '.py': 'python',
        '.js': 'javascript', 
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.cs': 'csharp',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.sh': 'bash',
    }
    
    # Check file extension first
    if filename:
      ext = Path(filename).suffix.lower()
      if ext in extension_map:
        return extension_map[ext]
    
    # Content-based detection
    if re.search(r'\bdef\b|\bclass\b|\bimport\b|\bfrom\b.*\bimport\b', code):
      return 'python'
    elif re.search(r'\bfunction\b|\bconst\b|\blet\b|\bvar\b', code):
      return 'javascript'
    elif re.search(r'\bpublic\s+class\b|\bprivate\b|\bprotected\b', code):
      return 'java'
    elif re.search(r'#include\s*<.*>|\bint\s+main\b', code):
      return 'c'
    
    return 'unknown'

  def _extract_code_structure(self, code: str, language: str) -> dict:
    """Extract structural information from code."""
    structure = {
      'language': language,
      'functions': [],
      'classes': [],
      'imports': [],
      'constants': [],
      'complexity_estimate': 'medium'
    }
    
    if language == 'python':
      try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
          if isinstance(node, ast.FunctionDef):
            structure['functions'].append({
              'name': node.name,
              'line': node.lineno,
              'args': [arg.arg for arg in node.args.args],
              'has_docstring': ast.get_docstring(node) is not None
            })
          elif isinstance(node, ast.ClassDef):
            structure['classes'].append({
              'name': node.name,
              'line': node.lineno,
              'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            })
          elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
              structure['imports'].extend([alias.name for alias in node.names])
            else:
              structure['imports'].append(f"from {node.module}" if node.module else "from .")
      except SyntaxError:
        # Handle syntax errors gracefully
        pass
    
    return structure

  async def call(self, part: ProcessorPart) -> AsyncIterable[ProcessorPart]:
    """Process a single code file and generate documentation.
    
    Args:
      part: ProcessorPart containing code to document
      
    Yields:
      ProcessorPart objects containing generated documentation
    """
    # Extract code content using the correct attributes
    try:
      if content_api.is_text(part.mimetype):
        code_content = part.text
      else:
        code_content = part.bytes.decode('utf-8', errors='ignore')
    except (ValueError, UnicodeDecodeError):
      # Skip parts that can't be processed as text
      return
    
    if not code_content.strip():
      return
        
    # Get filename from metadata if available
    filename = part.metadata.get('filename', '') if part.metadata else ''
    
    # Detect language and extract structure
    language = self._detect_language(code_content, filename)
    structure = self._extract_code_structure(code_content, language)
    
    # Create comprehensive analysis prompt with instructions
    analysis_prompt = f"""You are an expert code documentation generator. Your task is to analyze code and generate comprehensive, high-quality documentation.

DOCUMENTATION REQUIREMENTS:
- Format: {self._documentation_format}
- Docstring style: {self._docstring_style}
- Include examples: {self._include_examples}
- Include type hints: {self._include_type_hints}

ANALYSIS STEPS:
1. Detect programming language and conventions
2. Extract all functions, classes, methods, and modules
3. Analyze function signatures, parameters, and return values
4. Identify code purpose, algorithms, and design patterns
5. Generate appropriate documentation format

DOCUMENTATION STANDARDS:
- Write clear, concise descriptions
- Document all parameters and return values
- Include usage examples when requested
- Follow language-specific documentation conventions
- Add type information when available
- Note any important side effects or exceptions
- Include complexity analysis for algorithms when relevant

OUTPUT FORMAT:
- For markdown: Use proper headers, code blocks, and formatting
- For docstrings: Follow the specified style guide exactly
- Include table of contents for multi-file documentation
- Maintain consistent formatting throughout

Be thorough but concise. Focus on helping developers understand and use the code effectively.

---

Please generate comprehensive documentation for this {language} code:

FILENAME: {filename or 'unknown'}
CODE STRUCTURE: {structure}

CODE:
```{language}
{code_content}
```

Generate documentation that includes:
1. Overview and purpose
2. Function/method documentation
3. Class documentation (if applicable)
4. Usage examples (if requested)
5. Installation/setup instructions (if applicable)
6. API reference

Format the output as {self._documentation_format} with {self._docstring_style} style docstrings.
"""

    # Generate documentation using the model
    input_stream = streams.stream_content([ProcessorPart(analysis_prompt)])
    async for model_part in self._model(input_stream):
      # Create output part with proper metadata
      output_metadata = {
        'original_filename': filename,
        'language': language,
        'documentation_format': self._documentation_format,
        'structure': structure,
        'generated_by': 'CodeDocumentationProcessor'
      }
      
      yield ProcessorPart(
          model_part.text,  # Use .text instead of .data
          role='model',
          substream_name='documentation',
          mimetype='text/markdown' if self._documentation_format == 'markdown' else 'text/plain',
          metadata=output_metadata
      )


# Utility function for easy usage
def create_code_documentation_processor(
    api_key: str,
    **kwargs
) -> CodeDocumentationProcessor:
  """Create a CodeDocumentationProcessor with sensible defaults.
  
  Args:
    api_key: Google AI API key
    **kwargs: Additional configuration options
    
  Returns:
    Configured CodeDocumentationProcessor instance
  """
  return CodeDocumentationProcessor(api_key=api_key, **kwargs)


# Example usage function
async def document_code_file(
    file_path: str,
    api_key: str,
    output_format: str = "markdown"
) -> str:
  """Document a single code file.
  
  Args:
    file_path: Path to the code file
    api_key: Google AI API key  
    output_format: Documentation format
    
  Returns:
    Generated documentation as string
  """
  processor = CodeDocumentationProcessor(
      api_key=api_key,
      documentation_format=output_format
  )
  
  with open(file_path, 'r', encoding='utf-8') as f:
    code_content = f.read()
  
  code_part = ProcessorPart(
      code_content,
      metadata={'filename': file_path}
  )
  
  documentation_parts = []
  async for doc_part in processor.call(code_part):
    documentation_parts.append(doc_part.text)  # Use .text instead of .data
  
  return '\n'.join(documentation_parts)