"""Check the actual structure of genai_processors."""

import os
import ast

def find_base_classes():
    """Find processor base classes in the codebase."""
    core_path = "genai_processors/core"
    
    for filename in os.listdir(core_path):
        if filename.endswith('.py') and filename != '__init__.py':
            filepath = os.path.join(core_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'class' in content and ('Processor' in content or 'Part' in content):
                        print(f"\n{filename}:")
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef):
                                bases = [b.id if isinstance(b, ast.Name) else str(b) for b in node.bases]
                                print(f"  - {node.name} (bases: {bases})")
            except Exception as e:
                print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    find_base_classes()
