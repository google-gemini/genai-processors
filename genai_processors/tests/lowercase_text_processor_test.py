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
"""Tests for lowercase_text_processor."""

import pytest
from genai_processors import content_api
from genai_processors import processor
from genai_processors.contrib import lowercase_text_processor


class TestLowercaseTextProcessor:
  """Test cases for LowercaseTextProcessor."""
  
  def test_basic_text_conversion(self):
    """Test that text is converted to lowercase."""
    p = lowercase_text_processor.LowercaseTextProcessor()
    
    input_parts = [
        content_api.ProcessorPart(
            'Hello World! This is A TEST.',
            mimetype='text/plain',
        ),
        content_api.ProcessorPart(
            'ANOTHER Example String.',
            mimetype='text/plain',
        ),
    ]
    
    output = processor.apply_sync(p, input_parts)
    result_text = content_api.as_text(output)
    
    assert result_text == 'hello world! this is a test.another example string.'
  
  def test_mixed_content_types(self):
    """Test that non-text parts are passed through unchanged."""
    p = lowercase_text_processor.LowercaseTextProcessor()
    
    input_parts = [
        content_api.ProcessorPart(
            'Hello WORLD!',
            mimetype='text/plain',
        ),
        content_api.ProcessorPart(
            b'some binary data',
            mimetype='application/octet-stream',
        ),
        content_api.ProcessorPart(
            'Another TEXT String',
            mimetype='text/plain',
        ),
    ]
    
    output = list(processor.apply_sync(p, input_parts))
    
    # Check that text parts are converted to lowercase
    assert output[0].text == 'hello world!'
    assert output[0].mimetype == 'text/plain'
    
    # Check that binary data is unchanged
    assert output[1].bytes == b'some binary data'
    assert output[1].mimetype == 'application/octet-stream'
    
    # Check that second text part is converted
    assert output[2].text == 'another text string'
    assert output[2].mimetype == 'text/plain'
  
  def test_empty_text(self):
    """Test handling of empty text."""
    p = lowercase_text_processor.LowercaseTextProcessor()
    
    input_parts = [
        content_api.ProcessorPart(
            '',
            mimetype='text/plain',
        ),
    ]
    
    output = processor.apply_sync(p, input_parts)
    result_text = content_api.as_text(output)
    
    assert result_text == ''
  
  def test_match_function(self):
    """Test that the match function works correctly."""
    p = lowercase_text_processor.LowercaseTextProcessor()
    
    # Test match function directly
    text_part = content_api.ProcessorPart('test', mimetype='text/plain')
    binary_part = content_api.ProcessorPart(b'test', mimetype='application/octet-stream')
    
    text_match = p.match(text_part)
    binary_match = p.match(binary_part)
    
    assert text_match == True
    assert binary_match == False
  
  def test_metadata_preservation(self):
    """Test that metadata and other attributes are preserved."""
    p = lowercase_text_processor.LowercaseTextProcessor()
    
    original_metadata = {'key': 'value'}
    input_parts = [
        content_api.ProcessorPart(
            'TEST TEXT',
            mimetype='text/plain',
            metadata=original_metadata,
            substream_name='test_stream',
            role='user',
        ),
    ]
    
    output = list(processor.apply_sync(p, input_parts))
    
    assert output[0].text == 'test text'
    assert output[0].metadata == original_metadata
    assert output[0].substream_name == 'test_stream'
    assert output[0].mimetype == 'text/plain'
    assert output[0].role == 'user'
  
  def test_special_characters_and_numbers(self):
    """Test handling of special characters and numbers."""
    p = lowercase_text_processor.LowercaseTextProcessor()
    
    input_parts = [
        content_api.ProcessorPart(
            'Hello 123! @#$%^&*() WORLD_test',
            mimetype='text/plain',
        ),
    ]
    
    output = processor.apply_sync(p, input_parts)
    result_text = content_api.as_text(output)
    
    assert result_text == 'hello 123! @#$%^&*() world_test'
