import base64
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from genai_processors import content_api
from genai_processors import mcp
from mcp import types as mcp_types

# Empty input schema for test tools that take no required args
_EMPTY_SCHEMA = {'type': 'object', 'properties': {}}


class McpToolsTest(parameterized.TestCase, unittest.IsolatedAsyncioTestCase):
  """Tests for MCP tool callable factories."""

  def setUp(self):
    super().setUp()
    self.mock_session = mock.AsyncMock()

  async def test_create_mcp_tool_calls_session(self):
    """Test MCP tool is called with kwargs and returns proper text result."""
    self.mock_session.call_tool.return_value = mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type='text', text='Result')],
        isError=False,
    )
    tool = mcp_types.Tool(
        name='test_tool', description='Test tool', inputSchema=_EMPTY_SCHEMA
    )

    callable_fn = mcp.create_mcp_tool(self.mock_session, tool)
    result = await callable_fn(arg1='value1', arg2=42)

    self.mock_session.call_tool.assert_called_once_with(
        'test_tool', {'arg1': 'value1', 'arg2': 42}
    )
    self.assertIsNotNone(result.function_response)
    self.assertEqual(result.function_response.response['result'], 'Result')

  async def test_create_mcp_tool_error_handling(self):
    """Test that McpToolError is raised when isError=True."""
    self.mock_session.call_tool.return_value = mcp_types.CallToolResult(
        content=[
            mcp_types.TextContent(type='text', text='Something went wrong')
        ],
        isError=True,
    )
    tool = mcp_types.Tool(
        name='failing', description='A failing tool', inputSchema=_EMPTY_SCHEMA
    )

    callable_fn = mcp.create_mcp_tool(self.mock_session, tool)

    with self.assertRaises(mcp.McpToolError) as ctx:
      await callable_fn()

    self.assertEqual(ctx.exception.tool_name, 'failing')
    self.assertIn('Something went wrong', str(ctx.exception))

  @parameterized.parameters([
      (
          mcp_types.ImageContent(
              type='image',
              data=base64.b64encode(b'\x89PNG\r\n\x1a\n').decode('utf-8'),
              mimeType='image/png',
          ),
          content_api.ProcessorPart(
              value=b'\x89PNG\r\n\x1a\n', mimetype='image/png'
          ),
      ),
      (
          mcp_types.AudioContent(
              type='audio',
              data=base64.b64encode(b'\x00\x01\x02\x03\x04\x05').decode(
                  'utf-8'
              ),
              mimeType='audio/wav',
          ),
          content_api.ProcessorPart(
              value=b'\x00\x01\x02\x03\x04\x05', mimetype='audio/wav'
          ),
      ),
      (
          mcp_types.EmbeddedResource(
              type='resource',
              resource=mcp_types.TextResourceContents(
                  uri='file:///test.txt',
                  text='Hello from resource',
                  mimeType='text/plain',
              ),
          ),
          content_api.ProcessorPart(
              value='Hello from resource', mimetype='text/plain'
          ),
      ),
      (
          mcp_types.EmbeddedResource(
              type='resource',
              resource=mcp_types.BlobResourceContents(
                  uri='file:///test.pdf',
                  blob=base64.b64encode(b'\x89PDF-1.4').decode('utf-8'),
                  mimeType='application/pdf',
              ),
          ),
          content_api.ProcessorPart(
              value=b'\x89PDF-1.4', mimetype='application/pdf'
          ),
      ),
      (
          mcp_types.ResourceLink(
              type='resource_link',
              uri='https://example.com/resource',
              name='Example Resource',
          ),
          content_api.ProcessorPart(
              value='https://example.com/resource', mimetype='text/uri'
          ),
      ),
  ])
  async def test_create_mcp_tool_image_content(self, content, expected):
    """Test that ImageContent is returned inside function_response parts."""
    self.mock_session.call_tool.return_value = mcp_types.CallToolResult(
        content=[content],
        isError=False,
    )
    tool = mcp_types.Tool(
        name='test_tool', description='Test tool', inputSchema=_EMPTY_SCHEMA
    )

    callable_fn = mcp.create_mcp_tool(self.mock_session, tool)
    result = await callable_fn()
    self.assertEqual(
        result,
        content_api.ProcessorPart.from_function_response(
            name='test_tool',
            response=expected,
        ),
    )

  async def test_create_mcp_tool_multiple_content_blocks(self):
    """Test that multiple content blocks are concatenated in function_response."""
    self.mock_session.call_tool.return_value = mcp_types.CallToolResult(
        content=[
            mcp_types.TextContent(type='text', text='Part 1'),
            mcp_types.TextContent(type='text', text='Part 2'),
        ],
        isError=False,
    )
    tool = mcp_types.Tool(
        name='multi',
        description='Returns multiple parts',
        inputSchema=_EMPTY_SCHEMA,
    )

    callable_fn = mcp.create_mcp_tool(self.mock_session, tool)
    result = await callable_fn()
    self.assertEqual(
        result,
        content_api.ProcessorPart.from_function_response(
            name='multi',
            response=content_api.ProcessorContent(
                content_api.ProcessorPart('Part 1Part 2')
            ),
        ),
    )

  def test_create_mcp_tool_metadata_introspection(self):
    """Test that callable has correct name and docstring for FunctionCalling."""
    tool = mcp_types.Tool(
        name='my_special_tool',
        description='Does something special',
        inputSchema=_EMPTY_SCHEMA,
    )

    callable_fn = mcp.create_mcp_tool(self.mock_session, tool)

    self.assertEqual(callable_fn.__name__, 'my_special_tool')
    self.assertEqual(callable_fn.__doc__, 'Does something special')

  def test_create_mcp_tool_default_docstring(self):
    """Test that callable uses default docstring when description is None."""
    tool = mcp_types.Tool(
        name='no_desc_tool',
        description=None,
        inputSchema=_EMPTY_SCHEMA,
    )

    callable_fn = mcp.create_mcp_tool(self.mock_session, tool)

    self.assertEqual(callable_fn.__name__, 'no_desc_tool')
    self.assertEqual(callable_fn.__doc__, 'Calls the no_desc_tool MCP tool.')

  async def test_create_mcp_tool_argument_mapping_explicit(self):
    """Explicitly verify that kwargs are properly mapped to MCP arguments."""
    self.mock_session.call_tool.return_value = mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type='text', text='OK')],
        isError=False,
    )
    tool = mcp_types.Tool(
        name='get_weather',
        description='Get weather for a location',
        inputSchema={
            'type': 'object',
            'properties': {
                'location': {'type': 'string'},
                'units': {'type': 'string'},
            },
            'required': ['location'],
        },
    )

    callable_fn = mcp.create_mcp_tool(self.mock_session, tool)
    await callable_fn(location='San Francisco', units='celsius')

    self.mock_session.call_tool.assert_called_once_with(
        'get_weather', {'location': 'San Francisco', 'units': 'celsius'}
    )


if __name__ == '__main__':
  absltest.main()
