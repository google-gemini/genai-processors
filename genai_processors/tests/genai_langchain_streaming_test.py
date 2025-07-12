import unittest
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool

from genai_processors import content_api
from genai_processors.contrib.genai_langchain_streaming import GenAILangChainProcessor

class MockLLM:
    """Mock ChatGoogleGenerativeAI for testing"""
    def __init__(self):
        self.ainvoke_return = AIMessage(content="Mock response")
        self.streaming_responses = [
            AIMessage(content="Stream"),
            AIMessage(content=" response")
        ]
    
    async def ainvoke(self, *args, **kwargs):
        return self.ainvoke_return
    
    async def astream(self, *args, **kwargs):
        for response in self.streaming_responses:
            yield response

class MockAgentExecutor:
    """Simple mock AgentExecutor without inheritance"""
    async def ainvoke(self, *args, **kwargs):
        return {"output": "Agent response"}
    
    async def astream_events(self, *args, **kwargs):
        yield {
            "event": "on_llm_stream",
            "data": {"chunk": AIMessage(content="Tool")}
        }
        yield {
            "event": "on_tool_start",
            "name": "real_tool"
        }

# Define real LangChain tools for testing
@tool
def real_tool(input: str) -> str:
    """Real tool for testing."""
    return f"Processed: {input}"

def async_iter(parts):
    """Convert list of parts to async iterable"""
    async def gen():
        for part in parts:
            yield part
    return gen()

async def collect_async_iter(async_iterable):
    """Collect results from async iterable into list"""
    return [part async for part in async_iterable]

class GenAILangChainProcessorTest(unittest.IsolatedAsyncioTestCase):
    async def test_non_agent_non_streaming(self):
        """Test direct LLM call in non-streaming mode"""
        # Setup processor
        proc = GenAILangChainProcessor(
            streaming=False,
            enable_memory=False
        )
        
        # Mock LLM
        mock_llm = MockLLM()
        proc.llm = mock_llm
        
        # Test input
        input_parts = [content_api.ProcessorPart("Hello", mimetype="text/plain")]
        
        # Process
        output = await collect_async_iter(
            proc.call(async_iter(input_parts)))
        
        # Verify
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].text, "Mock response")

    async def test_non_agent_streaming(self):
        """Test direct LLM call in streaming mode"""
        # Setup processor
        proc = GenAILangChainProcessor(
            streaming=True,
            enable_memory=False
        )
        
        # Mock LLM
        mock_llm = MockLLM()
        proc.llm = mock_llm
        
        # Test input
        input_parts = [content_api.ProcessorPart("Hello", mimetype="text/plain")]
        
        # Process
        output = await collect_async_iter(
            proc.call(async_iter(input_parts)))
        
        # Verify streaming chunks
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0].text, "Stream")
        self.assertEqual(output[1].text, " response")

    async def test_agent_non_streaming(self):
        """Test agent execution in non-streaming mode"""
        # Setup processor with real tool
        proc = GenAILangChainProcessor(
            tools=[real_tool],
            streaming=False,
            enable_memory=False
        )
        
        # Mock agent executor
        mock_executor = MockAgentExecutor()
        proc.agent_executor = mock_executor
        
        # Test input
        input_parts = [content_api.ProcessorPart("Use tool", mimetype="text/plain")]
        
        # Process
        output = await collect_async_iter(
            proc.call(async_iter(input_parts)))
        
        # Verify
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].text, "Agent response")

    async def test_agent_streaming(self):
        """Test agent execution in streaming mode"""
        # Setup processor with real tool
        proc = GenAILangChainProcessor(
            tools=[real_tool],
            streaming=True,
            enable_memory=False
        )
        
        # Mock agent executor
        mock_executor = MockAgentExecutor()
        proc.agent_executor = mock_executor
        
        # Test input
        input_parts = [content_api.ProcessorPart("Use tool", mimetype="text/plain")]
        
        # Process
        output = await collect_async_iter(
            proc.call(async_iter(input_parts)))
        
        # Verify streaming events
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0].text, "Tool")
        self.assertEqual(output[1].text, "Using tool: real_tool")
        self.assertEqual(output[1].metadata["event"], "tool_start")

    async def test_multimodal_conversion(self):
        """Test conversion of multimodal content to LangChain messages"""
        proc = GenAILangChainProcessor()
        
        # Create test content with different types
        content = content_api.ProcessorContent(
            content_api.ProcessorPart("Text input", mimetype="text/plain"),
            content_api.ProcessorPart(b"image_data", mimetype="image/png"),
            content_api.ProcessorPart(b"audio_data", mimetype="audio/mpeg"),
        )
        
        # Convert to LangChain messages
        messages = proc._convert_to_langchain_messages(content)
        
        # Verify structure - each part becomes a separate message
        self.assertEqual(len(messages), 3)
        self.assertIsInstance(messages[0], HumanMessage)
        self.assertIsInstance(messages[1], HumanMessage)
        self.assertIsInstance(messages[2], HumanMessage)
        
        # Verify content types
        self.assertEqual(messages[0].content[0]["type"], "text")
        self.assertEqual(messages[1].content[0]["type"], "image_url")
        self.assertEqual(messages[2].content[0]["type"], "audio")

    async def test_memory_functionality(self):
        """Test conversation memory across multiple turns"""
        proc = GenAILangChainProcessor(enable_memory=True)
        
        # Mock LLM
        mock_llm = MockLLM()
        proc.llm = mock_llm
        
        # First turn
        input1 = [content_api.ProcessorPart("First message", role="user")]
        await collect_async_iter(proc.call(async_iter(input1)))
        
        # Second turn
        input2 = [content_api.ProcessorPart("Follow-up", role="user")]
        output = await collect_async_iter(proc.call(async_iter(input2)))
        
        # Verify memory has both messages
        self.assertEqual(len(proc.conversation_history), 2)
        self.assertEqual(proc.conversation_history[0].text, "First message")
        self.assertEqual(proc.conversation_history[1].text, "Follow-up")

    async def test_agent_factory_method(self):
        """Test agent creation via factory method"""
        proc = GenAILangChainProcessor.create_agent(tools=[real_tool])
        
        # Verify agent configuration
        self.assertTrue(hasattr(proc, 'agent_executor'))
        self.assertIn(real_tool, proc.tools)
        self.assertIn("helpful AI assistant", proc.system_instruction)

if __name__ == "__main__":
    unittest.main()




