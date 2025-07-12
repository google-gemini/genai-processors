"""
example_genai_langchain.py

Demonstrates how to use GenaiLangChainProcessor for:
  1. Agent-based Q&A with tools (weather, web search)
  2. Multimodal analysis combining text and image inputs
"""

import asyncio
import requests

from genai_processors.contrib.genai_langchain_streaming import GenAILangChainProcessor
from genai_processors import streams
from genai_processors.content_api import ProcessorContent, ProcessorPart
from langchain.tools import tool


@tool
def get_current_weather(location: str) -> str:
    """
    Placeholder tool: returns a fake weather report for a location.

    TODO: Swap out with a real weather-API integration.
    """
    return f"Weather in {location}: 72°F, sunny"


@tool
def search_web(query: str) -> str:
    """
    Placeholder tool: returns fake search results for a query.

    TODO: Replace with an actual web-search implementation.
    """
    return f"Search results for '{query}': …"


async def test_agent_processor():
    """
    Example: run a simple conversational agent that
    uses the weather and search tools, with memory enabled.
    """
    # 1. Define which tools the agent may call
    tools = [get_current_weather, search_web]

    # 2. Instantiate the processor in agent mode
    agent_processor = GenAILangChainProcessor.create_agent(
        tools=tools,
        streaming=True,
        enable_memory=True,
        temperature=0.7
    )

    # 3. Ask the first question (no prior context)
    print("=== First Question: Weather in London ===")
    q1 = ProcessorContent(
        ProcessorPart(
            "What's the weather in London right now?",
            mimetype="text/plain",
            role="user"
        )
    )
    async for part in agent_processor(streams.stream_content(q1)):
        print(part.text, end="", flush=True)
    print("\n")

    # 4. Ask a follow-up question (demonstrates memory usage)
    print("=== Follow-up Question: Weather in Paris ===")
    q2 = ProcessorContent(
        ProcessorPart(
            "What about Paris?",
            mimetype="text/plain",
            role="user"
        )
    )
    async for part in agent_processor(streams.stream_content(q2)):
        print(part.text, end="", flush=True)
    print("\nProcessing complete!\n")


async def test_multimodal_processor():
    """
    Example: send a text prompt and an image for analysis
    and have the model describe the image and suggest similar artworks.
    """
    # 1. Build the text prompt
    text_prompt = ProcessorContent(
        ProcessorPart(
            "Describe this image and suggest similar artworks.",
            mimetype="text/plain",
            role="user",
            metadata={"query_type": "art_analysis"}
        )
    )

    # 2. Fetch the image bytes (or load locally)
    image_url = "https://fastly.picsum.photos/id/11/2500/1667.jpg?hmac=xxjFJtAPgshYkysU_aqx2sZir-kIOjNR9vx0te7GycQ"
    response = requests.get(image_url)
    image_bytes = response.content

    # 3. Wrap the image bytes in a ProcessorPart
    image_prompt = ProcessorContent(
        ProcessorPart(
            image_bytes,
            mimetype="image/jpeg",
            role="user",
            metadata={"source": "web", "artist": "Unknown"}
        )
    )

    # 4. Combine text + image into one content stream
    combined = text_prompt + image_prompt

    # 5. Instantiate a plain LLM processor (no tools)
    mm_processor = GenAILangChainProcessor(
        model="gemini-2.0-flash",
        system_instruction="You are an art expert assistant.",
        streaming=True,
        temperature=0.7
    )

    # 6. Run the multimodal prompt
    print("=== Multimodal Processing ===")
    async for part in mm_processor(streams.stream_content(combined)):
        print(part.text, end="", flush=True)
    print("\nProcessing complete!\n")


def main():
    """Run both example scenarios in sequence."""
    asyncio.run(test_agent_processor())
    asyncio.run(test_multimodal_processor())


if __name__ == "__main__":
    main()
