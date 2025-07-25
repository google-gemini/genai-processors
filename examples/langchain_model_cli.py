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

"""Demonstrates how to use LangChainModel for multimodal and streaming inputs."""

import asyncio
from genai_processors import content_api
from genai_processors import streams
from genai_processors.contrib.langchain_model import LangChainModel
import httpx
import langchain_google_genai


async def fetch_image_bytes(url: str) -> bytes:
  """Fetch image bytes from a URL asynchronously."""
  async with httpx.AsyncClient() as client:
    response = await client.get(url)
    response.raise_for_status()
    return response.read()


async def test_multimodal_processor():
  """Does multimodal inference."""
  # 1. Build the text prompt and image as ProcessorParts
  image_url = 'https://fastly.picsum.photos/id/11/2500/1667.jpg?hmac=xxjFJtAPgshYkysU_aqx2sZir-kIOjNR9vx0te7GycQ'

  # Fetch image bytes
  image_bytes = await fetch_image_bytes(image_url)

  parts = [
      content_api.ProcessorPart(
          value='Describe this image and suggest similar artworks.',
          mimetype='text/plain',
          role='user',
          metadata={'query_type': 'art_analysis'},
      ),
      content_api.ProcessorPart(
          value=image_bytes,
          mimetype='image/jpeg',
          role='user',
          metadata={'source': 'web', 'artist': 'Unknown'},
      ),
  ]

  # 2. Combine parts into a single ProcessorContent
  content = content_api.ProcessorContent(parts)

  # 3. Create the LLM instance
  llm = langchain_google_genai.ChatGoogleGenerativeAI(model='gemini-1.5-flash')

  # 4. Instantiate the processor with a system instruction
  mm_processor = LangChainModel(
      llm=llm, system_instruction='You are an art expert assistant.'
  )

  # 5. Run the multimodal prompt, streaming back the response
  async for part in mm_processor(streams.stream_content(content)):
    print(part.text, end='', flush=True)

  print('\nProcessing complete!')


def main():
  """Run the example scenario."""
  asyncio.run(test_multimodal_processor())


if __name__ == '__main__':
  main()
