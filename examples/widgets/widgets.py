# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
"""Processor that produces richly formatted responses using widgets.

This is the implementation, use widgets_ais.py to run the server.

This demo demonstrates how to enrich model output with UI elements (widgets)
using async tools. Key concepts include:

*   **Async Tools**: Widgets (e.g., ImageGenerator, PlotGenerator) are
    registered as tools that do not block the model.
*   **Tiered Rendering**:
    * Model decides the widget/content.
    * Tool renders to HTML/low-level representation.
    * UI displays the content.
*   **Direct Streaming**: Tools stream responses directly to the UI using the
    reserved `ui` substream, bypassing the model to avoid context pollution
    and head-of-line blocking.

See README.md for more details.
"""

from collections.abc import AsyncIterable

from genai_processors import content_api
from genai_processors import processor
from genai_processors.core import function_calling
from genai_processors.core import genai_model
from genai_processors.core import realtime
from google.genai import types as genai_types

MODEL = 'gemini-3-flash-preview'
IMAGE_MODEL = 'gemini-2.5-flash-image'


# System Instruction for the root processor.
SYSTEM_INSTRUCTION = [
    'You are a Scientist agent. You help students by explaining how things'
    ' work.'
]


class ImageGenerator:
  """Image generator from a description."""

  def __init__(self, api_key: str):
    self._api_key = api_key
    self._image_generator = genai_model.GenaiModel(
        api_key=self._api_key,
        model_name=IMAGE_MODEL,
    )

  async def create_image_from_description(
      self,
      description: str,
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    """Creates an image from a description.

    The description should be detailed enough to give all the information
    needed to create the image, including style and tone, shapes, shades, etc.
    The image will be visible to the user only, but not you.

    Args:
      description: The description of the image to create.

    Yields:
      The image parts.
    """

    # Notify the model that all is done. Actual output goes directly to the UI.
    # NOTE: we should make this a property of the tool.
    yield content_api.ProcessorPart.from_function_response(
        response='Image generated', will_continue=False
    )

    content = content_api.ProcessorContent([
        (
            'Generate an illustration based on the following description. '
            'You may produce a short clarification after the image.\n'
        ),
        description,
    ])

    async for part in self._image_generator(content):
      # We use ui substream to send it directly to the UI.
      part.substream_name = 'ui'
      yield part


class PlotGenerator:
  """Plot generator from a description."""

  def __init__(self, api_key: str):
    self._api_key = api_key
    self._plot_generator = genai_model.GenaiModel(
        api_key=self._api_key,
        model_name=MODEL,
    )

  async def create_plot_from_description(
      self,
      description: str,
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    """Creates a plot from a description.

    The description should be detailed enough to give all the information
    needed to create the plot. The plot will be visible to the user only,
    but not you.

    Args:
      description: The description of the plot to create.

    Yields:
      The plot HTML.
    """

    # Notify the model that all is done. Actual output goes directly to the UI.
    yield content_api.ProcessorPart.from_function_response(
        response='Plot generated', will_continue=False
    )

    content = content_api.ProcessorContent([
        (
            'Generate a standalone HTML file containing an SVG visualization'
            ' of the data described below. Use inline SVG code directly'
            ' embedded in the HTML. Do NOT use any external JavaScript'
            ' libraries (no Bokeh, Plotly, D3, Chart.js, etc.). Output ONLY'
            ' the raw HTML code. Do NOT output markdown code blocks. Do NOT'
            ' output any explanation.\n'
            '\n'
            'The SVG should be self-contained and styled appropriately.'
            ' Include proper axes, labels, and legends as needed. Page height'
            ' must match the SVG height and SVG should take up all the width.'
            ' Do not leave big margins on any side of the SVG.\n'
        ),
        description,
    ])

    # By streaming the output, we allow the browser to render it as it is being
    # generated.
    async for part in self._plot_generator(content):
      # This is a bit ugly, but content_api.ProcessorPart.from_function_response
      # would render it as text instead of inline_data with
      # mime_type='text/html'.
      yield content_api.ProcessorPart(
          genai_types.Part(
              function_response=genai_types.FunctionResponse(
                  parts=[
                      genai_types.FunctionResponsePart(
                          inline_data=genai_types.FunctionResponseBlob(
                              data=part.text.encode(),
                              mime_type='text/html',
                          )
                      )
                  ],
              )
          ),
          # We use ui substream to send it directly to the UI.
          substream_name='ui',
      )


def create_dr_widget(
    api_key: str,
) -> processor.Processor:
  """Initializes the processor."""
  image_gen = ImageGenerator(api_key)
  plot_gen = PlotGenerator(api_key)
  turn_processor = genai_model.GenaiModel(
      api_key=api_key,
      model_name=MODEL,
      generate_content_config=genai_types.GenerateContentConfig(
          system_instruction=SYSTEM_INSTRUCTION,
          # We will be handling tool calls on the client side.
          automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
              disable=True
          ),
          tools=[
              image_gen.create_image_from_description,
              plot_gen.create_plot_from_description,
          ],
      ),
  )
  fc_processor = function_calling.FunctionCalling(
      model=realtime.LiveProcessor(turn_processor=turn_processor),
      fns=[
          image_gen.create_image_from_description,
          plot_gen.create_plot_from_description,
      ],
      is_bidi_model=True,
  )

  return fc_processor
