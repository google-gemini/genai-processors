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
"""Processor generating live illustrations from an audio stream."""

import asyncio
from collections.abc import AsyncIterable
import time

from genai_processors import content_api
from genai_processors import context as context_lib
from genai_processors import mime_types
from genai_processors import processor
from genai_processors.core import audio
from genai_processors.core import function_calling
from genai_processors.core import genai_model
from genai_processors.core import realtime
from google.genai import types as genai_types

# Model to use for the live api processor.
MODEL_LISTEN = 'gemini-3-flash-preview'

MODEL_IMAGE = 'gemini-2.5-flash-image'


# SI for the live processor.
SYSTEM_INSTRUCTION = [
    '### Role',
    (
        'You are a visual storytelling agent. Your goal is to create compelling'
        ' illustrations based on the conversation history. First transcribe and'
        ' print the part of the story narrated since your last turn (if the'
        ' last sentence is not finished, delay transcribing it till the next'
        ' turn. Do not add ellipsis). After transcribing move on to'
        ' illustrating. If there is not enough context e.g. no story has been'
        ' told yet, wait and do not illustrate. You can not ask for'
        ' clarifications.'
    ),
    '',
    '### Tool Usage & Workflow',
    (
        'You have access to two tools: `create_concept_art` and '
        '`create_image_from_description`. strictly follow this workflow:'
    ),
    '',
    '**Step 1: Asset Preparation (Concept Art)**',
    (
        'Before generating a final scene or an image, analyze if it involves a'
        ' **new** character, location, key object or key concept that are'
        ' likely to appear on more than one illustration.'
    ),
    '- IF new entities are present: Use `create_concept_art`.',
    (
        '  - Style: For characters use a plain background. Include front, '
        'side, and back views on a single image.'
    ),
    (
        '- Consistency: You must first generate concept art for key elements '
        'that are likely to appear in more than one illustration. On average '
        'there should be more illustrations than concept arts though.'
    ),
    '',
    '**Step 2: Final Illustration**',
    (
        'Use `create_image_from_description` to generate the illustration for'
        ' the next plot point / statement. Pass the description of what to draw'
        ' and the names of the relevant concept arts (if any) to the tool to'
        ' ensure consistency. It is fine to delay coverage of the most recent'
        ' events to avoid missing important details.'
    ),
    '',
    '### Generation Criteria',
    'Only generate a final image if ALL the following are true:',
    '1. The moment is significant, interesting, fun, or epic.',
    '2. It illustrates a key plot point from the recent history.',
    (
        '3. You have not already created or asked to create a similar image. '
        'The new image must cover the next plot point and thus be '
        'significantly different.'
    ),
    '4. All necessary concept arts exist.',
    '',
    (
        'If you cannot create an image (e.g., policy restrictions or technical '
        'error), output a clear message explaining why.'
    ),
]


@processor.part_processor_function
async def unwrap_function_response(
    part: content_api.ProcessorPart,
) -> AsyncIterable[content_api.ProcessorPartTypes]:
  """Returns the content of a function response."""
  if not part.function_response:
    if context_lib.is_reserved_substream(part.substream_name):
      part.role = 'model'
    yield part
    return

  if part.function_response.parts:
    for fn_part in part.function_response.parts:
      if fn_part.inline_data:
        # For media content, unwrap the inline data and yield a ProcessorPart.
        yield content_api.ProcessorPart(
            fn_part.inline_data.data,
            mimetype=fn_part.inline_data.mime_type,
            role='tool',
            substream_name=part.function_response.name,
            metadata=part.metadata,
        )
      else:
        raise RuntimeError(
            f'Function part {fn_part} did not contain inline data.'
        )
    return

  if 'result' in part.function_response.response:
    part_content = part.function_response.response['result']
  elif 'output' in part.function_response.response:
    part_content = part.function_response.response['output']
  else:
    part_content = str(part.function_response.response)
  yield content_api.ProcessorPart(
      part_content,
      substream_name=part.substream_name,
      role='tool',
      mimetype=part.mimetype,
      metadata=part.metadata,
  )


class ScheduleEndOfTurns(processor.Processor):
  """Schedules end of turns."""

  def __init__(self, period_sec: int = 20):
    if period_sec <= 0:
      raise ValueError(
          f'period_sec must be positive. current value: {period_sec}'
      )
    self._period_sec = period_sec
    self._turn_requested = False

    @processor.processor_function
    async def track_turn_start(
        content: processor.ProcessorStream,
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
      async for part in content:
        yield part
      self._turn_requested = True

    self.track_turn_start = track_turn_start

    @processor.processor_function
    async def track_turn_end(
        content: processor.ProcessorStream,
    ) -> AsyncIterable[content_api.ProcessorPartTypes]:
      try:
        async for part in content:
          yield part
      finally:
        self._turn_requested = False

    self.track_turn_end = track_turn_end

  async def call(
      self,
      content: processor.ProcessorStream,
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    """Adds and end of turn every `period_sec` seconds. Removes EoS and SoS."""
    last_eot_timestamp = time.time()

    async for part in content:
      # Only pass through text corresponding to the transcription of the user
      # speech or audio signals, or audio signals.
      if content_api.is_text(part.mimetype) and not content_api.is_end_of_turn(
          part
      ):
        yield part
      if content_api.is_audio(part.mimetype):
        yield part

      now = time.time()
      if (
          now - last_eot_timestamp > self._period_sec
          and not self._turn_requested
      ):
        yield content_api.END_OF_TURN
        last_eot_timestamp = now


class ImageGenerator:
  """Image generator from a description."""

  def __init__(self, api_key: str, system_instruction: str | None = None):
    self._system_instruction = system_instruction
    self._api_key = api_key
    self._image_generator = genai_model.GenaiModel(
        api_key=self._api_key,
        model_name=MODEL_IMAGE,
    )
    self._concept_arts: dict[
        str, asyncio.Future[content_api.ProcessorContent]
    ] = {}

  async def create_concept_art(
      self,
      name: str,
      description: str,
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    """Creates a concept art from a description and saves it for later use.

    Args:
      name: The name of the concept art to save.
      description: The description of the concept art to create.

    Yields:
      The image parts.
    """
    self._concept_arts[name] = asyncio.Future()

    concept = content_api.ProcessorContent()
    try:
      yield content_api.ProcessorPart.from_function_response(
          name='create_concept_art`',
          response=f'\nGenerating concept art for [{name}]...\n',
          scheduling='SILENT',
          role='model',
      )

      prompt = content_api.ProcessorContent(description)
      if self._system_instruction:
        prompt += [
            (
                '\nConsider the following instructions as the top priorities '
                'when generating the image:\n'
            ),
            self._system_instruction,
        ]

      async for part in self._image_generator(prompt):
        if not part.part.thought:
          concept += part
      self._concept_arts[name].set_result(concept)
    except Exception as e:  # pylint: disable=broad-except
      self._concept_arts[name].set_exception(e)

    # We yield generated parts as one response after self._concept_arts[name]
    # as phrases like "Here's an image of..." produced by the image generation
    # model can be perceived as if the art is already available.
    yield content_api.ProcessorPart.from_function_response(
        name='create_concept_art',
        response=[
            f'\nCreated concept art for [{name}]:\n',
            concept,
        ],
        role='model',
    )

  async def create_image_from_description(
      self,
      description: str,
      concept_arts: list[str] | None = None,
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    """Creates an image from a description.

    The description should be detailed enough to give all the information
    needed to create the image, including style and tone, shapes, shades, etc.

    Args:
      description: The description of the image to create.
      concept_arts: A list of relevant concept art names to use for image
        generation. All of them must have been requested previously using
        `create_concept_art`. You can pass None if no concept art is relevant
        for this image.

    Yields:
      The image parts.
    """
    yield content_api.ProcessorPart.from_function_response(
        name='create_image_from_description',
        response=(
            f'Generating image from description: {description} using concept'
            f' arts: {concept_arts}'
        ),
        scheduling='SILENT',
        role='model',
    )

    content = content_api.ProcessorContent()
    if concept_arts:
      content += 'You can use the following concept arts as references:\n'
      for name in concept_arts:
        try:
          content += [name, await self._concept_arts[name]]
        except KeyError as exc:
          raise ValueError(f'Concept art "{name}" not found.') from exc

    content += 'Generate an illustration based on the following description:\n'
    content += description

    if self._system_instruction:
      content += [
          (
              '\nConsider the following instructions as the top priorities when'
              ' generating the image:\n'
          ),
          self._system_instruction,
      ]

    async for part in self._image_generator(content):
      yield content_api.ProcessorPart.from_function_response(
          name='create_image_from_description',
          response=part,
          scheduling='SILENT',
          role='model',
      )


@processor.create_filter
def hide_uninteresting_parts(part: content_api.ProcessorPart) -> bool:
  """Removes parts that are not informative for the user."""
  # This function is specific to the needs of the illustrator UI and the
  # underlying model so it can afford to handle specific cases.
  if mime_types.is_text(part.mimetype):
    if not part.text.strip():
      # Filter out empty parts. Model outputs \n after function calls.
      return False
    if part.text.startswith('Running in background'):
      # This part is for the model, but only clutters the UI.
      return False
  return True


def create_live_illustrator(
    api_key: str,
    system_instruction: str | None = None,
    image_period_sec: int = 20,
) -> processor.Processor:
  """Initializes the processor."""
  image_gen = ImageGenerator(api_key, system_instruction)
  turn_processor = genai_model.GenaiModel(
      api_key=api_key,
      model_name=MODEL_LISTEN,
      generate_content_config=genai_types.GenerateContentConfig(
          system_instruction=SYSTEM_INSTRUCTION,
      ),
  )
  end_of_turns_scheduler = ScheduleEndOfTurns(period_sec=image_period_sec)
  fc_processor = function_calling.FunctionCalling(
      # It is important here to receive a stream of content that generates
      # parts regularly at a good frequency. The main loop inside live processor
      # is cadenced by a loop over the input stream. That's why we send the
      # audio signals up to the turn_processor, so that all async loop move
      # forward at a high frequency.
      model=realtime.LiveProcessor(
          turn_processor=(
              # We filter out function responses to speed
              processor.create_filter(lambda x: not x.function_response)
              + audio.AudioToWav()
              + end_of_turns_scheduler.track_turn_start
              + turn_processor
              + end_of_turns_scheduler.track_turn_end
          ),
          # We trigger a new turn when the user is done talking. Because we
          # filter end of speech parts before passing to the model, this means
          # the model will only get triggered when we send an end of turn. This
          # is done by the ScheduleEndOfTurns processor.
          trigger_model_mode=realtime.AudioTriggerMode.END_OF_SPEECH,
      ),
      fns=[
          image_gen.create_image_from_description,
          image_gen.create_concept_art,
      ],
      is_bidi_model=True,
  )
  return (
      end_of_turns_scheduler
      + fc_processor
      + unwrap_function_response
      + hide_uninteresting_parts
  )
