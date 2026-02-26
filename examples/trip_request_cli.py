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

"""Command line interface to process a trip request.

We use Gemini flash-lite to formalize freeform trip request into the dates and
destination. Then we use a second model to compose the trip itinerary.

This simple example shows how we can reduce perceived latency by running a fast
model to validate and acknowledge user request while the good but slow model is
handling it.

The approach from this example also can be used as a defense mechanism against
prompt injections. The first model without tool access formalizes the request
into the TripRequest dataclass. The attack surface is significantly reduced by
the narrowness of the output format and lack of tools. Then a second model is
run on this cleanup up input.

Before running this script, ensure the `GOOGLE_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

Usage:
  python3 trip_request_cli.py
"""
import asyncio
from collections.abc import AsyncIterable
import datetime
import os

import dataclasses_json
from genai_processors import content_api
from genai_processors import processor
from genai_processors import switch
from genai_processors.core import genai_model
from genai_processors.core import preamble
from google.genai import types as genai_types
from pydantic import dataclasses

# You need to define the API key in the environment variables.
API_KEY = os.environ['GOOGLE_API_KEY']


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class TripRequest:
  """A trip request required for GenAI models to generate structured output."""

  start_date: str
  end_date: str
  destination: str
  error: str

  def info(self) -> str:
    """Returns a string representation to be used in prompts."""
    return (
        '\nTrip information:\n'
        f'Start date: {self.start_date}\n'
        f'End date: {self.end_date}\n'
        f'Destination: {self.destination}\n'
    )


# A processor can be easily defined as a function with a dedicated decorator.
# This is the recommended way to define stateless processors.
@processor.part_processor_function
async def process_json_output(
    part: content_api.ProcessorPart,
) -> AsyncIterable[content_api.ProcessorPart]:
  """Process the json output of a GenAI model."""
  trip_request = part.get_dataclass(TripRequest)
  if trip_request.error:
    yield content_api.ProcessorPart(
        trip_request.error,
        substream_name='error',
    )
  else:
    yield content_api.ProcessorPart(trip_request.info())


async def run_trip_request() -> None:
  """Prepapre a trip plan."""

  # First processor extracts a json trip request from the user input.
  # We need a json dataclass (we use the wrapper from pydantic) to parse the
  # json output of the model. We add the current date to the prompt to make
  # sure the model uses the current date.
  extract_trip_request = preamble.Suffix(
      content_factory=lambda: f'Today is: {datetime.date.today()}'
  ) + genai_model.GenaiModel(
      api_key=API_KEY,
      model_name='gemini-2.5-flash',
      generate_content_config=genai_types.GenerateContentConfig(
          system_instruction=(
              'You are a travel agent. You are given a trip request from a'
              ' user. You need to check if the user provided all necessary'
              ' information. If the user request is missing any'
              ' information, you need to return an error message. If the'
              ' user request is complete, you need to return the user'
              ' request with the start date, end date and the destination.'
          ),
          response_schema=TripRequest,
          response_mime_type='application/json',
      ),
  )
  # Second processor generates a trip itinerary based on a valid trip request.
  generate_trip = genai_model.GenaiModel(
      api_key=API_KEY,
      # NOTE: To reduce cost of running the demo we use the flash model.
      # The real application would use a better but slower thinking model.
      # The perceived latency of that model would be hidden by the fast answer
      # from extract_trip_request and acknowledging to the user that we've
      # started planning the trip.
      model_name='gemini-2.5-flash-lite',
      generate_content_config=genai_types.GenerateContentConfig(
          system_instruction=(
              'You are a travel agent. You are given a trip request from a user'
              ' with dates and destination. Plan a trip with hotels and'
              ' activities. Split the plan into daily section. Plan one'
              ' activity per 1/2 day max.'
          ),
          # Ground with Google Search
          tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
      ),
  )

  # Returns a preamble part with a message to the user.
  msg_to_user = preamble.Preamble(
      content='OK, preparing a trip for the following request:\n',
  )

  # Plumb everything together with a logical switch that lets us handle errors.
  trip_request_agent = (
      extract_trip_request
      + process_json_output
      + switch.Switch(content_api.get_substream_name).case(
          # default substream name, no error.
          '',
          # For processors, the `parallel_concat` is a way to run them
          # concurrently while specify how their results should be merged, here
          # they should be concatenated.
          processor.parallel_concat([msg_to_user, generate_trip]),
      )
      # Any error substream name is handled by the default processor. Here we
      # return the input part unchanged.
      .default(processor.passthrough())
  )

  print('Enter a trip request. Use ctrl+D to quit.')
  print(
      'NOTE: there is no history, rewrite your request from scratch each time.'
  )
  while True:
    try:
      text = await asyncio.to_thread(input, '\nmessage > ')
    except EOFError:
      # Exit on ctrl+D.
      return

    # For each user input, we run a new trip request agent. No re-use of
    # previous user inputs here.
    async for part in trip_request_agent(text):
      if content_api.is_text(part.mimetype):
        print(part.text, end='', flush=True)


if __name__ == '__main__':
  if not API_KEY:
    raise ValueError(
        'API key is not set. Define a GOOGLE_API_KEY environment variable with'
        ' a key obtained from AI Studio.'
    )
  asyncio.run(run_trip_request())
