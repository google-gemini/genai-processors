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

from typing import AsyncIterable
import unittest

from genai_processors import content_api
from genai_processors import processor
from genai_processors.core import adk
from google.adk import runners
from google.adk.agents import base_agent
from google.adk.agents import live_request_queue
from google.adk.events import event as adk_event
from google.adk.sessions import in_memory_session_service
from google.genai import types as genai_types


@processor.processor_function
async def _echo_processor(
    content: AsyncIterable[content_api.ProcessorPart],
) -> AsyncIterable[content_api.ProcessorPart]:
  async for part in content:
    yield f' {part.role}:'
    yield part


APP_NAME = 'test_app'
USER_ID = 'test_user'
SESSION_ID = 'test_session'


async def _get_response(
    events: AsyncIterable[adk_event.Event], is_live: bool
) -> str:
  # NOTE: This is a crude code that doesn't handle any edge cases. Good enough
  # for tests but must not be used in production.
  response = ''
  async for event in events:
    if (event.turn_complete or is_live) and event.content:
      for part in event.content.parts:
        response += part.text
  return response


async def make_runner(
    agent: base_agent.BaseAgent,
) -> runners.Runner:
  session_service = in_memory_session_service.InMemorySessionService()
  await session_service.create_session(
      app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
  )
  return runners.Runner(
      agent=agent, app_name=APP_NAME, session_service=session_service
  )


async def make_turn(runner: runners.Runner, user_content: str) -> str:
  """Makes a single turn against the agent and returns its response."""
  events = runner.run_async(
      user_id=USER_ID,
      session_id=SESSION_ID,
      new_message=genai_types.Content(
          parts=[genai_types.Part(text=user_content)], role='user'
      ),
  )
  return await _get_response(events, is_live=False)


async def run_live(runner: runners.Runner, user_content: str) -> str:
  """Runs agent in live mode and returns its response."""
  request_queue = live_request_queue.LiveRequestQueue()
  request_queue.send_realtime(
      genai_types.Blob(data=user_content.encode(), mime_type='text/plain')
  )
  request_queue.close()

  events = runner.run_live(
      user_id=USER_ID,
      session_id=SESSION_ID,
      live_request_queue=request_queue,
  )
  return await _get_response(events, is_live=True)


class AdkTest(unittest.IsolatedAsyncioTestCase):

  async def test_run_async_impl(self):
    agent = adk.ProcessorAgent(lambda: _echo_processor, name='test_agent')
    runner = await make_runner(agent)

    self.assertEqual(await make_turn(runner, '1'), ' user:1')
    # On the next turn the whole history should be fed to the processor. Which
    # for the second turn would be:
    #   - user: '1'
    #   - model: ' user:', '1'
    #   - user: '2'
    self.assertEqual(
        await make_turn(runner, '2'),
        ' user:1 model: user: model:1 user:2',
    )

  async def test_run_live_impl(self):
    agent = adk.ProcessorAgent(lambda: _echo_processor, name='test_agent')
    runner = await make_runner(agent)

    response = await run_live(runner, 'live data')
    self.assertEqual(response, ' user:live data')


if __name__ == '__main__':
  unittest.main()
