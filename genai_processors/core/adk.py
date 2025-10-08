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
"""ADK - GenAI Processors integration."""

from typing import AsyncGenerator, AsyncIterable, Callable, override

from genai_processors import content_api
from genai_processors import processor
from google.adk.agents import base_agent
from google.adk.agents import invocation_context
from google.adk.events import event as adk_event
from google.adk.events import event_actions
from google.genai import types as genai_types


class ProcessorAgent(base_agent.BaseAgent):
  """ADK Custom agent that delegates processing its input to a Processor.

  Works both for turn-based and live modes. In case of turn-based mode the
  Processor will be called for each user turn with the whole conversation
  history up to that turn.

  This agent does use ADK token streaming. "Token streaming" checkbox in the ADK
  UI must be enabled for the response to be rendered correctly. If used
  programmatically, consumer must either only use events marked as partial=True
  or turn_complete=True, but not both.
  """

  def __init__(
      self, processor_factory: Callable[[], processor.Processor], *, name: str
  ):
    """Initializes the ProcessorAgent.

    Args:
      processor_factory: A function that returns a Processor to be applied to
        the incoming content. It will be called on each turn and each request.
        Unless the returned processor is stateless, it must return a new
        instance every time to avoid state sharing between users.
      name: The agent's name. It must be a valid Python identifier and unique
        within the agent tree. It can't be "user", since it's reserved for
        end-user's input.
    """
    super().__init__(name=name)
    self._processor_factory = processor_factory

  def _append_to_history(
      self,
      ctx: invocation_context.InvocationContext,
      content: genai_types.Content,
  ) -> adk_event.Event:
    # Parsing conversation history from the Event log requires handling many
    # edge cases which ADK considers to be implementation details. Currently
    # only adk.LlmAgent is priviledged to do that. As a temporary workaround we
    # will accumulate the history in the state. Downside is that event log
    # (stored in memory) will grow as n^2.
    key = f'history_{self.name}'
    history = ctx.session.state.get(key, [])
    history.append(content)
    return adk_event.Event(
        actions=event_actions.EventActions(state_delta={key: history}),
        author=self.name,
    )

  async def _stream_history(
      self, ctx: invocation_context.InvocationContext
  ) -> AsyncIterable[content_api.ProcessorPart]:
    for content in ctx.session.state[f'history_{self.name}']:
      for part in content.parts:
        yield content_api.ProcessorPart(part, role=content.role)

  @override
  async def _run_async_impl(
      self, ctx: invocation_context.InvocationContext
  ) -> AsyncGenerator[adk_event.Event, None]:
    p = self._processor_factory()

    yield self._append_to_history(ctx, ctx.user_content)
    response = genai_types.Content(parts=[], role='model')
    async for part in p(self._stream_history(ctx)):
      yield adk_event.Event(
          content=genai_types.Content(parts=[part.part], role='model'),
          author=self.name,
          partial=True,
          invocation_id=ctx.invocation_id,
      )
      response.parts.append(part.part)

    final_event = self._append_to_history(ctx, response)
    final_event.content = response
    final_event.turn_complete = True
    yield final_event

  @override
  async def _run_live_impl(
      self, ctx: invocation_context.InvocationContext
  ) -> AsyncGenerator[adk_event.Event, None]:
    async def stream_content():
      while True:
        request = await ctx.live_request_queue.get()
        if request.blob:
          yield content_api.ProcessorPart(
              request.blob.data,
              mimetype=request.blob.mime_type,
              substream_name='realtime',
              role='user',
          )
        if request.close:
          # NOTE: Currently ADK Web doesn't close connection and this case is
          # unreachable. _run_live_impl handlers will leak. Fixing this is in
          # ADK backlog.
          break

    p = self._processor_factory()
    async for part in p(stream_content()):
      if not part.role:
        part.role = 'model'
      yield adk_event.Event(
          content=genai_types.Content(parts=[part.part]), author=self.name
      )
