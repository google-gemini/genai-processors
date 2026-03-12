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
r"""Event detection processor.

This processor detects events from the images it receives in its input stream:

 *  All inputs are passed to the output.
 *  The last `max_images` encountered in the input are used to detect event.
    Non-image content is ignored.
 *  If an event is detected, the `ProcessorParts` corresponding to the current
    transition of states (set in the `output_dict` passed in the constructor)
    are injected into the output.
 *  Event detection is asynchronous so events may not necessarily be injected
    after the `ProcessorPart` on which it has been observed.

This processor keeps making calls to the model passed in the constructor to
detect events in the images. When the model is generating a response, the
processor keeps collecting up to `max_images` coming up in the input stream and
sends them all together to the model in the next call. `max_images` is a
parameter that controls the number of images to keep in the input stream and
should be tuned based on the image frequency in the input stream and the latency
of the model to generate a response. With a FPS of 1, a `max_images` of 5 is a
reasonable value.

The set of events should be defined by an enum.StrEnum class.

The model response will be a string that matches one of the values of the Enum.


```py
import enum

# Only use the enum.auto() function to define the values of the enum. Or ensure
# all the string values are lower case. The empty string is used to define the
# default start state.

class EventState(enum.StrEnum):
  SUNNY = enum.auto()
  RAINING = enum.auto()
  RAINING_WITH_MEATBALLS = enum.auto()
```

The config passed to the backend model constructor should contain the response
chema for the event detection model.

```python
config = genai_types.GenerateContentConfig(
    system_instruction=(
        'Determine the weather conditions under which these images have been'
        f' taken. Respond with "{EventName.SUNNY}" if the sun is shining, '
        f'"{EventName.RAINING}" if it is raining and'
        f'"{EventName.RAINING_WITH_MEATBALLS}" if the rain contains meatballs.'
        ' You can classify any edible objects as meatballs.'),
    response_mime_type='text/x.enum',
    response_schema=EventName,
)
backend = genai_model.GenaiModel(
    api_key=..., model_name=..., generate_content_config=config)
```

Each event state transition is associated to an output that the processor will
generate when the event transitionis detected. The output should be a
`ProcessorContentTypes` or None and is passed to the constructor via the
`output_dict` argument. When the output is None, the transition is detected but
no output is generated.

By default, self-transitions are ignored, i.e. if the
transition is from a state to the same state, no output will be generated.

```python
output_dict = {
    # The '*' wild card can be used to define all states, i.e. transitions from
    # any state (including the start state) to the event state.
    ('*', EventState.EVENT_1): ProcessorPart(
        'event_1 is detected',
        role='USER',
        metadata={'turn_complete': True},
    ),
    (EventState.EVENT_1, EventState.EVENT_2): ProcessorPart(
        'event_2 is detected',
        role='USER',
        metadata={'turn_complete': True},
    ),
    # No output for this transition.
    (EventState.EVENT_2, EventState.EVENT_1): None,
}
```


If an observed event transition is not included in the `output_dict`, the
processor will not generate any output for that transition and the new event
state will be ignored, i.e. the next transition starting state will be the same
as the previous one.

Lastly, the sensitivity dictionary passed to the constructor is used to
define the number of detections in a row that should happen before the event
detection processor sends a detection output. The keys of this dictionary should
be the event transitions and the values should be the number of detection in a
row.

This is helpful in noisy situations where you want to have confirmation of a
detection before you generate an output for the event detection.
"""

import asyncio
import collections
import time
from typing import AsyncIterable, Optional, TypeAlias
from absl import logging
from genai_processors import content_api
from genai_processors import processor
from genai_processors import streams
from genai_processors.core import timestamp

ProcessorPart = content_api.ProcessorPart
ProcessorPartTypes = content_api.ProcessorPartTypes

# Name of the default start state, i.e. no event detected.
START_STATE = ''
# A transition between two events. A state is represented here by a string,
# which is the name of the event (events should be defined by an enum.StrEnum).
EventTransition: TypeAlias = tuple[str, str]


class EventDetection(processor.Processor):
  """Event detection processor."""

  def __init__(
      self,
      backend: processor.Processor,
      output_dict: dict[
          EventTransition, content_api.ProcessorContentTypes | None
      ],
      sensitivity: Optional[dict[EventTransition, int]] = None,
      max_images: int = 5,
  ):
    """Initializes the event detection processor.

    Args:
      backend: The processor to use for the event detection.
      output_dict: A dictionary of transitions between events to the output to
        return when the transition is detected. A transition is a pair of event
        names `(from_event_state, to_event_state)`, where `from_event_state` can
        be the start state `START_STATE`, an event name, or the wild card `*` to
        define all transitions from any state (including the start state) to the
        event state. When the output is None, the transition is detected but no
        output is returned.
      sensitivity: A dictionary of transitions to the number of detection in a
        row that should happen before the event detection processor sends a
        detection output. By default, the sensitivity is 1 for a transition.
      max_images: The maximum number of images to keep in the input stream.
    """
    self._backend = backend
    self._sensitivity = sensitivity
    self._output_dict = {}
    self._init_output_dict(output_dict)
    self._last_transition = (START_STATE, START_STATE)
    self._transition_counter = (self._last_transition, 0)

    # deque of (image, timestamp) tuples.
    self._images = collections.deque[tuple[ProcessorPart, float]](
        maxlen=max_images
    )

  def _init_output_dict(
      self,
      output_dict: dict[EventTransition, content_api.ProcessorContentTypes],
  ):
    """Initializes the output dictionary."""
    # Collect the list of event states from the output_dict (including start
    # state).
    event_states = [START_STATE]
    for transition in output_dict:
      event_states.append(transition[1])
    # Add all wild card transitions.
    for transition, output in output_dict.items():
      if transition[0] == '*':
        for event_state in event_states:
          self._output_dict[(event_state, transition[1])] = output
    # Add the other transitions: it will override the wild card transitions by
    # more specific transitions.
    for transition, output in output_dict.items():
      if transition[0] != '*':
        self._output_dict[transition] = output

  async def detect_event(
      self,
      output_queue: asyncio.Queue[ProcessorPart],
  ):
    """Detects an event in the image."""
    images_with_timestamp = []
    start_time = None
    for image, t in self._images:
      if start_time is None:
        start_time = t
      images_with_timestamp.append(image)
      images_with_timestamp.append(
          content_api.ProcessorPart(timestamp.to_timestamp(t - start_time))
      )
    response_text = await self._backend(images_with_timestamp).text()

    logging.debug(
        '%s - Event detection response: %s / last transition: %s / transition'
        ' counter: %s',
        time.perf_counter(),
        response_text,
        self._last_transition,
        self._transition_counter,
    )
    if not response_text:
      logging.debug(
          '%s - No text response from the event detection model',
          time.perf_counter(),
      )
      return

    event_name = response_text.strip().lower()
    current_transition = (self._last_transition[1], event_name)
    if current_transition == self._transition_counter[0]:
      self._transition_counter = (
          current_transition,
          self._transition_counter[1] + 1,
      )
    else:
      self._transition_counter = (current_transition, 1)

    is_valid_transition = (
        current_transition in self._output_dict
        or current_transition[0] == START_STATE
    )
    is_valid_transition = (
        is_valid_transition and current_transition[1] != current_transition[0]
    )
    is_sensitivity_reached = (
        current_transition not in self._sensitivity
        or self._transition_counter[1] > self._sensitivity[current_transition]
    )
    if is_valid_transition and is_sensitivity_reached:
      logging.debug(
          '%s - New event transition: %s',
          time.perf_counter(),
          current_transition,
      )
      if (
          current_transition in self._output_dict
          and self._output_dict[current_transition] is not None
      ):
        for part in self._output_dict[current_transition]:
          output_queue.put_nowait(part)
      self._last_transition = current_transition
      self._transition_counter = (current_transition, 1)

  async def call(
      self, content: processor.ProcessorStream
  ) -> AsyncIterable[ProcessorPartTypes]:
    """Run the event detection processor."""
    output_queue = asyncio.Queue()

    async def consume_content():
      image_detection_task = None
      async for part in content:
        output_queue.put_nowait(part)
        if content_api.is_image(part.mimetype):
          self._images.append((part.part, time.perf_counter()))
          if image_detection_task is None or image_detection_task.done():
            # Only run one image detection task at a time when the previous
            # one is done. Use a single image to minimize detection latency.
            image_detection_task = processor.create_task(
                self.detect_event(output_queue)
            )
      output_queue.put_nowait(None)
      if image_detection_task is not None:
        try:
          image_detection_task.cancel()
          await image_detection_task
        except asyncio.CancelledError:
          pass

    consume_content_task = processor.create_task(consume_content())

    while chunk := await output_queue.get():
      yield chunk

    await consume_content_task
