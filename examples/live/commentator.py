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

r"""Live commentator using GenAI Processors.

You can run this example from a CLI directly or from AI Studio.

See commentator_cli.py or commentator_ais.py for usage.

The commentator observes the incoming video/audio feed and produces an audio
commentary on it. While it is the agent who drives the conversation, it is
interactive and can be interruped/corrected by a user.

This example makes good use of the various ways the Gemini API can receive
content:
 * `system_instruction`: guidelines for the model provided by the agent
   developers. They are hardcoded in this file in the PROMPT_PARTS variable.
 * `session.send_client_content` is used for turn-by-turn scenarios and is
   analogous to what `client.generate_content` does. We use it to provide any
   additional context (non audio and video feeds) to the model. The
   `event_detection` processor uses this option as it is optimal for its fixed
   cadence, turn-based operation.
 * `session.send_realtime_input` is used for realtime interactive input and has
   built-in voice activation detection (VAD). This allows us to have lower
   latency and makes the model automatically respond to user's utterances. Note
   that audio commentary from the model itself needs to be filtered out to avoid
   VAD activating erroneously. We rely on the echo cancellation built into the
   browser for that.
 * **Async function calls**: Gemini API now supports function calls that don't
   block the generation and can even stream their output back to the model. If
   the INTERRUPT scheduling strategy is used, they will interrupt ongoing
   inference and restart it with the latest information available.

The standard Gemini Live API is well suited for the use case where the user
drives the conversation and the model responds when prompted. The live
commentator agent, on the contrary, drives the conversation, keeps commentating
while still reacting to events and to what the user asks.

A naive approach would be to generate one second of commentary every second.
However, models can produce output much faster than realtime, and forcing them
to be in a lockstep with the real world will either leave the TPUs idle or
induce a huge overhead on maintaining or recalculating the KV cache. Also
text-to-speech works better if it can work on longer sentences.

So, we employ a more event-driven approach:

 * Let the model produce long-ish commentary.
 * Slow down the streaming of these comments to real-time using the
   `RateLimitAudio` processor.
 * Rely on Gemini API built-in voice activation detection to notice if the user
   speaks over, corrects, or asks the agent.
 * Run a cheap model to detect whether something notable has happened that the
   agent should interrupt its current speech and generate a new phrase.

The async function calls play the crucial role in driving the interruption
process. We register the `start_commentating` tool which drives the timing when
the model should be making comments. It is the model itself, not a tool, that
produces comments. By using `behavior="NON_BLOCKING"` the tool is able to
schedule the generation of the next comment without interrupting the current
one. It also makes the model ingest the latest observations and tells the model
whether it should continue its previous comment or react to an interruption.

Async function calls also allow the model to send structured signals to the
client while the default output is Audio. We use a `wait_for_user` async
function to let the model decide when to pause the commentary and wait for the
user to do something. As soon as the client receives the `wait_for_user` signal,
it pauses the commentary and sends back an empty response that does not trigger
any model generate call. The silent response is specific to async function calls
and enables this "fire-and-forget" calling style.

By representing the agent as an async function we also give the model ability to
 start and stop commentating at will. If the user asks to stop commentating, the
 model will cancel the tool call and turn into a regular voice activated agent.
 When asked to start commentating, the model will invoke `start_commentating`
 tool, which will awake the event detection loop.
"""

import asyncio
import collections
import dataclasses
import enum
import random
import re
import time
from typing import Any, AsyncIterable, Optional

from absl import logging
from genai_processors import content_api
from genai_processors import processor
from genai_processors import streams
from genai_processors.core import event_detection
from genai_processors.core import live_model
from genai_processors.core import rate_limit_audio
from genai_processors.core import text
from genai_processors.core import timestamp
from google.genai import types as genai_types
import numpy as np


# Model to use for the live api processor.
MODEL_LIVE = 'gemini-2.0-flash-live-001'

# Model to use for the event detection processor.
MODEL_DETECTION = 'gemini-2.0-flash-lite'

# Number of times the detection should detect a "no detection" in a row before
# stopping the commentator.
NO_DETECTION_SENSITIVITY = 3

# Audio config
RECEIVE_SAMPLE_RATE = 24000

# Maximum number of seconds the commentator will wait for a user signal before
# continuing the commentary.
MAX_SILENCE_WAIT_FOR_USER_SEC = 5

# Number of seconds to wait before checking if we need to comment again. This is
# used when a comment is blocked because of chattiness.
NO_COMMENT_DELAY_SEC = 3

# User message when the agent detects it should continue commentating.
COMMENT_MSG = (
    'Continue commentating on what you see, do not repeat the same comments.'
    ' Always address the persons in the camera as "you" and assume they can'
    ' hear you and are in the same room. Remember this is a conversation.'
    ' Do not cancel the commentating unless the user explicitly asks you to.'
)
# Response when the agent detects something special that should interrupt the
# commentating while it is ongoing.
INTERRUPT_COMMENT_MSG = (
    'Interrupt the current commentary to comment what you see now. If there is'
    ' no commentary going on, start a new one. Start your comment as if you'
    ' were interrupting yourself or interrupting a silence. Then commentate on'
    ' the new event that just triggered this interruption. If this event is'
    ' related to what you asked the user to do, continue the previous comment'
    ' or instructions. Do not cancel the commentary.'
)

# Response when the agent has waited too long for the user to respond.
WAIT_FOR_USER_MSG = (
    'Resume commentating. Do not repeat the same comments but check if anything'
    ' explains why the user is not responding.'
)

# Response when the agent is waiting for a user signal and receives an interrupt
# from the event detection processor.
INTERRUPT_WAIT_FOR_USER_MSG = (
    'Check if you need to make a `wait_for_user` call to wait for something to'
    ' happen. If you make a `wait_for_user` call, stay silent, do not output'
    ' anything but this call. Otherwise, start commentating, or answer the user'
    ' request or move to the next steps of your instructions or of the'
    ' conversation.'
)

# Response when the agent is saying something wrong or unsafe.
START_AGAIN_MSG = (
    'There was a glitch, I could not get your last message, what were you'
    ' saying?'
)

# Async function declarations.
TOOLS = [
    genai_types.Tool(
        function_declarations=[
            # start_commentating tool drives the timing when the model should
            # be making comments. It is the model itself, not a tool, that
            # produces comments. By using `behavior="NON_BLOCKING"` the tool is
            # able to schedule the generation in many ways (controlled by the
            # `scheduling` field in the function response) and to make the model
            # include the latest observations in its prompt before generation.
            genai_types.FunctionDeclaration(
                name='start_commentating',
                description=(
                    'Starts commentating on the video feed. The model should'
                    ' continue commentating until the user asks to stop. Caller'
                    ' must print() the result for this to work. Do not cancel'
                    ' the commentating unless the user asks you to.'
                ),
                behavior='NON_BLOCKING',
            )
        ]
    ),
    genai_types.Tool(
        function_declarations=[
            genai_types.FunctionDeclaration(
                name='wait_for_user',
                description=(
                    'Waits for the user to respond to your question or to do'
                    ' something special on the video stream, e.g. user picks up'
                    ' a pen & a piece of paper. This function will return'
                    ' quickly with an empty response. The next user request'
                    ' implies there has been a user response. It is up to the'
                    ' caller of this function to decide then whether to wait'
                    ' for the user again.'
                ),
                # We use the non-blocking behavior to ensure the model will not
                # output anything when receiving the function response:
                # scheduling is set to `SILENT`.
                behavior='NON_BLOCKING',
            )
        ]
    ),
]

# Prompt for the live api processor.
PROMPT_PARTS = [
    (
        'You are an agent that commentates on a video feed and'
        ' interacts with the user in a conversation. Commentate'
        ' highlighting the most important things and making it lively'
        ' and interesting for the user. Approach the commentating as if'
        ' you were on TV commentating and interviewing at the same'
        ' time. You can make jokes, explain interesting facts related'
        ' to what you see and hear, predict what could happen, judge'
        ' some actions or reactions, etc.'
    ),
    (
        'You can be interrupted in the middle of a commentary. If this'
        ' happens, you should answer the question from the user first.'
        ' Then you can continue commentating. If the user asks you to'
        ' adjust your style or tone, you should do so.'
    ),
    (
        'You can be asked to do many different things, such as teaching a skill'
        ', helping drawing something, explaing how to go into a yoga position,'
        ' or play a game like Simon says. You should do your best to help and'
        ' drive the conversation in that case, that is, be the coach or teacher'
        ' for the user in the video stream.'
    ),
    (
        'In some situations, you will need to wait for the user to do something'
        ' or to respond to your question or instructions. When this happens, be'
        ' silent and call `wait_for_user` to let the user respond or do what'
        ' they were instructed to do. Do not say `waiting for your answer` to'
        ' the user after or before you call `wait_for_user` and if you asked a'
        ' question.'
    ),
    (
        'When something changes in the image, video or audio, you'
        ' should commentate about it interrupting what you were saying'
        ' before. It is important to stay relevant to what is happening'
        ' recently and not in the images long before.'
    ),
    (
        'Always commentate assuming the persons in the video stream hear you'
        ' and are in the same room as you. You should always address'
        ' the persons in the video stream as in a normal conversation and not'
        ' say he or she when referring to them.'
    ),
    (
        'Do not make long comments at each turn, just one or two sentences. You'
        ' should not repeat the same comments twice.'
    ),
    (
        'Remember to always address the persons in the video stream in the'
        ' first person, and not he or she. You can address them by their'
        ' location or how they are dressed.'
    ),
]

# Higher resolution is needed for better text understanding.
# MEDIA_RESOLUTION_MEDIUM is the default but for higher FPS MEDIA_RESOLUTION_LOW
# might be a better fit to reduce latency and conserve tokens.
MEDIA_RESOLUTION = genai_types.MediaResolution.MEDIA_RESOLUTION_MEDIUM


class EventTypes(enum.StrEnum):
  DETECTION = 'yes'
  NO_DETECTION = 'no'
  INTERRUPTION = 'interruption'


EVENT_DETECTION_PROMPT = (
    'You are an agent that detects events in the video feed. You receive images'
    ' from a camera. You must detect whenever a person or a group of people are'
    ' facing the camera and are close enough to engage in a conversation. You'
    ' must as well detect whenever the camera is pointing at a computer'
    ' screen.\n'  # Do not fold
    'Whenever you see a person or a group of people close enough to the camera,'
    f" respond with '{EventTypes.DETECTION}'.\n"  # Do not fold
    f"Whenever you see a computer screen, respond '{EventTypes.DETECTION}'.\n"
    'Whenever you see something special, the user wanting to stop you, interact'
    ' with you or whenever you see something new that would change your'
    f" comment, respond with '{EventTypes.INTERRUPTION}'. Do not interrupt if"
    ' the user is talking to you, only when something related to the video feed'
    'changes significantly.\n'
    f"In all other cases, respond with '{EventTypes.NO_DETECTION}'.\n"
)


def audio_duration_sec(audio_data: bytes, sample_rate: int) -> float:
  """Returns the duration of the audio data in seconds."""
  # 2 bytes per sample (16bits), 24kHz sample rate
  return len(audio_data) / (2 * sample_rate)


class GenerationType(enum.Enum):
  """Type of the request that triggers a model generation call."""

  # Comment scheduled by the commentator.
  COMMENT = 1
  # Question from the user.
  USER_REQUEST = 2
  # Special event detected during the conversation.
  EVENT_INTERRUPTION = 3


@dataclasses.dataclass
class GenerationRequestInfo:
  """Info on a model Generate call request.

  We have three types of request:

  - USER_REQUEST: user is talking.
  - COMMENT: scheduled comment.
  - EVENT_INTERRUPTION: special event detected during the conversation.

  A generation request is active from the moment we sent a request to the model
  until the model returns the `generation_complete` signal. This means we have
  received all the audio parts we need. When we're after the
  `generation_complete` signal, the request is considered inactive. This is in
  contrast with the server side where the request will stay active until the
  model returns the `turn_complete` or `interrupt` signal.
  """

  # Time when the model call was made.
  generation_start_sec: Optional[float] = None
  # Type of the generation request.
  generation_type: Optional[GenerationType] = None
  # Time when the audio response stream started. None if no audio stream was
  # received.
  time_audio_start: Optional[float] = None
  # TTFT of the model call. None if no model call was made or if a model call is
  # in progress.
  ttft_sec: Optional[float] = None
  # Time of the audio response stream received so far.
  audio_duration: float = 0.0

  def update(self, media_blob: genai_types.Blob):
    """Updates the generation request with the new media data.

    Args:
      media_blob: The new media data.
    """
    if self.generation_start_sec is not None and self.ttft_sec is None:
      # New comment is starting to be streamed.
      self.time_audio_start = time.perf_counter()
      self.ttft_sec = self.time_audio_start - self.generation_start_sec
    self.audio_duration += audio_duration_sec(
        media_blob.data,
        RECEIVE_SAMPLE_RATE,
    )


class State(enum.Enum):
  """Finite set of states that the commentator can be in."""

  # Commentator is not commentating. User can still ask questions but there is
  # no comment request scheduled.
  OFF = 1
  # Commentator is commentating or responding to the user.
  TALKING = 2
  # User starts talking or, more precisely, the model has detected that the
  # audio-input is a user barging in.
  USER_IS_TALKING = 3
  # An event is detected in the video stream. This state is reached when the
  # commentator receives an event interruption. It ends when the model returns
  # the first audio part of the interruption comment.
  REQUESTING_INTERRUPTION = 4
  # The commentator is requesting a new comment. This state is reached when the
  # commentator sends a comment request to the model. It ends when the model
  # returns the first audio part of the comment.
  REQUESTING_COMMENT = 5
  # The commentator is requesting a response to the user. This state is
  # reached when the commentator sends a user request to the model. It ends
  # when the model returns the first audio part of the response.
  REQUESTING_RESPONSE = 6
  # The commentator is interrupting from a detection but has not received audio
  # parts yet.
  INTERRUPTED_FROM_DETECTION = 7
  # After talking is done, the commentator will wait for the user to respond to
  # a question or instruction, or to an interruption to occur.
  WAITING_FOR_USER = 8


class Action(enum.Enum):
  """Action that the commentator can take."""

  # Turns on the commentator.
  TURN_ON = 1
  # Turns off the commentator.
  TURN_OFF = 2
  # Starts streaming the media part back to the user. This is the first audio
  # part of a comment, an interruption or a response to a user request.
  STREAM_MEDIA_PART = 3
  # Sends a model call to respond to the user (text based).
  REQUEST_FROM_USER = 4
  # Sends a model call to react to an interruption from the event detection.
  REQUEST_INTERRUPT = 5
  # Sends a model call to get the next comment.
  REQUEST_FROM_COMMENTATOR = 6
  # Interrupts the current request (user barging in or event interruption).
  INTERRUPT = 7
  # Wait for the user to say or do something.
  WAIT_FOR_USER = 8


@dataclasses.dataclass
class CommentatorStateMachine:
  """(state, action) -> state transitions for the commentator."""

  # State of the commentator.
  state: State = State.OFF
  # Generation requests currently in progress, aka active.
  generation_request_info: Optional[GenerationRequestInfo] = None
  ttfts: collections.deque[float] = dataclasses.field(
      default_factory=collections.deque
  )
  # Commentator ID, defined by the Async Fn call that triggered the commentator.
  id: Optional[str] = None

  def update(self, action: Action, state_arg: Any = None) -> None:
    """Updates the commentator state for the new action."""
    start_state = self.state
    try:
      match (self.state, action):
        # --- Turn On/Off ---
        case (State.OFF, Action.TURN_ON | Action.INTERRUPT):
          if isinstance(state_arg, str):
            self.id = state_arg
          # When commentator is turned on, the model outputs a comment.
          self.state = State.TALKING
          # Turn on can only happen if a generation happened from the user.
          self.update(Action.REQUEST_FROM_COMMENTATOR)
        case (State.OFF, _):
          # Do nothing if the commentator is off and it's not a TURN_ON action.
          pass
        case (_, Action.TURN_OFF):
          self.state = State.OFF
          self.generation_request_info = None
          self.id = None
        # --- Interruptions from Event Detection ---
        case (
            State.TALKING | State.WAITING_FOR_USER | State.REQUESTING_COMMENT,
            Action.REQUEST_INTERRUPT,
        ):
          self.mark_start_generation(GenerationType.EVENT_INTERRUPTION)
          self.state = State.REQUESTING_INTERRUPTION
        case (State.REQUESTING_INTERRUPTION, Action.INTERRUPT):
          self.state = State.INTERRUPTED_FROM_DETECTION
        # --- Requests from User ---
        case (_, Action.INTERRUPT):
          self.state = State.USER_IS_TALKING
          self.mark_start_generation(GenerationType.USER_REQUEST)
          # Delay the generation start to model how long the user talked.
          # The interrupt signal is received when user starts talking.
          # `generation_start_sec` should be the time when the user stops
          # talking.
          self.generation_request_info.generation_start_sec = (
              time.perf_counter() + 2
          )
        case (_, Action.REQUEST_FROM_USER):
          self.state = State.REQUESTING_RESPONSE
          self.mark_start_generation(GenerationType.USER_REQUEST)
        # --- Requests from Commentator ---
        case (
            State.TALKING | State.WAITING_FOR_USER,
            Action.REQUEST_FROM_COMMENTATOR,
        ):
          self.state = State.REQUESTING_COMMENT
          self.mark_start_generation(GenerationType.COMMENT)
        case (_, Action.WAIT_FOR_USER):
          # Wait for user can only happen from the model.
          self.state = State.WAITING_FOR_USER
        # --- Streaming Media ---
        case (_, Action.STREAM_MEDIA_PART):
          if isinstance(state_arg, genai_types.Blob):
            self._update_media_blob(state_arg)
          if self.state != State.WAITING_FOR_USER:
            self.state = State.TALKING
    finally:
      logging.debug(
          '%s - Update: %s + %s -> %s',
          time.perf_counter(),
          start_state,
          action,
          self.state,
      )

  def mark_start_generation(self, generation_type: GenerationType):
    logging.debug(
        '%s - Start generation: %s', time.perf_counter(), generation_type
    )
    self.generation_request_info = GenerationRequestInfo(
        generation_start_sec=time.perf_counter(),
        generation_type=generation_type,
    )

  def _update_media_blob(self, media_part: genai_types.Blob):
    """Updates the generation request with the new media data."""
    if not self.generation_request_info:
      logging.debug(
          '%s - No generation request to update.', time.perf_counter()
      )
      return
    if self.generation_request_info.ttft_sec is None:
      self.generation_request_info.update(media_part)
      if self.generation_request_info.ttft_sec is not None:
        self.ttfts.append(self.generation_request_info.ttft_sec)
    else:
      self.generation_request_info.update(media_part)

  def predict_next_ttft(self) -> float:
    """Predict the next TTFT from the history of TTFTs."""
    if not self.ttfts:
      return 0.0
    avg = np.mean(self.ttfts)
    std = np.std(self.ttfts)
    # Subtract the standard deviation to get a lower bound on the next TTFT.
    # Underestimating the TTFT will result in the comment being triggered too
    # late, adding a delay to the conversation but making the comment more
    # aligned (time-wise) with the video stream.
    return max(0.4, avg - std)

  def tentative_trigger_time(self) -> Optional[float]:
    """Returns the tentative time when the commentator will trigger."""
    logging.debug(
        '%s - generation_request_info: %s ttft: %s',
        time.perf_counter(),
        self.generation_request_info,
        self.ttfts,
    )
    if (
        self.state != State.OFF
        and self.generation_request_info
        and self.generation_request_info.time_audio_start is not None
    ):
      return (
          self.generation_request_info.time_audio_start
          # In WAITING_FOR_USER state, the audio duration can be 0. We add a
          # minimum duration to make sure we trigger another model generate
          # at a reasonable time.
          + max(5.0, self.generation_request_info.audio_duration)
          - self.predict_next_ttft()
      )


class LiveCommentator(processor.Processor):
  """Processor generating live commentaries on a video and audio stream..

  The audio and video parts to commentate on should have the `realtime`
  substream name. Any other parts will be considered as parts of a user request.
  """

  def __init__(
      self,
      live_api_processor: live_model.LiveProcessor,
      chattiness: float = 1.0,
      unsafe_string_list: list[str] | None = None,
  ):
    """Initializes the processor.

    Args:
      live_api_processor: The live API processor to use.
      chattiness: Probability of triggering a comment when the model has
        finished talking or every 3 seconds. Set to 0 to disable commenting.
      unsafe_string_list: The strings to use for unsafe content. If None, the
        processor will not block unsafe content. If set, the processor will
        interrupt itself when it sees the string in the output.
    """
    self._processor = live_api_processor
    self._chattiness = chattiness
    self._commentator = CommentatorStateMachine()
    # Historic time to first token (TTFT) of the recent requests.
    # We use it request the next comment just before the current one finishes.
    self.ttfts = collections.deque(maxlen=50)
    self._unsafe_string_list = unsafe_string_list
    if unsafe_string_list is not None:
      pattern = '|'.join(re.escape(s) for s in unsafe_string_list)
      self._processor += text.MatchProcessor(
          pattern=pattern,
          substream_input='output_transcription',
          substream_output='unsafe_regex',
          remove_from_input_stream=False,
          flush_fn=lambda part: part.get_metadata('generation_complete', False)
          or part.get_metadata('interrupted')
          or part.get_metadata('interrupt_request')
          or part.get_metadata('turn_complete')
          or part.get_metadata('go_away'),
      )

  def set_chattiness(self, chattiness: float):
    self._chattiness = chattiness

  def _start_commentating(
      self,
      input_queue: asyncio.Queue[content_api.ProcessorPart],
      message: str = COMMENT_MSG,
      will_continue: bool = False,
      scheduling: genai_types.FunctionResponseScheduling = genai_types.FunctionResponseScheduling.WHEN_IDLE,
  ) -> None:
    """Triggers a comment from the model. Input queue is fed to the model."""
    if self._commentator.id is None:
      logging.debug(
          '%s - No commentator id, ignoring start_commentating: %s',
          time.perf_counter(),
          self._commentator,
      )
    else:
      logging.debug(
          '%s - Triggering start_commentating: %s',
          time.perf_counter(),
          self._commentator.id,
      )
      input_queue.put_nowait(
          content_api.ProcessorPart.from_function_response(
              function_call_id=self._commentator.id,
              name='start_commentating',
              response={'output': message},
              will_continue=will_continue,
              scheduling=scheduling,
          )
      )

  def _stop_commentating(
      self, input_queue: asyncio.Queue[content_api.ProcessorPart], fn_id: str
  ):
    """Cancels a comment from the model. Input queue is fed to the model."""
    input_queue.put_nowait(
        content_api.ProcessorPart.from_function_response(
            function_call_id=fn_id,
            name='start_commentating',
            response={},
            will_continue=False,
            scheduling=genai_types.FunctionResponseScheduling.SILENT,
        )
    )

  def _respond_to_wait_for_user(
      self, input_queue: asyncio.Queue[content_api.ProcessorPart], fn_id: str
  ):
    """Cancels wait_for_user from the model. Input queue is fed to the model."""
    input_queue.put_nowait(
        content_api.ProcessorPart.from_function_response(
            function_call_id=fn_id,
            name='wait_for_user',
            response={},
            will_continue=False,
            scheduling=genai_types.FunctionResponseScheduling.SILENT,
        )
    )

  async def _schedule_comment(
      self,
      at_time: float,
      input_queue: asyncio.Queue[content_api.ProcessorPart],
      message: str = COMMENT_MSG,
  ):
    """Schedules and triggers a comment from the model."""
    if self._chattiness < 1e-6:
      return
    # Wait for the last moment to trigger the comment. This minimizes the
    # delay between any event on the video stream and the comment.
    await asyncio.sleep(max(0, at_time - time.perf_counter()))
    while True:
      chattiness_dice = random.uniform(0, 1)
      if chattiness_dice < self._chattiness:
        logging.debug(
            '%s - Triggering comment: %s ', time.perf_counter(), message
        )
        self._commentator.update(Action.REQUEST_FROM_COMMENTATOR)
        self._start_commentating(
            input_queue, message=message, will_continue=True
        )
        break
      else:
        await asyncio.sleep(NO_COMMENT_DELAY_SEC)

  async def call(
      self, content: AsyncIterable[content_api.ProcessorPart]
  ) -> AsyncIterable[content_api.ProcessorPart]:
    """Run the main conversation loop."""
    schedule_task = None

    def reset_schedule_task():
      if schedule_task is not None and not schedule_task.done():
        schedule_task.cancel()

    # Input queue for the live api processor - we will inject here what we
    # want to send to the model besides the main content stream.
    input_queue = asyncio.Queue()
    input_stream = streams.merge(
        [content, streams.dequeue(input_queue)], stop_on_first=True
    )

    start_time = time.perf_counter()
    async for part in self._processor(input_stream):
      logging.log_every_n_seconds(
          logging.INFO,
          '%s - commentator running for: %s',
          10,
          time.perf_counter(),
          timestamp.to_timestamp(time.perf_counter() - start_time),
      )

      # Handle unsafe content.
      if part.substream_name == 'unsafe_regex':
        logging.info(
            '%s - Unsafe content detected: %s',
            time.perf_counter(),
            part,
        )
        self._commentator.update(Action.REQUEST_FROM_USER)
        input_queue.put_nowait(
            content_api.ProcessorPart(
                START_AGAIN_MSG
                + (
                    ' Do not mention the following expressions in your'
                    " next response: '%s'"
                    % "', '".join(self._unsafe_string_list)
                ),
                role='USER',
                substream_name='realtime',
            )
        )
        continue

      # Handle function calls.
      if part.function_call:
        logging.info(
            '%s - Received tool call: %s',
            time.perf_counter(),
            part,
        )
        fn_id = part.get_metadata('id')
        if part.part.function_call.name == 'start_commentating':
          if self._commentator.state != State.OFF:
            # We already have a comment in progress, ignore this one.
            logging.info(
                '%s - Ignoring start_commentating: %s',
                time.perf_counter(),
                fn_id,
            )
            self._stop_commentating(input_queue, fn_id)
          else:
            self._commentator.update(Action.TURN_ON, fn_id)
        elif part.part.function_call.name == 'wait_for_user':
          if self._commentator.state != State.OFF:
            logging.debug(
                '%s - Received wait_for_user: %s',
                time.perf_counter(),
                fn_id,
            )
            self._respond_to_wait_for_user(input_queue, fn_id)
            reset_schedule_task()
            self._commentator.update(Action.WAIT_FOR_USER, fn_id)
            # Schedule the resume comment now. We will cancel it and reschedule
            # it again if we receive media parts.
            tentative_trigger_time = self._commentator.tentative_trigger_time()
            if tentative_trigger_time is None:
              logging.debug(
                  '%s - No tentative trigger time, using current time',
                  time.perf_counter(),
              )
              tentative_trigger_time = time.perf_counter()
            schedule_task = processor.create_task(
                self._schedule_comment(
                    at_time=tentative_trigger_time
                    + MAX_SILENCE_WAIT_FOR_USER_SEC,
                    input_queue=input_queue,
                    message=INTERRUPT_WAIT_FOR_USER_MSG,
                )
            )
        continue

      # Handle start of turn, considered as a user request.
      if part.get_metadata('start_of_user_turn'):
        self._commentator.update(Action.REQUEST_FROM_USER)
        continue

      # Handle function cancellation:
      if part.tool_cancellation:
        if part.tool_cancellation == self._commentator.id:
          logging.debug(
              '%s - Cancelling comment function call: %s',
              time.perf_counter(),
              self._commentator.id,
          )
          self._commentator.update(Action.TURN_OFF)
          reset_schedule_task()
        continue

      # Handle when the model is done generating. All audios parts have
      # arrived at this point but they have not been all played back yet.
      if part.get_metadata('generation_complete'):
        logging.debug('%s - generation_complete', time.perf_counter())
        yield processor.ProcessorPart(
            '',
            role='model',
            metadata={'generation_complete': True},
        )
        if self._commentator.state != State.OFF:
          # We have received all audio parts, we reschedule the comment.
          reset_schedule_task()
          tentative_trigger_time = self._commentator.tentative_trigger_time()
          if self._commentator.state != State.WAITING_FOR_USER:
            # Schedule the next commentator turn.
            schedule_task = processor.create_task(
                self._schedule_comment(
                    at_time=tentative_trigger_time,
                    input_queue=input_queue,
                )
            )
          else:
            schedule_task = processor.create_task(
                self._schedule_comment(
                    at_time=tentative_trigger_time
                    + MAX_SILENCE_WAIT_FOR_USER_SEC,
                    input_queue=input_queue,
                    message=INTERRUPT_WAIT_FOR_USER_MSG,
                )
            )

      # Handle interruption from the user.
      if part.get_metadata('interrupted'):
        reset_schedule_task()
        self._commentator.update(Action.INTERRUPT)
        if self._commentator.state == State.USER_IS_TALKING:
          logging.debug('%s - Turn interrupted - user', time.perf_counter())
          yield content_api.ProcessorPart(
              '', role='model', metadata={'interrupted': True}
          )
        else:
          logging.debug(
              '%s - Turn interrupted - detection', time.perf_counter()
          )
        continue

      # Handle interrupt request from the event detection. Do not interrupt
      # yet, wait for the interruption to be confirmed by the model.
      if part.get_metadata('interrupt_request'):
        if self._commentator.state == State.WAITING_FOR_USER:
          response = INTERRUPT_WAIT_FOR_USER_MSG
        else:
          response = INTERRUPT_COMMENT_MSG
        self._commentator.update(Action.REQUEST_INTERRUPT)
        if self._commentator.state == State.REQUESTING_INTERRUPTION:
          self._start_commentating(
              input_queue,
              message=response,
              will_continue=True,
              scheduling=genai_types.FunctionResponseScheduling.INTERRUPT,
          )

      if part.get_metadata('go_away'):
        reset_schedule_task()
        return

      # Handle audio parts received from the live API.
      if part.part.inline_data:
        if self._commentator.state == State.INTERRUPTED_FROM_DETECTION:
          logging.debug(
              '%s - Yield interrupt from interruption, audio should stop now',
              time.perf_counter(),
          )
          yield content_api.ProcessorPart(
              '', role='model', metadata={'interrupted': True}
          )
        self._commentator.update(
            Action.STREAM_MEDIA_PART, part.part.inline_data
        )
      else:
        logging.debug('%s - non media part: %s', time.perf_counter(), part)

      yield part

    reset_schedule_task()


def create_live_commentator(
    api_key: str,
    chattiness: float = 1.0,
    unsafe_string_list: list[str] | None = None,
) -> processor.Processor:
  r"""Creates a live commentator.

  A live commentator processor takes audio and video as input and produces
  live commentaries on the audio and video stream. The commentaries are
  interrupted by events detected on the video stream. Input and video streams
  coming from devices should be passed to the processor with the `realtime`
  substream name.

  Args:
    api_key: The API key to use for the model.
    chattiness: Probability of triggering a comment when the model has finished
      talking or every NO_COMMENT_DELAY_SEC seconds. Set to 0 to disable
      commenting.
    unsafe_string_list: a list of strings that should not be sent back to the
      user. This is a sanity check to make sure the model does not output
      anything that is not allowed. None by default means nothing is blocked.
      When set, the commentator will interrupt itself if the model outputs this
      string and will not output the rest of the response.

  Returns:
    A live commentator processor.
  """
  event_detection_processor = event_detection.EventDetection(
      api_key=api_key,
      model=MODEL_DETECTION,
      config=genai_types.GenerateContentConfig(
          system_instruction=EVENT_DETECTION_PROMPT,
          max_output_tokens=10,
          response_mime_type='text/x.enum',
          response_schema=EventTypes,
          media_resolution=MEDIA_RESOLUTION,
      ),
      output_dict={
          ('*', EventTypes.DETECTION): [
              content_api.ProcessorPart(
                  'start commentating',
                  role='user',
                  substream_name='realtime',
                  metadata={'turn_complete': True},
              )
          ],
          (EventTypes.DETECTION, EventTypes.NO_DETECTION): [
              content_api.ProcessorPart(
                  'stop commentating',
                  role='user',
                  substream_name='realtime',
                  metadata={'turn_complete': True},
              )
          ],
          (EventTypes.DETECTION, EventTypes.INTERRUPTION): [
              content_api.ProcessorPart(
                  '',
                  role='user',
                  # Setting up a substream name here will ensure this part will
                  # not be sent to the Live API.
                  substream_name='event_detection',
                  metadata={'interrupt_request': True},
              )
          ],
          (EventTypes.INTERRUPTION, EventTypes.DETECTION): None,
      },
      sensitivity={
          (
              EventTypes.DETECTION,
              EventTypes.NO_DETECTION,
          ): NO_DETECTION_SENSITIVITY,
      },
  )
  live_api_processor = live_model.LiveProcessor(
      api_key=api_key,
      model_name=MODEL_LIVE,
      realtime_config=genai_types.LiveConnectConfig(
          tools=TOOLS,
          system_instruction=PROMPT_PARTS,
          output_audio_transcription={},
          realtime_input_config=genai_types.RealtimeInputConfig(
              turn_coverage='TURN_INCLUDES_ALL_INPUT'
          ),
          response_modalities=['AUDIO'],
          generation_config=genai_types.GenerationConfig(
              media_resolution=MEDIA_RESOLUTION
          ),
      ),
      http_options=genai_types.HttpOptions(api_version='v1alpha'),
  )
  return (
      event_detection_processor
      + LiveCommentator(
          live_api_processor=live_api_processor,
          chattiness=chattiness,
          unsafe_string_list=unsafe_string_list,
      )
      + rate_limit_audio.RateLimitAudio(RECEIVE_SAMPLE_RATE)
  )
