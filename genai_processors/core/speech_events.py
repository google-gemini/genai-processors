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
"""Shared speech event dataclasses.

These dataclasses represent speech activity transitions and are used by
multiple processors (e.g. ``Vad``, ``SpeechToText``) to emit structured
events via ``ProcessorPart.from_dataclass()``.

Downstream processors like ``LiveProcessor`` detect these events using
``mime_types.is_dataclass(part.mimetype, StartOfSpeech)`` etc.
"""

import dataclasses

import dataclasses_json
from genai_processors import content_api

TRANSCRIPTION_SUBSTREAM_NAME = 'input_transcription'
ENDPOINTING_SUBSTREAM_NAME = 'input_endpointing'

@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class StartOfSpeech:
  """Start of speech event.

  Emitted when the processor detects that the user has started speaking.
  """

  pass


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class EndOfSpeech:
  """End of speech event.

  Emitted when the processor detects that the user has stopped speaking.
  """

  pass


def is_start_of_speech(part: content_api.ProcessorPart) -> bool:
  """Returns True if the part is a StartOfSpeech event."""
  from genai_processors import mime_types

  return mime_types.is_dataclass(part.mimetype, StartOfSpeech)


def is_end_of_speech(part: content_api.ProcessorPart) -> bool:
  """Returns True if the part is an EndOfSpeech event."""
  from genai_processors import mime_types

  return mime_types.is_dataclass(part.mimetype, EndOfSpeech)
