# Real-time Processing

⭐ Real-time processing enables live audio/video streaming with bidirectional
communication.

## Overview

GenAI Processors supports two approaches for real-time processing:

1.  **Gemini Live API**
    ([`live_model.LiveProcessor`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/live_model.py)):
    for native bidirectional streaming with Gemini Live API. It is efficient but
    less flexible and is Gemini-specific as it relies on a server-side
    implementation.
2.  **Turn-based Real-time**
    ([`realtime.LiveProcessor`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/realtime.py)):
    a client-side, hackable alternative to the Gemini Live API that wraps any
    turn-based non-streaming model into a bidirectional streaming API.

This document focuses on Turn-based Real-time with `realtime.LiveProcessor`.

## Turn-Based Real-time with `realtime.LiveProcessor`

When you want to build a voice agent using standard models (non-Live API), you
can use `realtime.LiveProcessor` to convert a turn-based model into a real-time
processor. It takes an infinite input stream, creates a rolling prompt from it
by cutting it at given times (e.g. when the user is done talking), and feeds
this prompt to the `turn_processor` to generate a response.

```python
from genai_processors.core import genai_model
from genai_processors.core import realtime

model = genai_model.GenaiModel("gemini-2.0-flash")
realtime_proc = realtime.LiveProcessor(model)
```

### How it Works

`realtime.LiveProcessor` manages a conversation loop: - It uses
`window.RollingPrompt` to maintain conversation history within a sliding window
(by default it keeps parts up to `duration_prompt_sec`). - It listens for
signals like `speech_to_text.StartOfSpeech` and `speech_to_text.EndOfSpeech`
(typically from a VAD or STT processor) to detect user speech and silence. - It
triggers a call to the `turn_processor` when the user finishes speaking, or when
a final transcription is available, depending on `AudioTriggerMode`, or when the
client sends a `content_api.end_of_turn()` part. - It supports **interruption**:
if the user starts speaking while the model is generating a response, the
generation is cancelled.

### Triggering Model Turns from Voice Signals

You can configure when to trigger a model call using `trigger_model_mode`:

-   `AudioTriggerMode.END_OF_SPEECH`: Trigger model when user stops talking.
    This is faster and suitable for audio-based models.
-   `AudioTriggerMode.FINAL_TRANSCRIPTION`: Trigger model when the final
    transcription is available. This is more suitable for text-based models but
    adds slight latency.

The default is `FINAL_TRANSCRIPTION`.

```python
realtime_proc = realtime.LiveProcessor(
    model,
    trigger_model_mode=realtime.AudioTriggerMode.END_OF_SPEECH,
)
```

### Voice Activity Detection (VAD)

To generate `StartOfSpeech` and `EndOfSpeech` signals, you can use the
[`speech_to_text`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/speech_to_text.py)
module that uses the Cloud Speech API. You can also use your own VAD logic, the
only requirement is to output `speech_to_text.StartOfSpeech` and
`speech_to_text.EndOfSpeech`.

```python
from genai_processors.core import speech_to_text

stt_processor = speech_to_text.SpeechToText(...)

# Chain: STT -> realtime processor
pipeline = stt_processor + realtime_proc
```

### RollingPrompt and Windowing

`realtime.LiveProcessor` uses `RollingPrompt` to manage conversation history
efficiently for long-running sessions. A custom context compression policy can
be supplied, but by default it keeps the prompt within a certain duration by
dropping old parts. `RollingPrompt` is part of the
[`window`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/window.py)
module.

```python
from genai_processors.core import window

rolling = window.RollingPrompt(
    duration_prompt_sec=300,  # Keep 5 minutes of history
)
```

For more control over windowing behavior, you can use `window.Window` to invoke
a processor on a sliding window of conversation turns.

```python
from genai_processors.core import window

rolling = window.Window(
    window_processor = turn_processor,
    compress_history = window.keep_last_n_turns(5),
)

```

The `compress_history` defines how history should be compressed when calling the
`window_processor`.

**`drop_old_parts(age_sec)`**: Remove parts older than a specified age.

```python
compress = window.drop_old_parts(age_sec=120)
```

**`keep_last_n_turns(n)`**: Keep only the last N conversation turns.

```python
compress = window.keep_last_n_turns(turns_to_keep=3)
```

### Generating Audio from Text Outputs

When using a text-based LLM, you can use the
[text_to_speech](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/text_to_speech.py)
module to generate audio output from text. It is based on the Google
Text-To-Speech API but here again, you can define your own processor to create
audio parts, just replace the `tts_processor` below with your own
implementation.

```python
from genai_processors.core import speech_to_text
from genai_processors.core import text_to_speech

stt_processor = speech_to_text.SpeechToText(...)
tts_processor = text_to_speech.TextToSpeech(...)

# Chain: STT -> realtime processor -> TTS
pipeline = stt_processor + realtime_proc + tts_processor
```

Models usually generate audio much faster than they can be played back. This
creates a challenge when a user tries to interrupt the model: once audio hits
the playback buffer, it can't be "recalled." To fix this, the `RateLimitAudio`
processor (from the
[rate_limit_audio](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/rate_limit_audio.py)
module) buffers the output and throttles it to real-time speed. This ensures the
model's output stays synced with the audio the user actually hears, making
interruptions feel natural.

```python
from genai_processors.core import rate_limit_audio
from genai_processors.core import speech_to_text
from genai_processors.core import text_to_speech

stt_processor = speech_to_text.SpeechToText(...)
tts_processor = text_to_speech.TextToSpeech(...)
rate_limiter = rate_limit_audio.RateLimitAudio(sample_rate=24000)

# Chain: STT -> realtime processor -> TTS -> Rate Limiter
pipeline = stt_processor + realtime_proc + tts_processor + rate_limiter
```

## Audio/Video I/O

GenAI processors provides convenient processors to capture or render multi-modal
inputs and outputs.

### Microphone Input

The
[`audio_io`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/audio_io.py)
module defines an audio source processor that can be used in any pipeline.

```python
from genai_processors import streams
from genai_processors.core import audio_io

import pyaudio

pya = pyaudio.PyAudio()
pipeline = audio_io.PyAudioIn(pya) + speech_to_text.SpeechToText(...) + ...

async for part in pipeline(streams.endless_stream()):
    # The pipeline listens to the mic and generate audio parts.
    ...
```

### Speaker Output

When the model outputs raw audio, you can use the `audio_io.PyAudioOut`
processor to play the audio parts on the default speaker.

```python
from genai_processors import streams
from genai_processors.core import audio_io

import pyaudio

pya = pyaudio.PyAudio()

# model is an text-in, raw audio-out LLM.
pipeline = audio_io.PyAudioIn(pya) + speech_to_text.SpeechToText(...) + realtime_proc
pipeline = pipeline + audio_io.PyAudioOut(pya)


async for part in pipeline(streams.endless_stream()):
    # The pipeline listens to the mic and sends any audio output of the LLM to
    # the default speaker.
    pass
```

Note that the library doesn't include built-in echo cancellation, so the model
may "hear" and respond to its own output. The simplest fix is to rely on your
web browser's native echo cancellation; this is why we typically run our agent
UIs in AI Studio apps. If you aren't using a browser-based UI, we recommend
using headphones to keep the model's audio separate from the microphone.

### Video Input

The
[video](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/video.py)
module contains processor sources to capture images from a camera or from your
computer screen. It is used the same way as audio inputs:

```python
from genai_processors.core import realtime
from genai_processors.core import video

from genai_processors import streams

realtime_proc = realtime.LiveProcessor(
    model,
    trigger_model_mode=realtime.AudioTriggerMode.END_OF_SPEECH,
)

pipeline = video.VideoIn() + realtime_proc

async for part in pipeline(streams.endless_stream()):
    # the pipeline receives frames from the default camera (default 1 FPS).
    ...
```

## Complete Example: Live Voice Agent

See the
[real-time simple cli](https://github.com/google-gemini/genai-processors/blob/main/examples/realtime_simple_cli.py)
example to explore how to define a straightforward real-time agent (audio only)
with a chain of processors, handling interruptions and text entries smoothly.
