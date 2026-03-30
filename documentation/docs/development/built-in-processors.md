# Built-in Processors

The `genai_processors.core` package contains a rich set of processors for
building AI agents and pipelines. Beyond model interactions, it provides tools
for handling I/O, data fetching, text manipulation, function calling, and more.

For model-specific processors like `GenaiModel`, see
[Supported Models](supported-models.md).

## Speech and Audio

Processors for handling voice input and output.

### SpeechToText

Transcribes audio streams into text using Google Cloud Speech-to-Text, and
generates speech events like `StartOfSpeech` and `EndOfSpeech`.

```python
from genai_processors.core import speech_to_text

stt = speech_to_text.SpeechToText(project_id='your-gcp-project-id')
```

*See:
[`speech_to_text.py`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/speech_to_text.py)*

### TextToSpeech

Converts text streams into audible speech using Google Cloud Text-to-Speech.

```python
from genai_processors.core import text_to_speech

tts = text_to_speech.TextToSpeech(project_id='your-gcp-project-id')
```

*See:
[`text_to_speech.py`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/text_to_speech.py)*

### Audio I/O

Use `PyAudioIn` to capture microphone input and `PyAudioOut` to play audio to
speakers.

```python
import pyaudio
from genai_processors.core import audio_io

pya = pyaudio.PyAudio()
mic_input = audio_io.PyAudioIn(pya)
speaker_output = audio_io.PyAudioOut(pya)

pipeline = mic_input + stt + model + tts + speaker_output
```

*See:
[`audio_io.py`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/audio_io.py)*

### RateLimitAudio

For streaming TTS, `RateLimitAudio` splits audio into small chunks and yields
them at their natural playback speed, allowing for smoother playback and
interruption.

```python
from genai_processors.core import rate_limit_audio

rate_limiter = rate_limit_audio.RateLimitAudio(sample_rate=24000)
pipeline = model + tts + rate_limiter + speaker_output
```

*See:
[`rate_limit_audio.py`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/rate_limit_audio.py)*

## Video and Document

Processors for handling video streams and document formats like PDF.

### VideoIn

Captures video frames from a camera or screen recording as a stream of images.

```python
from genai_processors.core import video

camera = video.VideoIn(video_mode=video.VideoMode.CAMERA, substream_name='realtime')
screen = video.VideoIn(video_mode=video.VideoMode.SCREEN, substream_name='realtime')
```

*See:
[`video.py`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/video.py)*

### PDFExtract

Extracts text and images from PDF files. For pages containing images, it renders
the page as an image; otherwise, it extracts text.

```python
from genai_processors.core import pdf

pdf_extractor = pdf.PDFExtract()
```

*See:
[`pdf.py`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/pdf.py)*

### EventDetection

An advanced processor that uses a GenAI model to detect events in a stream of
images (e.g., from a live video feed). It identifies state transitions (e.g.,
from "no object" to "object detected") and injects corresponding event
notifications into the stream. This is useful for building agents that need to
react to visual changes in real-time. For more details on realtime, see
[Realtime Processing](../concepts/realtime.md).

```python
from genai_processors.core import event_detection

detector = event_detection.EventDetection(...)
```

*See:
[`event_detection.py`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/event_detection.py)*

## Text, Templating, and Output

Processors for manipulating text, extracting information, and parsing model
outputs.

### Preamble / Suffix

Adds fixed content to the beginning (`Preamble`) or end (`Suffix`) of a stream.
Useful for adding system prompts or instructions.

```python
from genai_processors.core import preamble

add_prompt = preamble.Preamble("You are a helpful assistant.")
add_suffix = preamble.Suffix("Answer in one sentence.")
```

*See:
[`preamble.py`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/preamble.py)*

### JinjaTemplate

Renders Jinja templates, allowing dynamic prompt generation with multimodal
content.

```python
from genai_processors.core import jinja_template

template = jinja_template.JinjaTemplate(
    template_str='Summary of {{ doc_name }}: {{ content }}',
    doc_name='Annual Report',
    content_varname='content'
)
```

*See:
[`jinja_template.py`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/jinja_template.py)*

### StructuredOutputParser

If a model is prompted to return JSON, this processor parses the streamed JSON
text and converts it into `ProcessorsParts` holding Python `dataclass` or `Enum`
instances based on a provided schema.

```python
from genai_processors.core import constrained_decoding

# and schema is a dataclass or enum
parser = constrained_decoding.StructuredOutputParser(schema=MyDataClass)
pipeline = model + parser
```

*See:
[`constrained_decoding.py`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/constrained_decoding.py)*

### UrlExtractor & HtmlCleaner

`UrlExtractor` finds URLs in text and replaces them with `FetchRequest` parts.
`HtmlCleaner` strips HTML tags to produce clean HTML or plain text.

```python
from genai_processors.core import text

url_extractor = text.UrlExtractor()
html_cleaner = text.HtmlCleaner(cleaning_mode='plain')
```

*See:
[`text.py`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/text.py)*

### MatchProcessor

Finds and extracts text matching a regex pattern from the stream. This is useful
for intercepting and handling specific text patterns in model output, such as
commands or structured data embedded in text before it is returned to the user.
It can also be used to detect unsafe keywords.

```python
from genai_processors.core import text

matcher = text.MatchProcessor(pattern=r'\[command:.*\]', substream_output='commands')
```

*See:
[`text.py`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/text.py)*

## Data Fetching

Processors to fetch content from various sources like Google Drive, GitHub, or
web URLs.

### UrlFetch

Fetches content from URLs contained in `FetchRequest` parts.

```python
from genai_processors.core import text
from genai_processors.core import web

url_fetcher = web.UrlFetch()
pipeline = text.UrlExtractor() + url_fetcher + text.HtmlCleaner()
```

*See:
[`web.py`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/web.py)*

### Drive

Fetches content from Google Docs, Sheets, or Slides as PDF or CSV.

```python
from genai_processors.core import drive

docs = drive.Docs()
sheets = drive.Sheets()
slides = drive.Slides()
```

*See:
[`drive.py`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/drive.py)*

### GitHub

Fetches file content from GitHub URLs.

```python
from genai_processors.core import github

github_fetcher = github.GithubProcessor(api_key='your-github-api-key')
```

*See:
[`github.py`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/github.py)*

### Filesystem

Reads local files matching a glob pattern.

```python
from genai_processors.core import filesystem

file_loader = filesystem.GlobSource(pattern='**/*.txt', base_dir='./docs')
```

*See:
[`filesystem.py`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/filesystem.py)*

## Function Calling

The `FunctionCalling` processor automates tool use by intercepting function
calls from a model, executing the corresponding Python functions, and feeding
results back to the model.

```python
from genai_processors.core import function_calling

def get_weather(city: str) -> str:
    # ... implementation ...
    return f"Weather in {city} is sunny."

model_with_tools = genai_model.GenaiModel(...)

agent = function_calling.FunctionCalling(model=model_with_tools, fns=[get_weather])
```

*See: [Function Calling Concept](../concepts/function-calling.md) and
[`function_calling.py`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/function_calling.py)*

## Stream Manipulation

### Window

Invokes a processor on a sliding window of content, useful for processing long
streams or video.

```python
from genai_processors.core import window

# Apply model to windows of 3 turns
windowed_processor = window.Window(
    model,
    compress_history=window.keep_last_n_turns(3)
)
```

*See:
[`window.py`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/window.py)*

### Timestamp

Adds timestamp parts to a stream, typically after image frames in a video.

```python
from genai_processors.core import timestamp

ts = timestamp.add_timestamps(with_ms=True)
```

*See:
[`timestamp.py`](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/timestamp.py)*
