# MiniMax text to speech

`MiniMaxTextToSpeech` converts non-empty text parts into audio through the
MiniMax Text-to-Audio API. It supports the global and China endpoints, current
speech models, streaming responses, and MP3, WAV, FLAC, and PCM audio.

```python
from genai_processors.contrib import minimax_text_to_speech

tts = minimax_text_to_speech.MiniMaxTextToSpeech(
    api_key='your-api-key',
    region='global_en',
    voice_setting={'voice_id': 'English_Graceful_Lady'},
    audio_setting={'format': 'mp3', 'sample_rate': 32000},
)

audio = await tts('Hello from GenAI Processors.').gather()
```

Set `region='cn_zh'` to use the China endpoint. The default model is
`speech-2.8-hd`; the processor also accepts the current `speech-2.8-turbo`,
`speech-2.6-*`, `speech-02-*`, and `speech-01-*` models.

The constructor exposes the API request settings for language boosting, output
encoding, custom pronunciation, audio encoding, voice modification, and
subtitles. Set `stream=True` to emit audio chunks as they arrive.

See the [Text-to-Audio API reference](https://platform.minimax.io/docs/api-reference/speech-t2a-http)
for voice IDs and setting details.
