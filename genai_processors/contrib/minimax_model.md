# MiniMax text generation

`MiniMaxModel` is a turn-based text processor using the direct Chat Completions
API. It defaults to `MiniMax-M3.1-Flash`; `RECOMMENDED_MODELS` lists Flash first
and `MiniMax-M3` second. Explicit model IDs, including saved custom and legacy
IDs, are sent unchanged. Availability is determined by the selected service.

```python
import os
from genai_processors.contrib import minimax_model

async with minimax_model.MiniMaxModel(
    api_key=os.environ['MINIMAX_API_KEY'],
) as model:
  history = ['Write a short greeting.']
  answer = await model(history).gather()
  history.extend(answer)
  history.append('Make it more formal.')
  follow_up = await model(history).gather()
```

The processor buffers a turn and emits one text part. It does not store history:
pass the complete conversation on each call, retaining the output parts rather
than only their text. Their `minimax_message` metadata preserves the full
assistant response, including `reasoning_content` and any inline thinking tags,
for subsequent requests. Text roles `user`, `system`, `assistant`, and `model`
are accepted (`model` is sent as `assistant`). Tools and non-text input are
rejected explicitly; this processor does not convert or discard media.

The default endpoint is `https://api.minimax.io/v1`. Set `region='china'` for
`https://api.minimaxi.com/v1`, or supply an explicit `base_url` for a compatible
service. Custom endpoints require their own model and capability verification.
A supplied `httpx.AsyncClient` remains owned by its caller; the processor closes
only a client it creates itself.

Omitted thinking settings use the API defaults. Flash requires adaptive thinking
and accepts `reasoning_effort='low'`, `'medium'`, `'high'`, `'xhigh'`, or `'max'`
(the API default); `'none'` and disabled thinking are rejected. M3 also permits
`thinking='disabled'`; reasoning effort does not tune M3's depth. Use
`max_completion_tokens` to set an optional output limit including reasoning.
The processor does not send `service_tier` or `reasoning_split`.

API references:

- [Global Chat Completions compatibility](https://platform.minimax.io/docs/api-reference/text-openai-api)
- [China Chat Completions compatibility](https://platform.minimaxi.com/docs/api-reference/text-openai-api)
