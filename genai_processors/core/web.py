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
"""Utilities for fetching documents mentioned in the part stream.

NOTE: THIS MODULE IS UNDER DEVELOPMENT AND IS NOT COMPLETE YET.

Referencing URLs of documents (web pages, images, PDFs) in a prompt is a
convenient way to provide rich context for a model. While registering `http.get`
as a tool is a more flexible and robust approach, it requires an extra model
call and a round trip for each document loaded. This model offers a more
hardwired but faster alternative.

We split the responsibility for fetching documents and deciding what needs
fetching. A special `genai_processors.core.text.FetchRequest` part must be used
to explicitly reference the document to be fetched. Then `UrlFetch` processor
would replace such FetchRequest Parts with the actual content.

It is very convenient to just mention URL as text in the prompt. However it
becomes easy to trigger the fetch unintentionally and can even be dangerous. So
it should be applied closer to the UI where user journeys are more well defined.
For example parsing URLs directly pasted in-to a chat interface is probably
fine. For extra safety you may want to require the URL be on its own line.
`genai_processors.core.text.UrlExtractor` is a processor for the task.

This process can be refined further: e.g. one can use a fast model
(gemini-flash-lite or gemma-nano) to decide whether the URL should be fetched
before passing the prompt to a larger LLM. This way we can reduce latency by
making decisions fast and fetching multiple documents in parallel.
"""

# A UrlFetch processor will be added to this module later.
