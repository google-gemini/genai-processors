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

"""
genai_langchain_processor.py

An advanced, modular LangChain integration for Google Gemini that supports:
  - Agent-based tool execution
  - Conversation memory
  - Streaming and non-streaming modes
  - Multimodal inputs (text, image, audio)
  - Customizable prompts via Jinja templates
  - Metadata preservation for downstream processing
"""

import base64
from collections.abc import AsyncIterable
from typing import Callable, List, Optional

from genai_processors import processor
from genai_processors.content_api import (
    ProcessorContent,
    ProcessorPart,
    is_text,
    is_image,
    is_audio,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain import hub


class GenAILangChainProcessor(processor.Processor):
    """
    Processor that wraps Google Gemini with LangChain to provide:

      - Full agent execution with optional tool support
      - Streaming or batch LLM responses
      - Persistent conversation memory
      - Jinja-based prompt templating
      - Handling of text, image, and audio inputs
      - Metadata propagation for each part
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        system_instruction: Optional[str] = None,
        tools: Optional[List[Callable[..., ProcessorPart]]] = None,
        prompt_template: Optional[str] = None,
        streaming: bool = True,
        enable_memory: bool = True,
        **model_kwargs,
    ):
        """
        Initialize the LangChain processor.

        Args:
            model:        Name of the Google Gemini model to use.
            system_instruction:
                          Optional system-level instruction to steer the assistant.
            tools:        Optional list of LangChain @tool callables.
            prompt_template:
                          Optional Jinja template for formatting prompts.
            streaming:    If True, stream token-by-token responses.
            enable_memory:
                          If True, retain conversation history across turns.
            model_kwargs: Additional keyword args passed to the LLM client.
        """
        super().__init__()
        self.model_name = model
        self.system_instruction = system_instruction or ""
        self.streaming = streaming
        self.enable_memory = enable_memory
        self.model_kwargs = model_kwargs
        self.conversation_history: List[ProcessorPart] = []

        # Initialize the ChatGoogleGenerativeAI client
        self.llm = ChatGoogleGenerativeAI(model=model, **model_kwargs)

        # Configure optional tools and agent
        self.tools = tools or []
        if self.tools:
            agent_prompt = hub.pull("hwchase17/openai-tools-agent")
            self.agent = create_tool_calling_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=agent_prompt,
            )
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
            )
        else:
            self.agent = None
            self.agent_executor = None

        # Compile prompt template if provided
        self.prompt_template = (
            ChatPromptTemplate.from_template(prompt_template)
            if prompt_template
            else None
        )

    async def call(
        self, content_stream: AsyncIterable[ProcessorPart]
    ) -> AsyncIterable[ProcessorPart]:
        """
        Main entry point: consume an async stream of ProcessorPart,
        process via agent or direct LLM, and yield results.
        """
        # 1. Buffer incoming parts and optionally record to memory
        content = await self._collect_content(content_stream)

        # 2. Dispatch to agent-based or direct LLM processing
        if self.agent:
            async for output in self._process_with_agent(content):
                yield output
        else:
            async for output in self._process_with_llm(content):
                yield output

    async def _collect_content(
        self, stream: AsyncIterable[ProcessorPart]
    ) -> ProcessorContent:
        """
        Buffer all incoming ProcessorPart into a ProcessorContent container.
        Also appends each part to memory if enabled.
        """
        buffer = ProcessorContent()
        async for part in stream:
            buffer += part
            if self.enable_memory:
                self.conversation_history.append(part)
        return buffer

    async def _process_with_agent(
        self, content: ProcessorContent
    ) -> AsyncIterable[ProcessorPart]:
        """
        Process content using a LangChain Agent, enabling tool calls.

        Streams both LLM tokens and tool-start notifications if streaming=True.
        """
        # Flatten input (and history) into text
        text_input = self._content_to_text(content, include_history=True)

        if self.streaming:
            # Stream token and tool events
            async for event in self.agent_executor.astream_events(
                {"input": text_input}, version="v1"
            ):
                    for part in self._handle_agent_event(event):
                        yield part
        else:
            # Single-shot agent invocation
            result = await self.agent_executor.ainvoke({"input": text_input})
            yield ProcessorPart(
                result["output"],
                mimetype="text/plain",
                role="model",
                metadata={"model": self.model_name},
            )

    async def _process_with_llm(
        self, content: ProcessorContent
    ) -> AsyncIterable[ProcessorPart]:
        """
        Process content directly with the LLM (no external tools).
        Applies system instructions, memory, and optional prompt templating.
        """
        # 1. Convert to LangChain messages
        msgs = self._convert_to_langchain_messages(content)

        # 2. Prepend system instruction if provided
        if self.system_instruction:
            msgs.insert(0, SystemMessage(content=self.system_instruction))

        # 3. Include history messages if memory enabled
        if self.enable_memory and self.conversation_history:
            history_msgs = self._convert_to_langchain_messages(
                ProcessorContent(*self.conversation_history)
            )
            msgs = history_msgs + msgs

        # 4. Apply prompt template or use raw messages
        payload = (
            {"input": self.prompt_template.format(messages=msgs)}
            if self.prompt_template
            else msgs
        )

        # 5. Invoke LLM (streaming or batch)
        if self.streaming:
            async for chunk in self.llm.astream(payload):
                yield ProcessorPart(
                    chunk.content,
                    mimetype="text/plain",
                    role="model",
                    metadata={"model": self.model_name},
                )
        else:
            result = await self.llm.ainvoke(payload)
            yield ProcessorPart(
                result.content,
                mimetype="text/plain",
                role="model",
                metadata={"model": self.model_name},
            )

    def _convert_to_langchain_messages(
        self, content: ProcessorContent
    ) -> List[HumanMessage]:
        """
        Turn ProcessorContent into a list of LangChain HumanMessage.
        Supports text, image, and audio parts, preserving metadata.
        """
        messages: List[HumanMessage] = []

        for part in content:
            fragments = []

            # Handle text parts
            if is_text(part.mimetype):
                fragments.append({
                    "type": "text",
                    "text": part.text,
                    "metadata": part.metadata
                })

            # Handle image parts
            if is_image(part.mimetype) and part.bytes:
                b64 = base64.b64encode(part.bytes).decode("utf-8")
                fragments.append({
                    "type": "image_url",
                    "image_url": f"data:{part.mimetype};base64,{b64}",
                    "metadata": part.metadata
                })

            # Handle audio parts
            if is_audio(part.mimetype) and part.bytes:
                fragments.append({
                    "type": "audio",
                    "text": f"[AUDIO {len(part.bytes)} bytes]",
                    "metadata": part.metadata
                })

            # If any fragments were created, add as a HumanMessage
            if fragments:
                messages.append(
                    HumanMessage(content=fragments, role=part.role)
                )

        return messages

    def _content_to_text(
        self,
        content: ProcessorContent,
        include_history: bool = False
    ) -> str:
        """
        Flatten ProcessorContent (and optionally memory) into a simple
        newline-delimited text string for agent input.
        """
        lines: List[str] = []

        # Prepend history if requested
        if include_history and self.enable_memory:
            for past in self.conversation_history:
                prefix = "User:" if past.role == "user" else "Assistant:"
                lines.append(f"{prefix} {past.text}")

        # Encode each part as text or placeholder
        for part in content:
            if is_text(part.mimetype):
                lines.append(part.text)
            elif is_image(part.mimetype):
                lines.append("[IMAGE]")
            elif is_audio(part.mimetype):
                lines.append("[AUDIO]")

        return "\n".join(lines)

    def _handle_agent_event(self, event: dict) -> List[ProcessorPart]:
        """
        Convert AgentExecutor events into one or more ProcessorPart outputs.
        """
        outputs: List[ProcessorPart] = []
        evt_type = event.get("event")
        data = event.get("data", {})

        if evt_type == "on_llm_stream":
            # Streamed token from LLM
            outputs.append(ProcessorPart(
                data["chunk"].content,
                mimetype="text/plain",
                role="model",
                metadata={
                    "model": self.model_name,
                    "event": "token"
                },
            ))
        elif evt_type == "on_tool_start":
            # Notification that a tool is starting
            outputs.append(ProcessorPart(
                f"Using tool: {event['name']}",
                mimetype="text/plain",
                role="system",
                metadata={
                    "tool": event["name"],
                    "event": "tool_start"
                },
            ))

        return outputs

    @classmethod
    def create_agent(
        cls,
        tools: List[Callable[..., ProcessorPart]],
        model: str = "gemini-2.0-flash",
        **kwargs
    ) -> "GenAILangChainProcessor":
        """
        Factory method to create a processor with a default agent
        and system instruction for tool-enabled use.
        """
        default_system = (
            "You are a helpful AI assistant. Use provided tools when needed. "
            "Always respond concisely and accurately."
        )
        return cls(
            model=model,
            system_instruction=default_system,
            tools=tools,
            **kwargs,
        )
