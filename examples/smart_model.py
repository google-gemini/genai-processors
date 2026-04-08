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

"""A collection of agents that invoke underlying model in a way to improve quality."""

from typing import AsyncIterable

from genai_processors import content_api
from genai_processors import processor
from genai_processors.core import function_calling
from genai_processors.core import genai_model
from google.genai import types as genai_types


_RETRY_A_LOT = genai_types.HttpOptions(
    retry_options=genai_types.HttpRetryOptions(attempts=1000)
)


class CriticReviser(processor.Processor):
  """Agent that uses a critic-reviser loop to improve responses.

  This class is not a library component as the specific prompts to use are
  application-dependent.
  """

  def __init__(
      self,
      model: processor.Processor,
      max_iterations: int = 5,
  ):
    """Initializes the SmartModel.

    Args:
      model: The base generative model to use.
      max_iterations: Maximum number of critic-reviser loop iterations.
    """
    self._model = model
    self._max_iterations = max_iterations

  async def call(
      self, content: processor.ProcessorStream
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    # We gather content from the stream as we will need to reuse it multiple
    # times.
    input_content = await content.gather()

    current_response = await self._model(input_content).gather()

    for _ in range(self._max_iterations):
      critic_response = await self._model([
          input_content,
          '\n\nDraft response:\n\n',
          current_response,
          (
              '\n\nYou are a harsh critic. Review the draft response to the'
              " user's prompt. If the draft fully answers the prompt and has no"
              " obvious flaws, simply output 'OK'. Otherwise, concisely list"
              ' the flaws or missing information. Do not rewrite the response.'
          ),
      ]).gather()

      critic_text = await critic_response.text(strict=False)
      if critic_text.strip().upper() == 'OK':
        break

      current_response = await self._model([
          input_content,
          '\n\nDraft response:\n\n',
          current_response,
          '\n\nCriticism:\n',
          critic_response,
          (
              '\n\nUpdate your previous draft response to address the'
              ' criticism. Keep the parts that are already good.'
          ),
      ]).gather()

    for part in current_response:
      yield part


_RESEARCHER_SYS_INSTRUCTIONS = """
Your primary purpose is to gather information to help users with tasks that require extensive online research following a multi-step fashion.

You ALWAYS approach the problem in a structured way and break it down into several sub-tasks by calling yourself recursively using the `research_topic` tool. Ensure no relevant subtopic is overlooked. Try to start with tasks that will narrow the field of search most. If you are at nesting_level > 0 for simple tasks you can conduct research inline, relying on the `google:search` tool to collect information. AIM TO HAVE 4 - 8 SUB_QUERIES ON THE ROOT LEVEL, and at most 2 on the second nesting level. Though you can deviate if needed. Don't go deeper than 3rd level.

Try to research multiple queries in parallel by calling the `research_topic` tool multiple times. It is OK to wait for the results of previous queries or decide to ask for additional research based on the outcome.

**Each time** before issuing the function calls, you must think first, in order to collect the most complete information, so others can fully rely on it to write an exhaustive, highly detailed report without finding references by themselves.

**Tool Usage:**
- `research_topic`:
  - Use the `research_topic` tool to call the agent recursively to conduct in-depth research on sub-queries. This tool can be used either to research distinct facets of the problem or (especially for multi-step problems) to narrow down the search scope e.g. by identifying location, date, event or person. In the latter case it is useful to use a chain of `research_topic` calls.
  - For example you can ask to compose a list of objects or people matching certain criteria.
  - Pass a clear and self-contained `query_to_investigate`.
  - When uncertain, or when addressing exceptionally complex topics, proactively conduct research to verify the accuracy of your emergent answers.
  - Ensure `nesting_level` is set to 1 more than your current nesting level. If you are the root agent (no nesting_level provided), you are strictly required to use this tool and MUST split the task into several sub-tasks. Try to start with tasks that will narrow the field of search most.
- `google:search`:
  - Avoid doing research at root level: delegate data gathering to sub-agents using `research_topic` tool.
  - Short keyword search queries work much better than long natural language questions.
  - For multi-step problems *start from the most unique features that are likely to reduce search space*.
  - You can narrow or expand search results by using more specific or borat terms. But avoid overly broad searches e.g. "best hiking route" is unlikely to be useful as there are too many routes matching the criteria. If you get many results but they are mostly irrelevant - you need to constrain the search.
  - Proactively seek high-quality authoritative sources. Tailor your search terms to surface the most credible sources for the specific domain.

Take detailed notes with citations and source URLs. Be thorough - explore multiple sources per query. You must cite EVERY SOURCE used to gather information for the report.

Once all research has been completed, you must write a comprehensive 2000-3000 word report with inline citations. If you are a sub-agent (nesting level > 0), you must return the detailed notes and findings for your specific query. Before providing your final answer, verify that your exact answer is semantically consistent with your detailed reasoning.

Use proper markdown formatting with headers, bullet points, and links.
"""

_REVISER_SYS_INSTRUCTIONS = r"""
You are a research report reviser. You will receive the original research question and the full research trace including all tool calls and their results — this is your primary source of evidence.

Your task is to critically evaluate the draft (in thoughts, don't output the criticism) and produce an improved final version. You will write an exhaustive, highly detailed report on this topic for an academic audience. Prioritize factual correctness and comprehensiveness, ensuring no relevant subtopic is overlooked. Crucially, your report must directly address all components and questions within the user's original query. Synthesize the information gathered from the research trace into a coherent, detailed narrative and structured presentation.

If specific data points, comprehensive lists, or real-time information requested by the user could not be fully or reliably obtained after diligent searching, you should provide the best available alternative information if possible. This includes an estimated data range that you can approximate with your best capability or the most recent or partial data that was found. State this limitation within the relevant section(s) of the report.

## Report Format Requirements

- Use professional, unbiased clear tone maintaining the continuous narrative flow.
- Use proper markdown formatting with headers, bullet points, and links.
- Start with a clear summary of key findings.
- Directly address the user's query in the opening.
- Acknowledge any complexities or uncertainties in the data.
- You must ground your report in the research evidence. Cite all the used sources using the [cite: source_id] format.
- If the user asks for something specific (a name, a number, a comparison, a recommendation), that specific thing must be clearly stated in the answer.
- It's very important that you write a long comprehensive report unless user explicitly expresses a different preference.
- If the user requests the answer in a specific form or format, you MUST produce the answer in that form.

Output ONLY the revised report, without preamble or meta-commentary.
"""


class Researcher(processor.Processor):
  r"""Agent that uses search and recursive calls to conduct research.

  To manage context it divides the problem into smaller ones and calls itself
  recursively. Then the research log is rewritten into a final clean report.
  """

  def __init__(self, model_name: str, api_key: str):
    """Initializes the Researcher.

    Args:
      model_name: The name of the generative model to use.
      api_key: The API key to use.
    """
    tools = [
        genai_types.Tool(google_search=genai_types.GoogleSearch()),
        self.research_topic,
    ]
    model = genai_model.GenaiModel(
        api_key=api_key,
        model_name=model_name,
        generate_content_config=genai_types.GenerateContentConfig(
            system_instruction=[_RESEARCHER_SYS_INSTRUCTIONS],
            tools=tools,
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            tool_config=genai_types.ToolConfig(
                include_server_side_tool_invocations=True
            ),
        ),
        http_options=_RETRY_A_LOT,
    )
    self._researcher = function_calling.FunctionCalling(model, fns=tools)

    self._reviser = genai_model.GenaiModel(
        api_key=api_key,
        model_name=model_name,
        generate_content_config=genai_types.GenerateContentConfig(
            system_instruction=[_REVISER_SYS_INSTRUCTIONS],
        ),
        http_options=_RETRY_A_LOT,
    )

  async def call(
      self, content: content_api.ContentStream
  ) -> AsyncIterable[content_api.ProcessorPartTypes]:
    # We gather input because we need to use it twice.
    input_content = await content.gather()
    research = await self._researcher(input_content).gather()

    # The main reason why we need the reviser is that the model may
    # produce intermediate results when the nested research arrive.
    # Here we rewrite research log to a clean final report.
    async for part in self._reviser([
        input_content,
        research,
        '\n\n============\n\n',
        'Produce the cleaned and revised reply.',
    ]):
      yield part

  # A tool that allows agent to call itself recursively.
  async def research_topic(
      self,
      query_to_investigate: str,
      nesting_level: int,
  ) -> content_api.ProcessorContentTypes:
    """Returns a research on the given topic.

    Args:
      query_to_investigate: The research query to investigate.
      nesting_level: The nesting level of the research, this will be 1 more than
        your nesting or 1 if you are working on the root query.

    Returns:
      A summary of the research conducted on the sub-query.
    """
    return await self(
        [f'Your nesting level is {nesting_level}.\n\n', query_to_investigate]
    ).gather()
