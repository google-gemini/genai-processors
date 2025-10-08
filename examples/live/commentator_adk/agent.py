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

r"""Live commentator ADK agent based on GenAI Processors.

## Setup

To install the dependencies for this script, run:

```
pip install genai-processors google-adk
```

Before running this script, ensure the `GOOGLE_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

## Run

Change directory to the parent folder (genai-processors/examples/live) and run
`adk web`. then navigate to http://localhost:8000/ select "commentator_adk"
agent and click on the "Use camera" button.

To restart a session click on the "New session" button and reload the page.
"""

import os

from genai_processors.core import adk
import commentator


# You need to define the API key in the environment variables.
# export GOOGLE_API_KEY=...
API_KEY = os.environ['GOOGLE_API_KEY']


root_agent = adk.ProcessorAgent(
    (lambda: commentator.create_live_commentator(API_KEY)),
    name='commentator_adk',
)
