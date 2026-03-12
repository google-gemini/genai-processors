# Live Illustrator 🏞️

This agent creates illustrations in real time for the audio narration captured
by your laptop’s microphone. Whether you're in a meeting, playing DnD, or
reading a story to your kids, it generates illustrations automatically based on
the conversation. It uses Gemini flash for the listener and Nano Banana (regular
or pro) for generating visuals. To improve consistency, when a new character or
key object / location are mentioned the agent first generates a "concept art",
which is then used as a reference when generating illustrations.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/google-gemini/genai-processors/blob/main/LICENSE)

This demo makes a good use of
[async functions / tools](https://github.com/google-gemini/genai-processors/blob/main/genai_processors/core/function_calling.py)
which have been added to Genai Processors v2.0. This way image generation does
not block the listener model and the agent can keep up with the narration,
working on multiple illustrations in parallel, if needed.

**Note:** The agent generates a high volume of images. You might get throttled
by Nano Banana or consume a considerable amount of tokens. You can control the
period between two image generation in the UI, we recommend to set it to 30
seconds to start.

## 🚀 How to run it

This example comes with a web UI based on AI Studio Applets. To run it:

*   Install the dependencies with:

    > ```sh
    > pip install genai-processors
    > ```

*   Define a `GOOGLE_API_KEY` environment variable with your API key (we need it
    to access to Google GenAI models).

    > ```sh
    > export GOOGLE_API_KEY=...
    > ```

*   go to the directory `live_illustrator` and launch the illustrator agent:

    > ```sh
    > python3 illustrator_ais.py
    > ```

*   Access the applet at
    https://aistudio.google.com/app/apps/github/google-gemini/genai-processors/tree/main/examples/live_illustrator/ais_app.

Tip: You can set the style in the System Instruction input field (e.g.
"watercolor style" or “create images with the New Yorker cartoonish style”).

Use the "Share" button to copy the generated illustrations. They can be pasted
into a Google doc for sharing.

## ☝️ Regarding Quality

This is not a replacement for a professional artist. Upon closer look you may
find many instances where illustrations are not consistent with the story or
each other. Drawing proper illustrations takes time. Instead we focused on a use
case where previously illustrations were infeasible, almost impossible.
