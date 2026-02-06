# Dynamic Widgets Demo

This demo shows how async tools can be used to enrich model output with UI
elements. Each UI element (e.g., a plot, map, 3D viewer) is registered as a
tool. In this case, these are `ImageGenerator` (backed by Imagen 3) and
`PlotGenerator` (backed by Gemini instructed to output SVG).

*   The model outputs the element by invoking the corresponding tool. This way
    we piggy-back on all security features baked into function calling such as
    resilience to prompt injection and escaping attacks.
*   Tools are async and thus do not block the model from proceeding to output
    the rest of the reply. In fact, they immediately respond with a
    `will_continue=False` message notifying the model that it won't hear back
    from these tools. A future optimization would be to make the model continue
    generating without interruption for the tool call.
*   The tool streams the response directly to the client by sending
    `ProcessorPart`s with `substream_name=status`. This is a "reserved"
    substream which is routed straight to the client and doesn't get sent to the
    model.
*   The client can use `part.function_response.id` to associate responses to the
    original `FunctionCall` part. This allows us to avoid head-of-line blocking:
    the model can continue producing an output without waiting for the widgets
    to be generated.

Note that this way we create a tiered approach to widget rendering:

*   **Model** decides which widget needs to be rendered and what content it
    should have.
*   **Tool** is responsible for rendering the widget into HTML or another "low
    level" representation.
*   **UI** is responsible for displaying this HTML in the right place.

This setup allows streaming content independently of widgets and rendering
widgets in parallel. We can even stream widgets themselves.

## How to run it

This example comes with a web UI based on AI Studio Applets.

1.  Install the dependencies with `pip install genai-processors[live]`.

2.  Define a `GOOGLE_API_KEY` environment variable with your API key.

3.  Launch the widgets agent:
    ```shell
    python3 -m genai_processors.examples.widgets.widgets_ais
    ```

4.  Access the [AI studio applet ](https://aistudio.google.com/app/apps/github/google-gemini/genai-processors/tree/main/examples/widgets/ais_app).
    The applet source is located in `ais_app/`.

5.  Try the prompt:
    > Describe what conic sections are illustrating each of them with a plot.

    Note how the browser renders plots while they are being generated.

## Limitations

 *  The prompts were not optimized for quality: the agent's output may contain
    errors. The purpose of this demo is to demonstrate how to render widgets,
    not how to generate good ones.
 *  Illustrations and plots are not conditioned to adhere to a specific style.
