# PydanticValidator

The PydanticValidator is a PartProcessor that validates the JSON content of a ProcessorPart against a specified [Pydantic](https://docs.pydantic.dev/latest/) model. It provides a simple, declarative way to enforce data schemas and improve the robustness of your AI pipelines.

## **Motivation**

In many AI applications, processors ingest data from external sources like user inputs or API calls. This data can be unpredictable or malformed. The PydanticValidator solves this by:

* **Preventing Errors:** It catches invalid data early, before it can cause errors in downstream processors like a GenaiModel or a database writer.
* **Ensuring Structure:** It guarantees that any data moving forward in the pipeline conforms to a reliable, expected structure.
* **Simplifying Logic:** It allows other processors to focus on their core tasks without being cluttered with boilerplate data validation code.

## **Installation**

The processor requires the pydantic library. To use it, install the optional dependencies for this processor using the [pydantic] extra:

```bash
pip install "genai_processors[pydantic]"
```

## **Configuration**

You can customize the validator's behavior by passing a ValidationConfig object during initialization.

```python
from genai_processors.contrib.pydantic_validator import PydanticValidator, ValidationConfig

config = ValidationConfig(fail_on_error=True, strict_mode=True)
validator = PydanticValidator(MyModel, config=config)
```

**ValidationConfig Parameters:**

* fail_on_error (bool, default: False):
  * If False, the processor will catch ValidationErrors, add error details to the part's metadata, and allow the stream to continue.
  * If True, the processor will re-raise the ValidationError, stopping the stream immediately. This is useful for "fail-fast" scenarios.
* strict_mode (bool, default: False):
  * If False, Pydantic will attempt to coerce types where possible (e.g., converting the string "123" to the integer 123).
  * If True, Pydantic will enforce strict type matching and will not perform type coercion.

## **Behavior and Metadata**

The PydanticValidator processes parts that contain valid JSON in their text field. For each part it processes, it yields one or more new parts:

1. **The Result Part:** The original part, now with added metadata.
2. **A Status Part:** A message sent to the STATUS_STREAM indicating the outcome.

### **On Successful Validation:**

* The yielded part's metadata['validation_status'] is set to 'success'.
* The metadata['validated_instance'] contains the actual Pydantic model object.
* The part's text is updated to be the formatted JSON representation of the validated data.
* A processor.status() message like ✅ Successfully validated... is yielded.

### **On Failed Validation:**

* The yielded part's metadata['validation_status'] is set to 'failure'.
* metadata['validation_errors'] contains a structured list of validation errors.
* metadata['original_data'] contains the raw data that failed validation.
* A processor.status() message like ❌ Validation failed... is yielded.

## **Practical Demonstration: Building a Robust Pipeline**

A common use case is to validate a stream of user data and route valid and invalid items to different downstream processors.
This example shows how to create a filter to separate the stream after validation.

```python
import asyncio
import json
from pydantic import BaseModel, Field

from genai_processors import streams, processor
from genai_processors.contrib.pydantic_validator import PydanticValidator

# 1. Define the data schema.
class UserEvent(BaseModel):
    user_id: int
    event_name: str = Field(min_length=3)

# 2. Create the validator.
validator = PydanticValidator(model=UserEvent)

# 3. Define downstream processors for success and failure cases.
class DatabaseWriter(processor.PartProcessor):
    def process_part(self, part: processor.ProcessorPart) -> processor.ProcessorPart:
        user_event = part.metadata['validated_instance']
        print(f"DATABASE: Writing event '{user_event.event_name}' for user {user_event.user_id}")
        return part

class ErrorLogger(processor.PartProcessor):
    def process_part(self, part: processor.ProcessorPart) -> processor.ProcessorPart:
        errors = part.metadata['validation_errors']
        print(f"ERROR_LOG: Found {len(errors)} validation errors.")
        return part

# 4. Create a stream with mixed-quality data.
input_stream = streams.stream_content([
    # Valid example
    processor.ProcessorPart(json.dumps({"user_id": 101, "event_name": "login"})),
    # Invalid user_id
    processor.ProcessorPart(json.dumps({"user_id": "102", "event_name": "logout"})),
    # Invalid event_name
    processor.ProcessorPart(json.dumps({"user_id": 103, "event_name": "up"})),
    # Ignore this part
    processor.ProcessorPart("This is not a JSON part and will be ignored."),
])

# 5. Build and run the pipeline.
async def main():
    print("--- Running Validation Pipeline ---")

    # First, run the validator on the raw input.
    validated_stream = validator(input_stream)

    # Create separate streams for valid and invalid parts by filtering the metadata.
    # We also filter out the status messages for this example.

    async def is_valid(part):
        return part.metadata.get("validation_status") == "success"

    async def is_invalid(part):
        return part.metadata.get("validation_status") == "failure"

    valid_events = streams.filter_stream(validated_stream, is_valid)
    invalid_events = streams.filter_stream(validated_stream, is_invalid)

    # Chain the downstream processors.
    db_pipeline = valid_events + DatabaseWriter()
    error_pipeline = invalid_events + ErrorLogger()

    # Run both pipelines concurrently.
    await asyncio.gather(
        streams.drain_stream(db_pipeline),
        streams.drain_stream(error_pipeline)
    )
    print("--- Pipeline Finished ---")

if __name__ == "__main__":
    asyncio.run(main())

# Expected Output:
# --- Running Validation Pipeline ---
# DATABASE: Writing event 'login' for user 101
# ERROR_LOG: Found 1 validation errors.
# ERROR_LOG: Found 1 validation errors.
# --- Pipeline Finished ---
```
