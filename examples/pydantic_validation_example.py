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
"""Basic example demonstrating PydanticValidator usage.

This example shows how to validate JSON data against a Pydantic model,
handle both successful and failed validations, and route data accordingly.

Usage:
    python3 examples/pydantic_validation_example.py

"""

import asyncio
import json
from pydantic import BaseModel, Field

from genai_processors import processor, streams
from genai_processors.contrib.pydantic_validator import PydanticValidator


class UserData(BaseModel):
    """Simple user data model for validation."""
    user_id: int
    username: str = Field(min_length=3)
    email: str
    is_active: bool = True


async def main():
    """Demonstrates basic PydanticValidator usage."""

    # Create validator and convert to processor for stream processing
    validator = PydanticValidator(UserData).to_processor()

    # Sample data with mixed quality
    test_data = [
        # Valid user data
        {"user_id": 1, "username": "alice", "email": "alice@example.com"},

        # Invalid: username too short
        {"user_id": 2, "username": "bo", "email": "bob@example.com"},

        # Invalid: missing email
        {"user_id": 3, "username": "charlie"},

        # Not JSON - will be passed through
        "This is just text and will be ignored",
    ]

    # Convert to processor parts
    parts = [
        processor.ProcessorPart(
            json.dumps(data) if isinstance(data, dict) else data
        )
        for data in test_data
    ]

    # Create input stream
    input_stream = streams.stream_content(parts)

    # Process through validator
    print("Processing data through PydanticValidator...")
    print("=" * 50)

    async for result_part in validator(input_stream):
        # Handle status messages
        if result_part.substream_name == processor.STATUS_STREAM:
            print(f"Status: {result_part.text}")
            continue

        # Handle validation results
        validation_status = result_part.metadata.get("validation_status")

        if validation_status == "success":
            validated_user = result_part.metadata["validated_instance"]
            print(
                f"✅ Valid user: {validated_user.username} "
                f"({validated_user.email})"
            )

        elif validation_status == "failure":
            errors = result_part.metadata["validation_errors"]
            print(f"❌ Validation failed with {len(errors)} errors:")
            for error in errors:
                print(f"   - {error['loc'][0]}: {error['msg']}")

        else:
            # Passed through without validation
            print(f"⚪ Passed through: {result_part.text[:50]}...")

        print()


if __name__ == "__main__":
    asyncio.run(main())
