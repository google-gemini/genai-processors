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
"""Processor that validates ProcessorPart content against a Pydantic model."""

from dataclasses import dataclass

import json
from typing import Any, AsyncIterable, Type

from pydantic import BaseModel, ValidationError

from genai_processors import processor
from genai_processors import mime_types


@dataclass
class ValidationConfig:
    """Configuration for PydanticValidator behavior."""

    fail_on_error: bool = False
    strict_mode: bool = False


class PydanticValidator(processor.PartProcessor):
    """A PartProcessor that validates ProcessorPart content using Pydantic
        models.

    This processor inspects incoming parts for JSON data in `part.text` and
    validates it against a given Pydantic model.

    It provides feedback through the status stream and can be configured
    to either fail fast or continue on validation errors.

    Example:
        ```python
        from pydantic import BaseModel
        from genai_processors.contrib import (
            PydanticValidator,
            ValidationConfig,
        )

        class User(BaseModel):
            name: str
            age: int

        # Permissive validation that continues on errors (default)
        permissive_validator = PydanticValidator(User)

        # Strict validation that fails fast
        strict_validator = PydanticValidator(
            User,
            config=ValidationConfig(fail_on_error=True, strict_mode=True)
        )
        ```
    """

    def __init__(
        self,
        model: Type[BaseModel],
        config: ValidationConfig | None = None,
    ):
        """Initializes the validator with a Pydantic model and configuration.

        Args:
            model: The Pydantic model class to validate against.
            config: Validation configuration. Uses defaults if None.

        Raises:
            TypeError: If model is not a Pydantic BaseModel subclass.
        """
        if not issubclass(model, BaseModel):
            raise TypeError("model must be a Pydantic BaseModel subclass")

        self.model = model
        self.config = config or ValidationConfig()

    def match(self, part: processor.ProcessorPart) -> bool:
        """Determine if this part should be processed.

        Args:
            part: The ProcessorPart to check.

        Returns:
            True if the part has JSON data that can be validated.
        """
        # Check for JSON content in parts with JSON mimetype
        if mime_types.is_json(part.mimetype) and part.text:
            return True

        # Check if text content can be parsed as JSON
        if part.text:
            try:
                json.loads(part.text)
                return True
            except (json.JSONDecodeError, TypeError):
                pass

        return False

    def _get_data_to_validate(
        self, part: processor.ProcessorPart
    ) -> Any | None:
        """Extracts JSON data from a part's text."""
        if part.text:
            try:
                return json.loads(part.text)
            except json.JSONDecodeError:
                return None
        return None

    def _create_part_with_merged_metadata(
        self,
        content: str,
        mimetype: str,
        original_part: processor.ProcessorPart,
        additional_metadata: dict[str, Any],
    ) -> processor.ProcessorPart:
        """Helper to create a ProcessorPart with merged metadata."""
        return processor.ProcessorPart(
            content,
            mimetype=mimetype,
            metadata={**original_part.metadata, **additional_metadata},
        )

    async def _handle_success(
        self, validated_data: BaseModel, original_part: processor.ProcessorPart
    ) -> AsyncIterable[processor.ProcessorPart]:
        """Yields parts for a successful validation.

        We store Pydantic instance in metadata for downstream processors.
        This may cause serialization issues if metadata is later serialized.
        The instance provides convenient access to validated data with proper
        types.

        """
        # Store the validated data as JSON text
        validated_json_text = json.dumps(validated_data.model_dump(), indent=2)

        # Include the validated instance in metadata for convenience
        success_metadata = {
            "validation_status": "success",
            "validated_model": self.model.__name__,
            "validated_instance": validated_data,  # Live Pydantic instance
        }

        validated_part = self._create_part_with_merged_metadata(
            validated_json_text,
            f'application/json; validated_model={self.model.__name__}',
            original_part,
            success_metadata,
        )
        yield validated_part
        yield processor.status(
            f"✅ Successfully validated data against {self.model.__name__}"
        )

    async def _handle_failure(
        self,
        error: ValidationError,
        raw_data: Any,
        original_part: processor.ProcessorPart,
    ) -> AsyncIterable[processor.ProcessorPart]:
        """Yields parts for a failed validation."""
        yield processor.status(
            f"❌ Validation failed against {self.model.__name__}: "
            f"{len(error.errors())} errors"
        )
        if self.config.fail_on_error:
            raise error

        error_details = {
            "validation_status": "failure",
            "validated_model": self.model.__name__,
            "validation_errors": error.errors(),
            "original_data": raw_data,
        }
        failed_part = self._create_part_with_merged_metadata(
            original_part.text,  # Preserve original text
            original_part.mimetype,
            original_part,
            error_details,
        )
        yield failed_part

    async def call(
        self, part: processor.ProcessorPart
    ) -> AsyncIterable[processor.ProcessorPart]:
        """Validates a ProcessorPart using the Pydantic model."""
        data_to_validate = self._get_data_to_validate(part)

        if data_to_validate is None:
            # If no valid JSON data is found, pass the part through.
            yield part
            return

        try:
            validation_kwargs = {"strict": self.config.strict_mode}
            validated_data = self.model.model_validate(
                data_to_validate, **validation_kwargs
            )
            async for p in self._handle_success(validated_data, part):
                yield p

        except ValidationError as e:
            async for p in self._handle_failure(e, data_to_validate, part):
                yield p
