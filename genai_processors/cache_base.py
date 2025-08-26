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
"""Interface declaration for `ProcessorContent` cache."""

import abc
from collections.abc import Callable

from genai_processors import content_api


# Using CacheMiss = object() as a sentinel value doesn't play nicely with typing
class CacheMiss:
  """Sentinel value to represent a cache miss."""


CacheMissT = type[CacheMiss]

ProcessorContentTypes = content_api.ProcessorContentTypes
ProcessorContent = content_api.ProcessorContent


class CacheBase(abc.ABC):
  """Abstract base class for a cache for ProcessorContent."""

  @property
  @abc.abstractmethod
  def hash_fn(
      self,
  ) -> Callable[[ProcessorContentTypes], str | None]:
    """Returns the function used to convert a query into a lookup key."""

  @abc.abstractmethod
  async def lookup(
      self,
      query: ProcessorContentTypes | None = None,
      *,
      key: str | None = None,
  ) -> ProcessorContent | CacheMissT:
    """Looks up a ProcessorContent-like value in the cache for a given query."""

  @abc.abstractmethod
  async def put(
      self,
      *,
      query: ProcessorContentTypes | None = None,
      key: str | None = None,
      value: ProcessorContentTypes,
  ) -> None:
    """Puts a ProcessorContent value into the cache for a given query."""

  @abc.abstractmethod
  async def remove(self, query: ProcessorContentTypes) -> None:
    """Removes a value from the cache for a given query."""

  @abc.abstractmethod
  def with_key_prefix(self, prefix: str) -> 'CacheBase':
    """Creates a new Cache instance where generated string keys are prefixed."""
