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

"""Google DeepMind GenAI processors library."""

__version__ = '1.1.1'

from . import content_api as content_api_
from . import processor as processor_
from . import streams as streams_

# Aliases
ProcessorPart = content_api_.ProcessorPart
ProcessorContent = content_api_.ProcessorContent
ProcessorPartTypes = content_api_.ProcessorPartTypes
ProcessorContentTypes = content_api_.ProcessorContentTypes
Processor = processor_.Processor
PartProcessor = processor_.PartProcessor
ProcessorFn = processor_.ProcessorFn
PartProcessorWithMatchFn = processor_.PartProcessorWithMatchFn

apply_sync = processor_.apply_sync
apply_async = processor_.apply_async
chain = processor_.chain
parallel = processor_.parallel
parallel_concat = processor_.parallel_concat
create_filter = processor_.create_filter
part_processor_function = processor_.part_processor_function

stream_content = streams_.stream_content
gather_stream = streams_.gather_stream
