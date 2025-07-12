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
"""Examples are in the root of the git repository, see the parent folder.

To make examples easily discoverable we need them to be in the parent folder.
But when doing local development with `PYTHONPATH=.`Python will look for them
here. This magic tells Python where the examples packages are really located.
"""

import importlib
import sys


class ExamplesImporter(object):

  def find_spec(self, fullname: str, path: str, target=None):
    del path, target  # Unused.
    if fullname.startswith('genai_processors.examples'):
      return importlib.util.find_spec(fullname.replace('genai_processors.', ''))
    return None


sys.meta_path.append(ExamplesImporter())
