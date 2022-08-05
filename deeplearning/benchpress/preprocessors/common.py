# coding=utf-8
# Copyright 2022 Chris Cummins and Foivos Tsimpourlas.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Common preprocessor passes."""
import typing
# import pathlib
# from absl import flags

from deeplearning.benchpress.preprocessors import public
# from deeplearning.benchpress.util import crypto

from deeplearning.benchpress.util import logging as l

# FLAGS = flags.FLAGS

def _MinimumLineCount(text: str, min_line_count: int) -> str:
  """Private implementation of minimum number of lines.

  Args:
    text: The source to verify the line count of.

  Returns:
    src: The unmodified input src.

  Raises:
    NoCodeException: If src is less than min_line_count long.
  """
  if len(text.strip().split("\n")) < min_line_count:
    raise ValueError
  return text

@public.benchpress_preprocessor
def MinimumLineCount3(text: str) -> str:
  """Check that file contains a minimum number of lines.

  Args:
    text: The source to verify the line count of.

  Returns:
    src: The unmodified input src.

  Raises:
    NoCodeException: If src is less than min_line_count long.
  """
  return _MinimumLineCount(text, 3)


@public.benchpress_preprocessor
def StripDuplicateEmptyLines(text: str) -> str:
  """A preprocessor pass which removes duplicate empty lines.

  Args:
    text: The text to preprocess.

  Returns:
    The input text, where duplicate empty lines have been removed.
  """
  last_line = None
  lines = []
  for line in text.split("\n"):
    if line.strip() or last_line:
      lines.append(line)
    last_line = line.rstrip()
  return "\n".join(lines)


@public.benchpress_preprocessor
def StripTrailingWhitespace(text: str) -> str:
  """A preprocessor pass which strips trailing whitespace from all lines.

  Whitespace at the end of each line is removed, as is any trailing whitespace
  at the end of the input.

  Args:
    text: The text to preprocess.

  Returns:
    The input text, with trailing whitespace removed.
  """
  return "\n".join(l.rstrip() for l in text.split("\n")).rstrip()

@public.benchpress_preprocessor
def StripMultipleWhitespaces(text: str) -> str:
  """
  Preprocessor replaces sequences of whitespaces with a single whitespace.

  Args:
    text: The text to preprocess.

  Returns:
    The input text, with trailing whitespace removed.
  """
  while "  " in text:
    text = text.replace('  ', ' ')
  return text
