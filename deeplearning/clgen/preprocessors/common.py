# Copyright (c) 2016-2020 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""Common preprocessor passes."""

from deeplearning.clgen.preprocessors import public
from absl import flags
from eupy.native import logger as l
FLAGS = flags.FLAGS


def _MinimumLineCount(text: str, min_line_count: int) -> str:
  """Private implementation of minimum number of lines.

  Args:
    text: The source to verify the line count of.

  Returns:
    src: The unmodified input src.

  Raises:
    NoCodeException: If src is less than min_line_count long.
  """
  l.getLogger().debug("deeplearning.clgen.preprocessors.common._MinimumLineCount()")
  if len(text.strip().split("\n")) < min_line_count:
    raise ValueError
  return text


@public.clgen_preprocessor
def MinimumLineCount3(text: str) -> str:
  """Check that file contains a minimum number of lines.

  Args:
    text: The source to verify the line count of.

  Returns:
    src: The unmodified input src.

  Raises:
    NoCodeException: If src is less than min_line_count long.
  """
  l.getLogger().debug("deeplearning.clgen.preprocessors.common.MinimumLineCount3()")
  return _MinimumLineCount(text, 3)


@public.clgen_preprocessor
def StripDuplicateEmptyLines(text: str) -> str:
  """A preprocessor pass which removes duplicate empty lines.

  Args:
    text: The text to preprocess.

  Returns:
    The input text, where duplicate empty lines have been removed.
  """
  l.getLogger().debug("deeplearning.clgen.preprocessors.common.StripDuplicateEmptyLines()")
  last_line = None
  lines = []
  for line in text.split("\n"):
    if line.strip() or last_line:
      lines.append(line)
    last_line = line.rstrip()
  return "\n".join(lines)


@public.clgen_preprocessor
def StripTrailingWhitespace(text: str) -> str:
  """A preprocessor pass which strips trailing whitespace from all lines.

  Whitespace at the end of each line is removed, as is any trailing whitespace
  at the end of the input.

  Args:
    text: The text to preprocess.

  Returns:
    The input text, with trailing whitespace removed.
  """
  l.getLogger().debug("deeplearning.clgen.preprocessors.common.StripTrailingWhitespace()")
  return "\n".join(l.rstrip() for l in text.split("\n")).rstrip()

@public.clgen_preprocessor
def ExtractSingleKernels(text: str) -> str:

  # l.getLogger().error("HELLO!")
  # l.getLogger().info(text)
  # raise TypeError
  # exit()
  return [text, text]
  # return text  