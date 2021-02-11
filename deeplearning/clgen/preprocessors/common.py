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
import typing
# import pathlib
# from absl import flags

from deeplearning.clgen.preprocessors import public
# from deeplearning.clgen.util import crypto

from eupy.native import logger as l

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
  return _MinimumLineCount(text, 3)


@public.clgen_preprocessor
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
  return "\n".join(l.rstrip() for l in text.split("\n")).rstrip()

@public.clgen_preprocessor
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

@public.clgen_preprocessor
def RemoveAllWhiteSpace(text: str) -> str:
  """
  WARNING! This preprocessor must not be used before Compile filter.

  Preprocessor removes entirely all whitespaces.
  Kernels are more compact, but whitespaces must be
  added between tokens to restore compilability.

  Args:
    text: The text to preprocess.

  Returns:
    The input text, with all whitespaces removed.
  """
  return text.replace(' ', '')

@public.clgen_preprocessor
def RemoveNewLines(text: str) -> str:
  """
  Preprocessor removes entirely all new lines.

  Args:
    text: The text to preprocess.

  Returns:
    The input text, with all whitespaces removed.
  """
  return text.replace('\n', '')

@public.clgen_preprocessor
def ExtractSingleKernels(text: str) -> typing.List[str]:
  """
  A preprocessor that splits a single source file to discrete kernels
  along with their potential global declarations.

  Args:
    text: The text to preprocess.

  Returns:
    List of kernels (strings).
  """
  # OpenCL kernels can only be void
  kernel_specifier = 'kernel void'
  kernel_chunks = text.split(kernel_specifier)
  actual_kernels, global_space = [], []

  for idx, chunk in enumerate(kernel_chunks):
    if idx == 0:
      # There is no way the left-most part is not empty or global
      if chunk != '':
        global_space.append(chunk)
    else:
      # Given this preprocessor is called after compile,
      # we are certain that brackets will be paired
      num_lbrack, num_rbrack, chunk_idx = 0, 0, 0
      while ((num_lbrack  == 0 
      or      num_lbrack  != num_rbrack)
      and     chunk_idx   <  len(chunk)):

        try:
          cur_tok = chunk[chunk_idx]
        except IndexError:
          l.getLogger().warn(chunk)
        if   cur_tok == "{":
          num_lbrack += 1
        elif cur_tok == "}":
          num_rbrack += 1
        chunk_idx += 1

      while chunk_idx < len(chunk):
        # Without this line, global_space tends to gather lots of newlines and wspaces
        # Then they are replicated and become massive. Better isolate only actual text there.
        if chunk[chunk_idx] == ' ' or chunk[chunk_idx] == '\n':
          chunk_idx += 1
        else:
          break

      # Add to kernels all global space met so far + 'kernel void' + the kernel's body
      actual_kernels.append(''.join(global_space) + kernel_specifier + chunk[:chunk_idx])
      if ''.join(chunk[chunk_idx:]) != '':
        # All the rest below are appended to global_space
        global_space.append(chunk[chunk_idx:])

  return actual_kernels

@public.clgen_preprocessor
def ExtractOnlySingleKernels(text: str) -> typing.List[str]:
  """
  A preprocessor that splits a single source file to discrete kernels
  along without any global declarations..

  Args:
    text: The text to preprocess.

  Returns:
    List of kernels (strings).
  """
  # OpenCL kernels can only be void
  kernel_specifier = 'kernel void'
  kernel_chunks  = text.split(kernel_specifier)
  actual_kernels = []

  for idx, chunk in enumerate(kernel_chunks):
    if idx != 0:
      # Given this preprocessor is called after compile,
      # we are certain that brackets will be paired
      num_lbrack, num_rbrack, chunk_idx = 0, 0, 0
      while ((num_lbrack  == 0 
      or      num_lbrack  != num_rbrack)
      and     chunk_idx   <  len(chunk)):

        try:
          cur_tok = chunk[chunk_idx]
        except IndexError:
          l.getLogger().warn(chunk)
        if   cur_tok == "{":
          num_lbrack += 1
        elif cur_tok == "}":
          num_rbrack += 1
        chunk_idx += 1

      while chunk_idx < len(chunk):
        # Without this line, global_space tends to gather lots of newlines and wspaces
        # Then they are replicated and become massive. Better isolate only actual text there.
        if chunk[chunk_idx] == ' ' or chunk[chunk_idx] == '\n':
          chunk_idx += 1
        else:
          break
      # Add to kernels all global space met so far + 'kernel void' + the kernel's body
      actual_kernels.append(kernel_specifier + chunk[:chunk_idx])
  return actual_kernels

# @public.clgen_preprocessor
# def ExtractOnlySingleKernelsWithHeaders(text: str) -> typing.List[str]:
#   """
#   A preprocessor that splits a single source file to discrete kernels
#   along with their potential global declarations.

#   Args:
#     text: The text to preprocess.

#   Returns:
#     List of kernels (strings).
#   """
#   # OpenCL kernels can only be void
#   kernel_specifier = 'kernel void'
#   kernel_chunks = text.split(kernel_specifier)
#   actual_kernels, global_space = [], []

#   for idx, chunk in enumerate(kernel_chunks):
#     if idx == 0:
#       # There is no way the left-most part is not empty or global
#       if chunk != '':
#         global_space.append(chunk)
#     else:
#       # Given this preprocessor is called after compile,
#       # we are certain that brackets will be paired
#       num_lbrack, num_rbrack, chunk_idx = 0, 0, 0
#       while ((num_lbrack  == 0 
#       or      num_lbrack  != num_rbrack)
#       and     chunk_idx   <  len(chunk)):

#         try:
#           cur_tok = chunk[chunk_idx]
#         except IndexError:
#           l.getLogger().warn(chunk)
#         if   cur_tok == "{":
#           num_lbrack += 1
#         elif cur_tok == "}":
#           num_rbrack += 1
#         chunk_idx += 1

#       while chunk_idx < len(chunk):
#         # Without this line, global_space tends to gather lots of newlines and wspaces
#         # Then they are replicated and become massive. Better isolate only actual text there.
#         if chunk[chunk_idx] == ' ' or chunk[chunk_idx] == '\n':
#           chunk_idx += 1
#         else:
#           break

#       # Add to kernels all global space met so far + 'kernel void' + the kernel's body
#       actual_kernels.append(kernel_specifier + chunk[:chunk_idx])
#       if ''.join(chunk[chunk_idx:]) != '':
#         # All the rest below are appended to global_space
#         global_space.append(chunk[chunk_idx:])

#   print(global_space)
#   if global_space:
#     sha = crypto.sha256_str(''.join(global_space).replace('\n', '').replace(' ', ''))
#     header_path = pathlib.Path(FLAGS.workspace_dir).resolve() / "header_files"
#     header_path.mkdir(parents = True, exist_ok = True)
#     print("HA")
#     if not (header_path / "{}.h".format(sha)).exists():
#       with open(header_path / "{}.h".format(sha), 'w') as f:
#         f.write(''.join(global_space))
#     else:
#       print("TSAPOU")
#   else:
#     print("EMPTY")
#   return actual_kernels
