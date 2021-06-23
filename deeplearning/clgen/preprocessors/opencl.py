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
"""Preprocessor passes for the OpenCL programming language."""
import typing
import os
# import glob
# import pathlib

from deeplearning.clgen.util import environment
from deeplearning.clgen.preprocessors import clang
from deeplearning.clgen.preprocessors import normalizer
from deeplearning.clgen.preprocessors import public
from absl import flags

FLAGS = flags.FLAGS

LIBCLC         = environment.LIBCLC
OPENCL_HEADERS = environment.OPENCL_HEADERS
AUX_INCLUDE    = environment.AUX_INCLUDE

CL_H           = os.path.join(OPENCL_HEADERS, "CL/cl.h")
OPENCL_H       = os.path.join(environment.DATA_CL_INCLUDE, "opencl.h")
OPENCL_C_H     = os.path.join(environment.DATA_CL_INCLUDE, "opencl-c.h")
OPENCL_C_BASE  = os.path.join(environment.DATA_CL_INCLUDE, "opencl-c-base.h")
SHIMFILE       = os.path.join(environment.DATA_CL_INCLUDE, "opencl-shim.h")
STRUCTS        = os.path.join(environment.DATA_CL_INCLUDE, "structs.h")

def GetClangArgs(use_shim: bool) -> typing.List[str]:
  """Get the arguments to pass to clang for handling OpenCL.

  Args:
    use_shim: If true, inject the shim OpenCL header.
    error_limit: The number of errors to print before arboting

  Returns:
    A list of command line arguments to pass to Popen().
  """
  # args = [
  #   "-I" + str(LIBCLC),
  #   "-include",
  #   str(OPENCL_H),
  #   "-target",
  #   "nvptx64-nvidia-nvcl",
  #   f"-ferror-limit=0",
  #   "-xcl",
  #   "-Wno-everything",
  # ]

  args = [
    "-xcl",
    "--target=nvptx64-nvidia-nvcl",
    "-cl-std=CL2.0",
    "-ferror-limit=0",
    "-include{}".format(OPENCL_C_H),
    "-include{}".format(OPENCL_C_BASE),
    "-include{}".format(CL_H),
    "-include{}".format(STRUCTS),
    "-I{}".format(str(OPENCL_HEADERS)),
    "-I{}".format(str(LIBCLC)),
    "-I{}".format(str(AUX_INCLUDE)),
    "-Wno-everything",
  ]
  # header_files = glob.glob(
  #   str(
  #     pathlib.Path(FLAGS.workspace_dir).resolve() / "header_files" / "*.h"
  #   )
  # )
  # if header_files:
  #   for f in header_files:
  #     args += "-include{}".format(f)
  if use_shim:
    args += ["-include", str(SHIMFILE)]
  return args

def _ClangPreprocess(text: str, use_shim: bool) -> str:
  """Private preprocess OpenCL source implementation.

  Inline macros, removes comments, etc.

  Args:
    text: OpenCL source.
    use_shim: Inject shim header.

  Returns:
    Preprocessed source.
  """
  return clang.Preprocess(text, GetClangArgs(use_shim=use_shim))

def _ExtractTypedefs(text: str, dtype: str) -> str:
  """
  Preprocessor extracts all struct type definitions.

  Args:
    text: The text to preprocess.

  Returns:
    The input text, with all whitespaces removed.
  """
  text = text.split('typedef {}'.format(dtype))
  dtypes = []
  new_text = [text[0]]
  for t in text[1:]:
    lb, rb = 0, 0
    ssc = False
    for idx, ch in enumerate(t):
      if ch == "{":
        lb += 1
      elif ch == "}":
        rb += 1
      elif ch == ";" and ssc == True:
        dtypes.append("typedef {}".format(dtype) + t[:idx + 1])
        new_text.append(t[idx + 1:])
        break

      if lb == rb and lb != 0:
        ssc = True
  print("\n\n".join(dtypes))
  return ''.join(new_text)

def DeriveSourceVocab(text: str, token_list: typing.Set[str] = set()) -> typing.Dict[str, str]:
  """Pass CL code through clang's lexer and return set of
  tokens with appropriate delimiters for vocabulary construction.

  Args:
    text: Source code.
    token_list: Optional external list of tokens for opencl grammar.

  Returns:
    Set of unique source code tokens.
  """
  return clang.DeriveSourceVocab(text, token_list, ".cl", GetClangArgs(use_shim = False))

def AtomizeSource(text: str, vocab: typing.Set[str]) -> typing.List[str]:
  """
  Atomize OpenCL source with clang's lexer into token atoms.

  Args:
    text: The source code to compile.
    vocab: Optional set of learned vocabulary of tokenizer.

  Returns:
    Source code as a list of tokens.
  """
  return clang.AtomizeSource(text, vocab, ".cl", GetClangArgs(use_shim = False))

@public.clgen_preprocessor
def ClangPreprocess(text: str) -> str:
  """Preprocessor OpenCL source.

  Args:
    text: OpenCL source to preprocess.

  Returns:
    Preprocessed source.
  """
  return _ClangPreprocess(text, False)


@public.clgen_preprocessor
def ClangPreprocessWithShim(text: str) -> str:
  """Preprocessor OpenCL source with OpenCL shim header injection.

  Args:
    text: OpenCL source to preprocess.

  Returns:
    Preprocessed source.
  """
  return _ClangPreprocess(text, True)


@public.clgen_preprocessor
def Compile(text: str, return_diagnostics = False) -> str:
  """Check that the OpenCL source compiles.

  This does not modify the input.

  Args:
    text: OpenCL source to check.

  Returns:
    Unmodified OpenCL source.
  """
  # We must override the flag -Wno-implicit-function-declaration from
  # GetClangArgs() to ensure that undefined functions are treated as errors.
  return clang.Compile(
    text,
    ".cl",
    GetClangArgs(use_shim=False),# + ["-Werror=implicit-function-declaration"],
    return_diagnostics = return_diagnostics,
  )

@public.clgen_preprocessor
def ClangFormat(text: str) -> str:
  """Run clang-format on a source to enforce code style.

  Args:
    text: The source code to run through clang-format.

  Returns:
    The output of clang-format.

  Raises:
    ClangFormatException: In case of an error.
    ClangTimeout: If clang-format does not complete before timeout_seconds.
  """
  return clang.ClangFormat(text, ".cl")

@public.clgen_preprocessor
def ExtractStructTypedefs(text: str) -> str:
  """
  Preprocessor extracts all struct type definitions.

  Args:
    text: The text to preprocess.

  Returns:
    The input text, with all whitespaces removed.
  """
  return _ExtractTypedefs(text, 'struct')

@public.clgen_preprocessor
def ExtractUnionTypedefs(text: str) -> str:
  """
  Preprocessor extracts all union type definitions.

  Args:
    text: The text to preprocess.

  Returns:
    The input text, with all whitespaces removed.
  """
  return _ExtractTypedefs(text, 'union')

@public.clgen_preprocessor
def RemoveTypedefs(text: str) -> str:
  """
  Preprocessor removes all type aliases with typedefs, except typedef structs.

  Args:
    text: The text to preprocess.

  Returns:
    The input text, with all whitespaces removed.
  """
  text = text.split('\n')
  for i, l in enumerate(text):
    if "typedef " in l and "typedef struct" not in l and "typedef enum" not in l and "typedef union" not in l:
      text[i] = ""
  return '\n'.join(text)

@public.clgen_preprocessor
def InvertKernelSpecifier(text: str) -> str:
  """
  Inverts 'void kernel' specifier to 'kernel void'.

  Args:
    text: The text to preprocess.

  Returns:
    The input text, with all whitespaces removed.
  """
  return text.replace("void kernel ", "kernel void ")

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

@public.clgen_preprocessor
def StringKernelsToSource(text: str) -> str:
  """
  Preprocessor converts inlined C++ string kernels to OpenCL programs.

  Args:
    text: The text to preprocess.

  Returns:
    OpenCL kernel.
  """
  if '\\n"' in text:
    return ClangPreprocessWithShim(text.replace('\\n"', '').replace('"', ''))
  else:
    return text

@public.clgen_preprocessor
def NormalizeIdentifiers(text: str) -> str:
  """Normalize identifiers in OpenCL source code.

  Args:
    text: The source code to rewrite.

  Returns:
    Source code with identifier names normalized.

  Raises:
    RewriterException: If rewriter found nothing to rewrite.
    ClangTimeout: If rewriter fails to complete within timeout_seconds.
  """
  return normalizer.NormalizeIdentifiers(
    text, ".cl", GetClangArgs(use_shim=False)
  )

@public.clgen_preprocessor
def MinimumStatement1(text: str) -> str:
  """Check that file contains at least one statement.

  Args:
    text: The source to verify.

  Returns:
    src: The unmodified input src.

  Raises:
    NoCodeException: If src has no semi-colons.
  """
  if ';' not in text:
    raise ValueError
  return text

@public.clgen_preprocessor
def SanitizeKernelPrototype(text: str) -> str:
  """Sanitize OpenCL prototype.

  Ensures that OpenCL prototype fits on a single line.

  Args:
    text: OpenCL source.

  Returns:
    Source code with sanitized prototypes.
  """
  # Ensure that prototype is well-formed on a single line:
  try:
    prototype_end_idx = text.index("{") + 1
    prototype = " ".join(text[:prototype_end_idx].split())
    return prototype + text[prototype_end_idx:]
  except ValueError:
    # Ok so erm... if the '{' character isn't found, a ValueError
    # is thrown. Why would '{' not be found? Who knows, but
    # whatever, if the source file got this far through the
    # preprocessing pipeline then it's probably "good" code. It
    # could just be that an empty file slips through the cracks or
    # something.
    return text


@public.clgen_preprocessor
def StripDoubleUnderscorePrefixes(text: str) -> str:
  """Remove the optional __ qualifiers on OpenCL keywords.

  The OpenCL spec allows __ prefix for OpenCL keywords, e.g. '__global' and
  'global' are equivalent. This preprocessor removes the '__' prefix on those
  keywords.

  Args:
    text: The OpenCL source to preprocess.

  Returns:
    OpenCL source with __ stripped from OpenCL keywords.
  """
  # List of keywords taken from the OpenCL 1.2. specification, page 169.
  replacements = {
    "__const": "const",
    "__constant": "constant",
    "__global": "global",
    "__kernel": "kernel",
    "__local": "local",
    "__private": "private",
    "__read_only": "read_only",
    "__read_write": "read_write",
    "__restrict": "restrict",
    "__write_only": "write_only",
  }
  for old, new in replacements.items():
    text = text.replace(old, new)
  return text
