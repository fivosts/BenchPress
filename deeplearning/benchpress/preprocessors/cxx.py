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
"""Preprocessor functions for C++."""
import re
import sys

from deeplearning.benchpress.util import environment
from deeplearning.benchpress.preprocessors import clang
from deeplearning.benchpress.preprocessors import normalizer
from deeplearning.benchpress.preprocessors import public

from absl import flags

FLAGS = flags.FLAGS

LIBCXX_HEADERS = environment.LIBCXX_HEADERS
CLANG_HEADERS = environment.CLANG_HEADERS

C_COMMENT_RE = re.compile(
  r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
  re.DOTALL | re.MULTILINE,
)

# Flags to compile C++ files with. I've replicated the default search path,
# but substituted the sandboxed header locations in place of the defaults.
#   bazel-phd/bazel-out/*-py3-opt/bin/deeplearning/clgen/preprocessors/\
#     cxx_test.runfiles/llvm_mac/bin/clang -xc++ -E - -v
CLANG = environment.CLANG
CLANG_ARGS = [
  "-xc++",
  "-isystem",
  str(LIBCXX_HEADERS),
  "-isystem",
  "/usr/local/include",
  "-isystem",
  str(CLANG_HEADERS),
  "-isystem",
  "/usr/include",
  "-Wno-ignored-pragmas",
  "-ferror-limit=1",
  "-Wno-implicit-function-declaration",
  "-Wno-incompatible-library-redeclaration",
  "-Wno-macro-redefined",
  "-Wno-unused-parameter",
  "-Wno-long-long",
  "-Wcovered-switch-default",
  "-Wdelete-non-virtual-dtor",
  "-Wstring-conversion",
  "-DLLVM_BUILD_GLOBAL_ISEL",
  "-D__STDC_CONSTANT_MACROS",
  "-D__STDC_FORMAT_MACROS",
  "-D__STDC_LIMIT_MACROS",
  "-D_LIBCPP_HAS_C_ATOMIC_IMP",
]

def Preprocess(src: str,
               copts: typing.Optional[typing.List[str]] = None,
               timeout_seconds: int = 60,
              ):
  """Run input code through the compiler frontend to inline macros.

  Args:
    src: The source code to preprocess.
    copts: A list of flags to be passed to clang.
    timeout_seconds: The number of seconds to allow before killing clang.

  Returns:
    The preprocessed code.

  Raises:
    ClangException: In case of an error.
    ClangTimeout: If clang does not complete before timeout_seconds.
  """
  copts = copts or []
  cmd = ["timeout", "-s9", str(timeout_seconds), str(CLANG)] + ["-E", "-c", "-", "-o", "-"] + copts

  process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    stdin=subprocess.PIPE,
    universal_newlines=True,
  )
  stdout, stderr = process.communicate(stdin)
  if process.returncode == 9:
    raise TimeoutError(f"clang timed out after {timeout_seconds}s")

  process.stdout = stdout
  process.stderr = stderr
  if process.returncode:
    raise RuntimeError("{}: {}".format(process.stderr, process.returncode))
  return process.stdout

@public.clgen_preprocessor
def ClangPreprocess(text: str) -> str:
  try:
    return clang.StripPreprocessorLines(Preprocess(text, CLANG_ARGS))
  except Exception as e:
    raise e

def CompileLlvmBytecode(text: str) -> str:
  """A preprocessor which attempts to compile the given code.

  Args:
    text: Code to compile.

  Returns:
    LLVM IR of input source code.
  """
  return clang.CompileLlvmBytecode(text, ".cpp", CLANG_ARGS)

def CompileOptimizer(text: str,
                     optimization: typing.List[str],
                     timeout_seconds: int = 60,
                     ) -> str:
  """Compile source code to IR and apply optimization pass to source code.
  Args:
    src: The source code to compile.
    optimization: optimization pass to apply.
  Returns:
    Dictionary with 70-dimensional InstCount feature vector.
  """
  return clang.CompileOptimizer(text, ".cpp", CLANG_ARGS, optimization)

@public.clgen_preprocessor
def Compile(text: str) -> str:
  """A preprocessor which attempts to compile the given code.

  Args:
    text: Code to compile.

  Returns:
    The input code, unmodified.
  """
  return clang.Compile(text, ".cpp", CLANG_ARGS)

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
  return clang.ClangFormat(text, ".cpp")


@public.clgen_preprocessor
def NormalizeIdentifiers(text: str) -> str:
  """Normalize identifiers in C++ source code.

  Args:
    text: The source code to rewrite.

  Returns:
    Source code with identifier names normalized.

  Raises:
    RewriterException: If rewriter found nothing to rewrite.
    ClangTimeout: If rewriter fails to complete within timeout_seconds.
  """
  return normalizer.NormalizeIdentifiers(text, ".cpp", CLANG_ARGS)


@public.clgen_preprocessor
def StripComments(text: str) -> str:
  """Strip C/C++ style comments.

  Written by @markus-jarderot https://stackoverflow.com/a/241506/1318051
  """

  def Replacer(match):
    """Regex replacement callback."""
    s = match.group(0)
    if s.startswith("/"):
      return " "  # note: a space and not an empty string
    else:
      return s

  return C_COMMENT_RE.sub(Replacer, text)
