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
"""This file contains utility code for working with clang.

This module does not expose any preprocessor functions for CLgen. It contains
wrappers around Clang binaries, which preprocessor functions can use to
implement specific behavior. See deeplearning.clgen.preprocessors.cxx.Compile()
for an example.
"""
import json
import re
import pathlib
import humanize
import subprocess
import tempfile
import typing
import string
import clang.cindex
from absl import flags
from deeplearning.clgen.util import environment
from eupy.native import  logger as l
from absl import flags

FLAGS = flags.FLAGS

# The marker used to mark stdin from clang pre-processor output.
CLANG_STDIN_MARKER = re.compile(r'# \d+ "<stdin>" 2')
# Options to pass to clang-format.
# See: http://clang.llvm.org/docs/ClangFormatStyleOptions.html
CLANG_FORMAT_CONFIG = {
  "BasedOnStyle": "Google",
  "ColumnLimit": 5000,
  "IndentWidth": 2,
  "AllowShortBlocksOnASingleLine": False,
  "AllowShortCaseLabelsOnASingleLine": False,
  "AllowShortFunctionsOnASingleLine": False,
  "AllowShortLoopsOnASingleLine": False,
  "AllowShortIfStatementsOnASingleLine": False,
  "DerivePointerAlignment": False,
  "PointerAlignment": "Left",
  "BreakAfterJavaFieldAnnotations": True,
  "BreakBeforeInheritanceComma": False,
  "BreakBeforeTernaryOperators": False,
  "AlwaysBreakAfterReturnType": "None",
  "AlwaysBreakAfterDefinitionReturnType": "None",
}
clang.cindex.Config.set_library_path(environment.LLVM_LIB)
if environment.LLVM_VERSION != 6:
  # LLVM 9 needs libclang explicitly defined.
  clang.cindex.Config.set_library_file(environment.LLVM_LIB + "/libclang.so.{}".format(environment.LLVM_VERSION))

CLANG        = environment.CLANG
CLANG_FORMAT = environment.CLANG_FORMAT
OPT          = environment.OPT
LLVM_EXTRACT = environment.LLVM_EXTRACT
LLVM_DIS     = environment.LLVM_DIS

def StripPreprocessorLines(src: str) -> str:
  """Strip preprocessor remnants from clang frontend output.

  Args:
    src: Clang frontend output.

  Returns:
    The output with preprocessor output stripped.
  """
  lines = src.split("\n")
  # Determine when the final included file ends.
  for i in range(len(lines) - 1, -1, -1):
    if CLANG_STDIN_MARKER.match(lines[i]):
      break
  else:
    return ""
  # Strip lines beginning with '#' (that's preprocessor stuff):
  return "\n".join([line for line in lines[i:] if not line.startswith("#")])

def Preprocess(
  src: str,
  cflags: typing.List[str],
  timeout_seconds: int = 60,
  strip_preprocessor_lines: bool = True,
):
  """Run input code through the compiler frontend to expand macros.

  This uses the repository clang binary.

  Args:
    src: The source code to preprocess.
    cflags: A list of flags to be passed to clang.
    timeout_seconds: The number of seconds to allow before killing clang.
    strip_preprocessor_lines: Whether to strip the extra lines introduced by
      the preprocessor.

  Returns:
    The preprocessed code.

  Raises:
    ClangException: In case of an error.
    ClangTimeout: If clang does not complete before timeout_seconds.
  """
  cmd = [
    "timeout",
    "-s9",
    str(timeout_seconds),
    str(CLANG),
    "-E",
    "-c",
    "-",
    "-o",
    "-",
  ] + cflags

  process = subprocess.Popen(
    cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True,
  )
  stdout, stderr = process.communicate(src)
  if process.returncode == 9:
    raise ValueError(
      f"Clang preprocessor timed out after {timeout_seconds}s"
    )
  elif process.returncode != 0:
    raise ValueError(stderr)
  if strip_preprocessor_lines:
    return StripPreprocessorLines(stdout)
  else:
    return stdout

def CompileLlvmBytecode(src: str,
                        suffix: str,
                        cflags: typing.List[str],
                        header_file: str = None,
                        timeout_seconds: int = 60
                        ) -> str:
  """Compile input code into textual LLVM byte code using clang system binary.

  Args:
    src: The source code to compile.
    suffix: The suffix to append to the source code temporary file. E.g. '.c'
      for a C program.
    cflags: A list of flags to be passed to clang.
    timeout_seconds: The number of seconds to allow before killing clang.

  Returns:
    The textual LLVM byte code.

  Raises:
    ValueError: In case of an error.
    ValueError: If clang does not complete before timeout_seconds.
  """
  builtin_cflags = ["-S", "-emit-llvm", "-o", "-"]
  try:
    tdir = FLAGS.local_filesystem
  except Exception:
    tdir = None
  with tempfile.NamedTemporaryFile("w", prefix="clgen_preprocessors_clang_", suffix=suffix, dir = tdir) as f:
    f.write(src)
    f.flush()

    extra_args = []
    if header_file:
      htf = tempfile.NamedTemporaryFile('w', prefix = "clgen_preprocessors_clang_header_", suffix = ".h", dir = tdir)
      htf.write(header_file)
      htf.flush()
      extra_args = ['-include{}'.format(htf.name)]

    cmd = (
      ["timeout", "-s9", str(timeout_seconds), str(CLANG), f.name]
      + builtin_cflags
      + cflags
      + extra_args
    )
    process = subprocess.Popen(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      universal_newlines=True,
    )
    stdout, stderr = process.communicate()
  if process.returncode == 9:
    raise ValueError(f"Clang timed out after {timeout_seconds}s")
  elif process.returncode != 0:
    raise ValueError("/*\n{}\n*/\n{}".format(stderr, src))
  return stdout

def CompileStdin(src: str,
                 suffix: str,
                 cflags: typing.List[str],
                 header_file: str = None,
                 timeout_seconds: int = 60
                 ) -> str:
  """Compile input code into textual LLVM byte code using clang system binary from standard input.

  Args:
    src: The source code to compile.
    suffix: The suffix to append to the source code temporary file. E.g. '.c'
      for a C program.
    cflags: A list of flags to be passed to clang.
    timeout_seconds: The number of seconds to allow before killing clang.

  Returns:
    The textual LLVM byte code.

  Raises:
    ValueError: In case of an error.
    ValueError: If clang does not complete before timeout_seconds.
  """
  builtin_cflags = ["-S", "-emit-llvm", "-o", "-"]
  try:
    tdir = FLAGS.local_filesystem
  except Exception:
    tdir = None

  extra_args = []
  if header_file:
    htf = tempfile.NamedTemporaryFile('w', prefix = "clgen_preprocessors_clang_header_", suffix = ".h", dir = tdir)
    htf.write(header_file)
    htf.flush()
    extra_args = ['-include{}'.format(htf.name)]

  cmd = (
    ["timeout", "-s9", str(timeout_seconds), str(CLANG), f.name]
    + builtin_cflags
    + cflags
    + extra_args
  )
  process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stdin = subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True,
  )
  stdout, stderr = process.communicate(input = src)
  if process.returncode == 9:
    raise ValueError(f"Clang timed out after {timeout_seconds}s")
  elif process.returncode != 0:
    raise ValueError("/*\n{}\n*/\n{}".format(stderr, src))
  return stdout

def HumanReadableBytecode(bc_path: pathlib.Path, timeout_seconds: int = 60) -> str:
  """Run llvm-dis to disassemble binary bytecode file to human readable format.

  Args:
    bc_path: The path to bytecode.
    timeout_seconds: The number of seconds to allow before killing clang.

  Returns:
    The textual LLVM byte code.

  Raises:
    ValueError: In case of an error.
    ValueError: If clang does not complete before timeout_seconds.
  """
  try:
    tdir = FLAGS.local_filesystem
  except Exception:
    tdir = None
  with tempfile.NamedTemporaryFile("w", prefix="human_readable_ll", suffix='.ll', dir = tdir) as f:

    cmd = (
      ["timeout",
       "-s9",
       str(timeout_seconds),
       str(LLVM_DIS),
       str(bc_path),
       "-o",
       str(f.name)]
    )
    process = subprocess.Popen(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    readable_bc = open(str(f.name), 'r').read()

  if process.returncode == 9:
    raise ValueError(f"Clang timed out after {timeout_seconds}s")
  elif process.returncode != 0:
    raise ValueError("/*\n{}\n*/\n{}".format(stderr, str(bc_path)))
  return readable_bc

def CompileOptimizer(src: str,
                     suffix: str,
                     cflags: typing.List[str],
                     optimization: typing.List[str],
                     header_file: str = None,
                     timeout_seconds: int = 60,
                     ) -> str:
  """Compile source code to IR and apply optimization pass to source code.

  Args:
    src: The source code to compile.
    suffix: The suffix to append to the source code temporary file. E.g. '.c'
      for a C program.
    cflags: A list of flags to be passed to clang.
    timeout_seconds: The number of seconds to allow before killing clang.

  Returns:
    Dictionary with 70-dimensional InstCount feature vector.

  Raises:
    ValueError: In case of an error.
    ValueError: If clang does not complete before timeout_seconds.
  """
  try:
    bc = CompileLlvmBytecode(src, suffix, cflags, header_file, timeout_seconds)
  except ValueError as e:
    raise ValueError("Compilation failed")
  try:
    tdir = FLAGS.local_filesystem
  except Exception:
    tdir = None

  with tempfile.NamedTemporaryFile("w", prefix="clgen_preprocessors_clang_", suffix='.ll', dir = tdir) as f:
    f.write(bc)
    f.flush()

    if header_file:
      """
      If the investigated kernel needs header files to be included,
      then, call llvm-extract afterwards, extract that kernel and write
      it to f.name.
      """
      # Hacky way, but llvm-extract requires exact kernel function name
      k_name = src.split('kernel void')[1].split()
      k_name = k_name[1] if "attribute" in k_name[0] else k_name[0]
      k_name = k_name.split('(', 1)[0]

      ext_cmd = (
        ["timeout", "-s9", str(timeout_seconds), str(LLVM_EXTRACT)]
        + [f.name, "--func={}".format(k_name), "-o", f.name]
      )
      ext_proc = subprocess.Popen(
        ext_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
      )
      ext_out, ext_err = ext_proc.communicate()
      if ext_err:
        raise ValueError(ext_err)

    cmd = (
      ["timeout", "-s9", str(timeout_seconds), str(OPT)]
      + optimization
      + [f.name, "-o", "/dev/null"]
    )
    process = subprocess.Popen(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      universal_newlines=True,
    )
    stdout, stderr = process.communicate()
  if process.returncode == 9:
    raise ValueError(f"Clang timed out after {timeout_seconds}s")
  elif process.returncode != 0:
    raise ValueError("/*\n{}\n*/\n{}".format(stderr, src))
  return stdout

def CompileOptimizerIR(bytecode: str,
                       suffix: str,
                       optimization: typing.List[str],
                       timeout_seconds: int = 60,
                       ) -> str:
  """Apply optimization pass directly to LLVM-IR bytecode.

  Args:
    bytecode: The source code to compile.
    suffix: The suffix to append to the source code temporary file. E.g. '.c'
      for a C program.
    timeout_seconds: The number of seconds to allow before killing clang.

  Returns:
    Dictionary with 70-dimensional InstCount  or 58-dimensional Autophase feature vector.

  Raises:
    ValueError: In case of an error.
    ValueError: If clang does not complete before timeout_seconds.
  """
  try:
    tdir = FLAGS.local_filesystem
  except Exception:
    tdir = None

  with tempfile.NamedTemporaryFile("w", prefix="clgen_preprocessors_clang_", suffix='.ll', dir = tdir) as f:
    f.write(bc)
    f.flush()

    cmd = (
      ["timeout", "-s9", str(timeout_seconds), str(OPT)]
      + optimization
      + [f.name, "-o", "/dev/null"]
    )
    process = subprocess.Popen(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      universal_newlines=True,
    )
    stdout, stderr = process.communicate()
  if process.returncode == 9:
    raise ValueError(f"Clang timed out after {timeout_seconds}s")
  elif process.returncode != 0:
    raise ValueError("/*\n{}\n*/\n{}".format(stderr, bytecode))
  return stdout

def Compile(src: str,
            suffix: str,
            cflags: typing.List[str],
            header_file: str = None,
            return_diagnostics: bool = False,
            ) -> str:
  """Check input source code for if it compiles.

  Args:
    src: The source code to compile.
    suffix: The suffix to append to the source code temporary file. E.g. '.c'
      for a C program.
    cflags: A list of flags to be passed to clang.

  Returns:
    The text, unmodified.

  Raises:
    ValueError: In case of an error.
  """
  builtin_cflags = ["-S", "-emit-llvm", "-o", "-"]
  try:
    tdir = FLAGS.local_filesystem
  except Exception:
    tdir = None
  with tempfile.NamedTemporaryFile("w", prefix="clgen_preprocessors_clang_", suffix=suffix, dir = tdir) as f:
    f.write(src)
    f.flush()

    extra_args = []
    if header_file:
      htf = tempfile.NamedTemporaryFile('w', prefix = "clgen_preprocessors_clang_header_", suffix = ".h", dir = tdir)
      htf.write(header_file)
      htf.flush()
      extra_args = ['-include{}'.format(htf.name)]

    try:
      unit = clang.cindex.TranslationUnit.from_source(f.name, args = builtin_cflags + cflags + extra_args)
    except clang.cindex.TranslationUnitLoadError as e:
      raise ValueError(e)

    diagnostics = [str(d) for d in unit.diagnostics if d.severity > 2]
    # diagnostics = [str(d) for d in unit.diagnostics if d.severity > 2 and not "implicit declaration of function" not in str(d)]

    if len(diagnostics) > 0:
      if return_diagnostics:
        return src, [(d.location.line, d.location.column) for d in unit.diagnostics if d.severity > 2]
      else:
        raise ValueError("/*\n{}\n*/\n{}".format('\n'.join(diagnostics), src))
    else:
      if return_diagnostics:
        return src, []
      else:
        return src

def Parse(src: str,
          suffix: str,
          cflags: typing.List[str],
          return_diagnostics: bool = False
          ) -> str:
  """Parse input code using clang.Cindex python module.

  Args:
    src: The source code to compile.
    suffix: The suffix to append to the source code temporary file. E.g. '.c'
      for a C program.
    cflags: A list of flags to be passed to clang.

  Returns:
    The textual LLVM byte code.

  Raises:
    ValueError: In case of an error.
  """
  try:
    tdir = FLAGS.local_filesystem
  except Exception:
    tdir = None
  with tempfile.NamedTemporaryFile("w", prefix="clgen_preprocessors_clang_", suffix=suffix, dir = tdir) as f:

    try:
      unit = clang.cindex.TranslationUnit.from_source(f.name, args = cflags, unsaved_files = [(f.name, src)])
    except clang.cindex.TranslationUnitLoadError as e:
      raise ValueError(e)

    diagnostics = [d for d in unit.diagnostics if d.severity > 2 and d.category_number in {1, 4}]

    if len(diagnostics) > 0:
      if return_diagnostics:
        return src, [(d.location.line, d.location.column) for d in diagnostics]
      else:
        raise ValueError("/*\n{}\n*/\n{}".format('\n'.join([str(d) for d in diagnostics]), src))
    else:
      if return_diagnostics:
        return src, []
      else:
        return src

def ClangFormat(src: str, suffix: str, timeout_seconds: int = 60) -> str:
  """Run clang-format on a source to enforce code style.

  Args:
    src: The source code to run through clang-format.
    suffix: The suffix to append to the source code temporary file. E.g. '.c'
      for a C program.
    timeout_seconds: The number of seconds to allow clang-format to run for.

  Returns:
    The output of clang-format.

  Raises:
    ClangFormatException: In case of an error.
    ClangTimeout: If clang-format does not complete before timeout_seconds.
  """

  cmd = [
    "timeout",
    "-s9",
    str(timeout_seconds),
    str(CLANG_FORMAT),
    "-assume-filename",
    f"input{suffix}",
    "-style={}".format(json.dumps(CLANG_FORMAT_CONFIG))
  ]
  process = subprocess.Popen(
    cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    universal_newlines=True,
  )
  stdout, stderr = process.communicate(src)
  if process.returncode == 9:
    raise ValueError(f"clang-format timed out after {timeout_seconds}s")
  elif process.returncode != 0:
    raise ValueError(stderr)
  return stdout

def ExtractFunctions(src: str,
                     suffix: str,
                     cflags: typing.List[str]
                     ) -> typing.List[str]:
  """Splits translation unit into separate functions using tokenizer.
  WARNING! Functions might need formatting after this preprocessor,
           if you care about formatting.

  Args:
    src: The source code to compile.
    suffix: The suffix to append to the source code temporary file. E.g. '.c'
      for a C program.
    cflags: A list of flags to be passed to clang.

  Returns:
    List of separate string functions

  Raises:
    ValueError: In case of an error.
  """
  try:
    tdir = FLAGS.local_filesystem
  except Exception:
    tdir = None
  with tempfile.NamedTemporaryFile(
    "w", prefix="clgen_preprocessors_clang_", suffix=suffix, dir = tdir
  ) as f:
    try:
      unit = clang.cindex.TranslationUnit.from_source(f.name, args = cflags, unsaved_files = [(f.name, src)])#, args = args + builtin_cflags)
    except clang.cindex.TranslationUnitLoadError as e:
      raise ValueError(e)

  def next_token(token_iter):
    """Return None if iterator is consumed."""
    try:
      return next(token_iter)
    except StopIteration:
      return None

  functions = []
  tokiter = unit.get_tokens(extent = unit.cursor.extent)
  token = next_token(tokiter)

  while token:
    # Do sth with token
    cur = clang.cindex.Cursor.from_location(unit, token.extent.start)
    if cur.kind == clang.cindex.CursorKind.FUNCTION_DECL:
      # Found starting point of function declaration.
      func = []
      func.append(token.spelling)
      token = next_token(tokiter)
      while token and token.spelling != ")":
        # Go until the closing parenthesis of parameters.
        func.append(token.spelling)
        token = next_token(tokiter)
      while token and token.spelling != "{" and token.spelling != ";":
        # Reject comments etc. until opening brace or semi-colon.
        func.append(token.spelling)
        token = next_token(tokiter)
      if token and token.spelling == "{":
        # Function with a body.
        lbr, rbr = 1, 0
        while token and lbr != rbr:
          func.append(token.spelling)
          token = next_token(tokiter)
          if token and token.spelling == "{":
            lbr += 1
          elif token and token.spelling == "}":
            rbr += 1
        if token:
          func.append(token.spelling)
        functions.append(' '.join(func))
        token = next_token(tokiter)
      else:
        # Just a function declaration.
        token = next_token(tokiter)
    else:
      token = next_token(tokiter)
  return functions

def DeriveSourceVocab(src: str,
                      token_list: typing.Set[str],
                      suffix: str,
                      cflags: typing.List[str],
                      ) -> typing.Dict[str, str]:
  """Pass source code through clang's lexer and return set of tokens.

  Args:
    src: The source code to compile.
    token_list: External set of grammar tokens for target language.
    suffix: The suffix to append to the source code temporary file. E.g. '.c'
      for a C program.
    cflags: A list of flags to be passed to clang.

  Returns:
    Set of unique source code tokens

  Raises:
    ValueError: In case of an error.
  """
  builtin_cflags = ["-S", "-emit-llvm", "-o", "-"]
  try:
    tdir = FLAGS.local_filesystem
  except Exception:
    tdir = None
  with tempfile.NamedTemporaryFile(
    "w", prefix="clgen_preprocessors_clang_", suffix=suffix, dir = tdir
  ) as f:
    f.write(src)
    f.flush()
    try:
      unit = clang.cindex.TranslationUnit.from_source(f.name, args = builtin_cflags + cflags)
    except clang.cindex.TranslationUnitLoadError as e:
      raise ValueError(e)

    tokens = {}
    for ch in string.printable:
      # Store all printable characters as char-based, to save time iterating literals.
      tokens["{}-char-based".format(ch)] = ''
    for idx, t in enumerate(unit.get_tokens(extent = unit.cursor.extent)):
      str_t = str(t.spelling)
      if str_t in token_list or t.kind in {clang.cindex.TokenKind.KEYWORD, clang.cindex.TokenKind.PUNCTUATION}:
        tokens[str_t] = ' '
      else:
        if t.kind != clang.cindex.TokenKind.LITERAL and clang.cindex.Cursor.from_location(unit, t.extent.end).kind not in {clang.cindex.CursorKind.CALL_EXPR}:
          tokens[str_t] = ' '

    return tokens

def AtomizeSource(src: str,
                  vocab: typing.Set[str],
                  suffix: str,
                  cflags: typing.List[str],
                  ) -> typing.List[str]:
  """
  Split source code into token atoms with clang's lexer.

  Args:
    src: The source code to compile.
    vocab: Optional set of learned vocabulary of tokenizer.
    suffix: The suffix to append to the source code temporary file. E.g. '.c'
      for a C program.
    cflags: A list of flags to be passed to clang.

  Returns:
    Source code as a list of tokens.

  Raises:
    ValueError: In case of an error.
  """
  builtin_cflags = ["-S", "-emit-llvm", "-o", "-"]
  try:
    tdir = FLAGS.local_filesystem
  except Exception:
    tdir = None
  with tempfile.NamedTemporaryFile(
    "w", prefix="clgen_preprocessors_clang_", suffix=suffix, dir = tdir
  ) as f:
    f.write(src)
    f.flush()
    try:
      unit = clang.cindex.TranslationUnit.from_source(f.name, args = builtin_cflags + cflags)
    except clang.cindex.TranslationUnitLoadError as e:
      raise ValueError(e)
    tokens = []
    lookout_metaToken, cmt = False, None
    for idx, t in enumerate(unit.get_tokens(extent = unit.cursor.extent)):
      str_t = t.spelling
      if str_t in {'START', 'MASK', 'HOLE', 'END', 'PAD'} and len(tokens) == 0:
        l.logger().warn("Please inspect the following code, having triggered a meta token existence without left brace preceding:")
        l.logger().warn(src)
      if str_t in {'START', 'MASK', 'HOLE', 'END', 'PAD'} and len(tokens) > 0 and tokens[-1] == '[':
        cmt = str_t
        lookout_metaToken = True
      elif str_t in vocab:
        if lookout_metaToken and str_t == ']':
          tokens[-1] = "[{}]".format(cmt)
          lookout_metaToken = False
        else:
          tokens.append(str(t.spelling))
      else:
        for ch in str_t:
          tokens.append("{}-char-based".format(ch))

    return tokens

def GreweFeatureExtraction(src: str,
                           suffx: str,
                           cflags: typing.List[str]
                           ) -> typing.Dict[str, float]:
  """
  !!! Under construction.
  """
  builtin_cflags = ["-S", "-emit-llvm", "-o", "-"]
  try:
    tdir = FLAGS.local_filesystem
  except Exception:
    tdir = None
  with tempfile.NamedTemporaryFile(
    "w", prefix="clgen_preprocessors_clang_", suffix=suffix, dir = tdir
  ) as f:
    f.write(src)
    f.flush()
    try:
      unit = clang.cindex.TranslationUnit.from_source(f.name, args = builtin_cflags + cflags)
    except clang.cindex.TranslationUnitLoadError as e:
      return None

  def next_token(token_iter):
    """Return None if iterator is consumed."""
    try:
      return next(token_iter)
    except StopIteration:
      return None

  feat_vec = {
    'comp': 0.0,
    'rational': 0.0,
    'mem': 0.0,
    'localmem': 0.0,
    'coalesced': 0.0,
    'atomic': 0.0,
    'F2:coalesced/mem': 0.0,
    'F4:comp/mem': 0.0,
  }
  tokiter = unit.get_tokens(extent = unit.cursor.extent)
  token = next_token(tokiter)

  while token:
    # Do sth with token
    cur = clang.cindex.Cursor.from_location(unit, token.extent.start)

  return {}