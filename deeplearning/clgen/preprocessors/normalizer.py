"""Python entry point to the clang_rewriter binary."""
import os
import subprocess
import tempfile
import typing

from deeplearning.clgen.util import environment
from absl import flags
from deeplearning.clgen.util import logging as l

FLAGS = flags.FLAGS

# Path of the clang rewriter binary.
CLANG_REWRITER     = environment.CLANG_REWRITER
SEQ_CLANG_REWRITER = environment.SEQ_CLANG_REWRITER

# On Linux we must preload the LLVM libraries.
CLANG_REWRITER_ENV = os.environ.copy()
libclang = os.path.join(environment.LLVM, "lib/libclang.so")
liblto   = os.path.join(environment.LLVM, "lib/libLTO.so")
CLANG_REWRITER_ENV["LD_PRELOAD"] = f"{libclang}:{liblto}"


def NormalizeIdentifiers(text: str,
                         suffix: str,
                         cflags: typing.List[str],
                         sequential_rewrite = False,
                         timeout_seconds: int = 60
                         ) -> str:
  """Normalize identifiers in source code.

  An LLVM rewriter pass which renames all functions and variables with short,
  unique names. The variables and functions defined within the input text
  are rewritten, with the sequence 'A', 'B', ... 'AA', 'AB'... being used for
  function names, and the sequence 'a', 'b', ... 'aa', 'ab'... being used for
  variable names. Functions and variables which are defined in #include files
  are not renamed. Undefined function and variable names are not renamed.

  Args:
    text: The source code to rewrite.
    suffix: The suffix to append to the source code temporary file. E.g. '.c'
      for a C program.
    cflags: A list of flags to be passed to clang.
    timeout_seconds: The number of seconds to allow before killing the rewriter.

  Returns:
    Source code with identifier names normalized.

  Raises:
    RewriterException: If rewriter found nothing to rewrite.
    ClangTimeout: If rewriter fails to complete within timeout_seconds.
  """
  try:
    tdir = FLAGS.local_filesystem
  except Exception:
    tdir = None
  with tempfile.NamedTemporaryFile("w", suffix=suffix, dir = tdir) as f:
    f.write(text)
    f.flush()
    REWRITER = SEQ_CLANG_REWRITER if sequential_rewrite else CLANG_REWRITER
    cmd = (
      ["timeout", "-s9", str(timeout_seconds), str(REWRITER), f.name]
      + ["-extra-arg=" + x for x in cflags]
      + ["--"]
    )
    l.logger().debug("$ {}{}".format(
                    f'LD_PRELOAD={CLANG_REWRITER_ENV["LD_PRELOAD"]} '
                    if "LD_PRELOAD" in CLANG_REWRITER_ENV
                    else "",
                    " ".join(cmd),
      ),
    )
    
    process = subprocess.Popen(
      cmd,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      universal_newlines=True,
      env=CLANG_REWRITER_ENV,
    )
    stdout, stderr = process.communicate()
    l.logger().debug("stdout: {}".format(stdout))
    l.logger().debug("stderr: {}".format(stderr))
  # If there was nothing to rewrite, the rewriter exits with error code:
  EUGLY_CODE = 204
  if process.returncode == EUGLY_CODE:
    # Propagate the error:
    raise ValueError(stderr)
  elif process.returncode == 9:
    raise ValueError(
      f"clang_rewriter failed to complete after {timeout_seconds}s"
    )
  # The rewriter process can still fail because of some other compilation
  # problem, e.g. for some reason the 'enable 64bit support' pragma which should
  # be included in the shim isn't being propogated correctly to the rewriter.
  # However, the rewriter will still correctly process the input, so we ignore
  # all error codes except the one we care about (EUGLY_CODE).
  return stdout
