import subprocess
import tempfile

from deeplearning.clgen.util import environment

CLGEN_FEATURES = environment.CLGEN_FEATURES
CLGEN_REWRITER = environment.CLGEN_REWRITER

def kernel_features(src: str, *extra_args, file_name = None) -> str:
  """
  Invokes clgen_features extractor on a single kernel.

  Params:
    src: (str) Kernel in string format.
    extra_args: Extra compiler arguments passed to feature extractor.
  Returns:
    Feature vector and diagnostics in str format.
  """
  if file_name:
    tfile = lambda: open("/tmp/{}.cl".format(file_name), 'w')
  else:
    tfile = lambda: tempfile.NamedTemporaryFile(
      'w', prefix = "feature_extractor_", suffix = '.cl'
    )
  with tfile() as f:
    f.write(src)
    f.flush()
    cmd = [str(CLGEN_FEATURES), f.name]

    process = subprocess.Popen(
      cmd,
      stdout = subprocess.PIPE,
      stderr = subprocess.PIPE,
      universal_newlines = True,
    )
    stdout, stderr = process.communicate()
  return stdout
