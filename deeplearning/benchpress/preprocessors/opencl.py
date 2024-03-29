# coding=utf-8
# Copyright 2022 Foivos Tsimpourlas and Chris Cummins.
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
"""Preprocessor passes for the OpenCL programming language."""
import typing
import os
import pathlib
import io
import subprocess
import tempfile
import math
import pandas as pd

from deeplearning.benchpress.util import environment
from deeplearning.benchpress.util import crypto
from deeplearning.benchpress.preprocessors import clang
from deeplearning.benchpress.preprocessors import normalizer
from deeplearning.benchpress.preprocessors import public

from deeplearning.benchpress.util import logging as l

from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
  "verbose_cldrive",
  False,
  "Select to print verbose command messages for cldrive."
)

# LibCLC
LIBCLC         = environment.LIBCLC
# OpenCL standard headers
OPENCL_HEADERS = environment.OPENCL_HEADERS
# Auxiliary .cl kernels that may need be included
AUX_INCLUDE    = environment.AUX_INCLUDE
# CLDrive executable, if exists.
CLDRIVE        = environment.CLDRIVE
CL_PLATFORMS   = None

CL_H           = os.path.join(OPENCL_HEADERS, "CL/cl.h")
OPENCL_H       = os.path.join(environment.DATA_CL_INCLUDE, "opencl.h")
OPENCL_C_H     = os.path.join(environment.DATA_CL_INCLUDE, "opencl-c.h")
OPENCL_C_BASE  = os.path.join(environment.DATA_CL_INCLUDE, "opencl-c-base.h")
SHIMFILE       = os.path.join(environment.DATA_CL_INCLUDE, "opencl-shim.h")
STRUCTS        = os.path.join(environment.DATA_CL_INCLUDE, "structs.h")

def GetClangArgs(use_shim: bool, use_aux_headers: bool, extra_args: typing.List[str] = []) -> typing.List[str]:
  """Get the arguments to pass to clang for handling OpenCL.

  Args:
    use_shim: If true, inject the shim OpenCL header.
    error_limit: The number of errors to print before arboting

  Returns:
    A list of command line arguments to pass to Popen().
  """
  args = [
    "-xcl",
    "--target=nvptx64-nvidia-nvcl",
    "-cl-std=CL2.0",
    "-ferror-limit=0",
    "-include{}".format(OPENCL_C_H),
    "-include{}".format(OPENCL_C_BASE),
    "-include{}".format(CL_H),
    "-I{}".format(str(OPENCL_HEADERS)),
    "-I{}".format(str(LIBCLC)),
    "-Wno-everything",
    "-O1",
  ]
  if use_aux_headers:
    args += [
      "-include{}".format(STRUCTS),
      "-I{}".format(str(AUX_INCLUDE)),
    ]
  if use_shim:
    args += ["-include", str(SHIMFILE)]
  return args + extra_args

def getOpenCLPlatforms() -> None:
  """
  Identify compatible OpenCL platforms for current system.
  """
  global CL_PLATFORMS
  CL_PLATFORMS = {
    'CPU': None,
    'GPU': None,
  }
  try:
    cmd = subprocess.Popen(
      "{} --clinfo".format(CLDRIVE).split(),
      stdout = subprocess.PIPE,
      stderr = subprocess.PIPE,
      universal_newlines = True,
    )
    stdout, stderr = cmd.communicate()
    if stderr:
      raise ValueError(stderr)
  except Exception as e:
    l.logger().error(cmd)
    l.logger().error(e)
  lines = stdout.split('\n')
  for line in lines:
    if line and line[:3] == "GPU" and not CL_PLATFORMS['GPU']:
      CL_PLATFORMS['GPU'] = line
    elif line and line[:3] == "CPU" and not CL_PLATFORMS['CPU']:
      CL_PLATFORMS['CPU'] = line
  return

def _ClangPreprocess(text: str, use_shim: bool, use_aux_headers: bool, extra_args: typing.List[str]) -> str:
  """Private preprocess OpenCL source implementation.

  Inline macros, removes comments, etc.

  Args:
    text: OpenCL source.
    use_shim: Inject shim header.

  Returns:
    Preprocessed source.
  """
  return clang.Preprocess(text, GetClangArgs(use_shim = use_shim, use_aux_headers = use_aux_headers, extra_args = extra_args))

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

def DeriveSourceVocab(text: str, token_list: typing.Set[str] = set(), extra_args: typing.List[str] = []) -> typing.Dict[str, str]:
  """Pass CL code through clang's lexer and return set of
  tokens with appropriate delimiters for vocabulary construction.

  Args:
    text: Source code.
    token_list: Optional external list of tokens for opencl grammar.

  Returns:
    Set of unique source code tokens.
  """
  return clang.DeriveSourceVocab(text, token_list, ".cl", GetClangArgs(use_shim = False, use_aux_headers = True, extra_args = extra_args))

def AtomizeSource(text: str, vocab: typing.Set[str], extra_args: typing.List[str] = []) -> typing.List[str]:
  """
  Atomize OpenCL source with clang's lexer into token atoms.

  Args:
    text: The source code to compile.
    vocab: Optional set of learned vocabulary of tokenizer.

  Returns:
    Source code as a list of tokens.
  """
  return clang.AtomizeSource(text, vocab, ".cl", GetClangArgs(use_shim = False, use_aux_headers = True, extra_args = extra_args))

def ContentHash(src: str) -> str:
  """
  Re-write code with deterministic, sequential rewriter, remove whitespaces and new lines
  and calculate the hash of the string.

  Args:
    src: The source code to compute.

  Returns:
    256-bit hash of pure source code string.
  """
  rw = SequentialNormalizeIdentifiers(src)
  return crypto.sha256_str(rw.replace(" ", "").replace("\n", ""))

def IRContentHash(src: str, header_file = None, use_aux_headers: bool = True) -> str:
  """
  Collect optimized LLVM-IR of source code and compute its hash.

  Args:
    src: The source code to compute.

  Returns:
    256-bit hash of pure source code string.
  """
  bc = CompileLlvmBytecode(src, header_file = header_file, use_aux_headers = use_aux_headers)
  return crypto.sha256_str(''.join(bc.split('\n')[2:]))

def RunCLDrive(src: str,
               header_file: str = None,
               num_runs   : int = 1000,
               gsize      : int = 4096,
               lsize      : int = 1024,
               extra_args : typing.List[str] = [],
               timeout    : int = 0
               ) -> str:
  """
  If CLDrive executable exists, run it over provided source code.
  """
  if not CLDRIVE:
    l.logger().warn("CLDrive executable has not been found. Skipping CLDrive execution.")
    return ""

  global CL_PLATFORMS
  if not CL_PLATFORMS:
    getOpenCLPlatforms()

  try:
    tdir = FLAGS.local_filesystem
  except Exception:
    tdir = None

  with tempfile.NamedTemporaryFile("w", prefix="benchpress_opencl_cldrive", suffix = '.cl', dir = tdir) as f:
    if header_file:
      with tempfile.NamedTemporaryFile("w", prefix="benchpress_opencl_clheader", suffix = '.h', dir = tdir) as hf:
        f.write("#include \"{}\"\n{}".format(pathlib.Path(hf.name).resolve().name, src))
        f.flush()
        hf.write(header_file)
        hf.flush()
        cmd = "{} {} --srcs={} --cl_build_opt=\"-I{}{}\" --num_runs={} --gsize={} --lsize={} --envs={},{}".format(
                "timeout -s9 {}".format(timeout) if timeout > 0 else "",
                CLDRIVE,
                f.name,
                pathlib.Path(hf.name).resolve().parent,
                ",{}".format(",".join(extra_args)) if len(extra_args) > 0 else "",
                num_runs,
                gsize,
                lsize,
                CL_PLATFORMS['CPU'],
                CL_PLATFORMS['GPU']
              )
        if FLAGS.verbose_cldrive:
          print(cmd)
          print(src)
        proc = subprocess.Popen(
          cmd.split(),
          stdout = subprocess.PIPE,
          stderr = subprocess.PIPE,
          universal_newlines = True,
        )
        stdout, stderr = proc.communicate()
    else:
      f.write(src)
      f.flush()
      cmd = "{} {} --srcs={} {} --num_runs={} --gsize={} --lsize={} --envs={},{}".format(
              "timeout -s9 {}".format(timeout) if timeout > 0 else "",
              CLDRIVE,
              f.name,
              "--cl_build_opt={}".format(",".join(extra_args)) if len(extra_args) > 0 else "",
              num_runs,
              gsize,
              lsize,
              CL_PLATFORMS['CPU'],
              CL_PLATFORMS['GPU']
            )
      if FLAGS.verbose_cldrive:
        print(cmd)
        print(src)
      proc = subprocess.Popen(
        cmd.split(),
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        universal_newlines = True,
      )
      try:
        stdout, stderr = proc.communicate()
      except UnicodeDecodeError:
        return "", ""
    if proc.returncode == 9:
      stderr = "TIMEOUT"
  return stdout, stderr

def CollectCLDriveLabel(df: pd.DataFrame, stdout: str, stderr: str) -> str:
  """
  Read data from CLDrive execution and compute label.
  """
  cpu_error = None
  gpu_error = None

  if stderr == "TIMEOUT":
    return "TIMEOUT"
  if df is None:
    return "I/O_ERROR"

  try:
    avg_time_cpu_ns = (df[df['device'].str.contains("CPU")].transfer_time_ns.mean() + df[df['device'].str.contains("CPU")].kernel_time_ns.mean())
    avg_time_gpu_ns = (df[df['device'].str.contains("GPU")].transfer_time_ns.mean() + df[df['device'].str.contains("GPU")].kernel_time_ns.mean())
  except Exception:
    avg_time_cpu_ns = None
    avg_time_gpu_ns = None

  if avg_time_cpu_ns is None or avg_time_gpu_ns is None or math.isnan(avg_time_cpu_ns) or math.isnan(avg_time_gpu_ns):
    label = "ERR"
    if stdout == "":
      cpu_error = "NO_STDOUT"
      gpu_error = "NO_STDOUT"
      label = "CPU-{}_GPU-{}".format(cpu_error, gpu_error)
    elif "CL_OUT_OF_RESOURCES" in stderr:
      cpu_error = "CL_OUT_OF_RESOURCES"
      gpu_error = "CL_OUT_OF_RESOURCES"
      label = "CPU-{}_GPU-{}".format(cpu_error, gpu_error)
    elif df is not None:
      try:
        cpu_error = df[df['device'].str.contains("CPU")].outcome[0]
        if cpu_error == "CL_ERROR" and "-9999" in stderr:
          cpu_error = "INVALID_BUFFER_READ_WRITE"
      except KeyError:
        cpu_error = stderr
      except ValueError:
        cpu_error = stderr
      except Exception:
        cpu_error = stderr
      try:
        gpu_error = df[df['device'].str.contains("GPU")].outcome[1]
        if gpu_error == "CL_ERROR" and "-9999" in stderr:
          gpu_error = "INVALID_BUFFER_READ_WRITE"
      except KeyError:
        gpu_error = stderr
      except ValueError:
        gpu_error = stderr
      except Exception:
        gpu_error = stderr
      label = "CPU-{}_GPU-{}".format(cpu_error, gpu_error)
  else:
    label = "GPU" if avg_time_cpu_ns > avg_time_gpu_ns else "CPU"

  # if label == "ERR" or cpu_error == "CL_ERROR" or gpu_error == "CL_ERROR":
  #   l.logger().warn(stdout)
  #   l.logger().warn(stderr)
  return label

def CLDrivePretty(src: str,
                  header_file = None,
                  num_runs: int = 5,
                  gsize: int = 4096,
                  lsize: int = 1024,
                  timeout: int = 0
                  ) -> typing.Tuple[pd.DataFrame, str]:
  """
  Run CLDrive with given configuration but pretty print stdout and stderror.
  """
  stdout, stderr = RunCLDrive(src, header_file = header_file, num_runs = num_runs, gsize = gsize, lsize = lsize, timeout = timeout)
  for x in stdout.split('\n'):
    print(x)
  for x in stderr.split('\n'):
    print(x)
  return stdout, stderr

def CLDriveDataFrame(src: str,
                     header_file: str = None,
                     num_runs   : int = 5,
                     gsize      : int = 4096,
                     lsize      : int = 1024,
                     extra_args : typing.List[str] = [],
                     timeout    : int = 0
                     ) -> typing.Tuple[pd.DataFrame, str]:
  """
  Run CLDrive with given configuration and return pandas dataframe along with collected label.
  """
  stdout, stderr = RunCLDrive(src, header_file = header_file, num_runs = num_runs, gsize = gsize, lsize = lsize, extra_args = extra_args, timeout = timeout)
  try:
    df = pd.read_csv(io.StringIO(stdout), sep = ",")
  except Exception as e:
    df = None
  return df, CollectCLDriveLabel(df, stdout, stderr)

def CLDriveNumBytes(src: str,
                    header_file = None,
                    gsize: int = 4096,
                    lsize: int = 1024,
                    timeout: int = 0) -> int:
  """
  Run CLDrive once for given configuration to identify number of transferred bytes.
  """
  stdout, stderr = RunCLDrive(src, header_file = header_file, num_runs = 5, gsize = gsize, lsize = lsize, timeout = timeout)


  try:
    df = pd.read_csv(io.StringIO(stdout), sep = ",")
  except pd.errors.EmptyDataError:
    return None
  label = CollectCLDriveLabel(df, stdout, stderr)
  return df[df['device'].str.contains("CPU")].transferred_bytes[0] if label in {"CPU", "GPU"} else None

def CLDriveExecutionTimes(src: str,
                          header_file = None,
                          num_runs: int = 1000,
                          gsize: int = 4096,
                          lsize: int = 1024,
                          timeout: int = 0
                          ) -> typing.Tuple[typing.List[int], typing.List[int], typing.List[int], typing.List[int]]:
  """
  Run CLDrive once for given configuration to identify number of transferred bytes.
  """
  stdout, stderr = RunCLDrive(src, header_file = header_file, num_runs = num_runs, gsize = gsize, lsize = lsize, timeout = timeout)
  try:
    df = pd.read_csv(io.StringIO(stdout), sep = ",")
    transfer_time_cpu  = df[df['device'].str.contains("CPU")].transfer_time_ns
    execution_time_cpu = df[df['device'].str.contains("CPU")].kernel_time_ns
    transfer_time_gpu  = df[df['device'].str.contains("GPU")].transfer_time_ns
    execution_time_gpu = df[df['device'].str.contains("GPU")].kernel_time_ns
  except pd.errors.EmptyDataError:
    # CSV is empty which means src failed miserably.
    transfer_time_cpu  = None
    execution_time_cpu = None
    transfer_time_gpu  = None
    execution_time_gpu = None
  except pd.errors.ParserError:
    # CSV is empty which means src failed miserably.
    transfer_time_cpu  = None
    execution_time_cpu = None
    transfer_time_gpu  = None
    execution_time_gpu = None

  return transfer_time_cpu, execution_time_cpu, transfer_time_gpu, execution_time_gpu

def CLDriveLabel(src: str,
                 header_file = None,
                 num_runs: int = 1000,
                 gsize: int = 4096,
                 lsize: int = 1024,
                 timeout: int = 0
                 ) -> str:
  """
  Run CLDrive on given configuration and compute whether it should run on CPU vs GPU based on where it will execute faster (transfer time + execution time).
  """
  stdout, stderr = RunCLDrive(src, header_file = header_file, num_runs = num_runs, gsize = gsize, lsize = lsize, timeout = timeout)
  df = None
  try:
    df = pd.read_csv(io.StringIO(stdout), sep = ",")
    avg_time_cpu_ns = (df[df['device'].str.contains("CPU")].transfer_time_ns.mean() + df[df['device'].str.contains("CPU")].kernel_time_ns.mean())
    avg_time_gpu_ns = (df[df['device'].str.contains("GPU")].transfer_time_ns.mean() + df[df['device'].str.contains("GPU")].kernel_time_ns.mean())
  except pd.errors.EmptyDataError:
    # CSV is empty which means src failed miserably.
    avg_time_cpu_ns   = None
    avg_time_gpu_ns   = None
  except pd.errors.ParserError:
    # Unexpected parsing error.
    avg_time_cpu_ns   = None
    avg_time_gpu_ns   = None

  return CollectCLDriveLabel(df, stdout, stderr)

@public.benchpress_preprocessor
def ClangPreprocess(text: str, extra_args = []) -> str:
  """Preprocessor OpenCL source.

  Args:
    text: OpenCL source to preprocess.

  Returns:
    Preprocessed source.
  """
  return _ClangPreprocess(text, False, True, extra_args = extra_args)


@public.benchpress_preprocessor
def ClangPreprocessWithShim(text: str, extra_args = []) -> str:
  """Preprocessor OpenCL source with OpenCL shim header injection.

  Args:
    text: OpenCL source to preprocess.

  Returns:
    Preprocessed source.
  """
  return _ClangPreprocess(text, True, True, extra_args = extra_args)

def CompileLlvmBytecode(text: str, header_file = None, use_aux_headers: bool = True, extra_args: typing.List[str] = []) -> str:
  """A preprocessor which attempts to compile the given code.

  Args:
    text: Code to compile.

  Returns:
    LLVM IR of input source code.
  """
  # We must override the flag -Wno-implicit-function-declaration from
  # GetClangArgs() to ensure that undefined functions are treated as errors.
  return clang.CompileLlvmBytecode(
    text,
    ".cl",
    GetClangArgs(use_shim = False, use_aux_headers = use_aux_headers, extra_args = extra_args),# + ["-Werror=implicit-function-declaration"],
    header_file = header_file,
  )

def CompileStdin(text: str, header_file = None, use_aux_headers: bool = True, extra_args: typing.List[str] = []) -> str:
  """A preprocessor which attempts to compile the given code.

  Args:
    text: Code to compile.

  Returns:
    LLVM IR of input source code.
  """
  # We must override the flag -Wno-implicit-function-declaration from
  # GetClangArgs() to ensure that undefined functions are treated as errors.
  return clang.CompileStdin(
    text,
    ".cl",
    GetClangArgs(use_shim = False, use_aux_headers = use_aux_headers, extra_args = extra_args),# + ["-Werror=implicit-function-declaration"],
    header_file = header_file,
  )

def HumanReadableBytecode(bc_path: pathlib.Path) -> str:
  """Run llvm-dis to disassemble binary bytecode file to human readable format.
  Args:
    bc_path: The path to bytecode.
  Returns:
    The textual LLVM byte code.
  """
  return clang.HumanReadableBytecode(bc_path)

def CompileOptimizer(text: str,
                     optimization    : typing.List[str],
                     timeout_seconds : int = 60,
                     header_file     : str = None,
                     use_aux_headers : bool = True,
                     extra_args      : typing.List[str] = []
                     ) -> str:
  """Compile source code to IR and apply optimization pass to source code.
  Args:
    src: The source code to compile.
    optimization: optimization pass to apply.
  Returns:
    Dictionary with 70-dimensional InstCount feature vector.
  """
  return clang.CompileOptimizer(
    src = text,
    suffix = ".cl",
    cflags = GetClangArgs(use_shim = False, use_aux_headers = use_aux_headers, extra_args = extra_args),
    optimization = optimization,
    header_file = header_file,
  )

def CompileOptimizerStdin(text: str,
                          optimization    : typing.List[str],
                          timeout_seconds : int = 60,
                          header_file     : str = None,
                          use_aux_headers : bool = True,
                          extra_args      : typing.List[str] = []
                          ) -> str:
  """Compile source code to IR and apply optimization pass to source code.
  Args:
    src: The source code to compile.
    optimization: optimization pass to apply.
  Returns:
    Dictionary with 70-dimensional InstCount feature vector.
  """
  return clang.CompileOptimizerStdin(
    src = text,
    suffix = ".cl",
    cflags = GetClangArgs(use_shim = False, use_aux_headers = use_aux_headers, extra_args = extra_args),
    optimization = optimization,
    header_file = header_file,
  )

def CompileOptimizerIR(bytecode: str,
                       optimization    : typing.List[str],
                       timeout_seconds : int = 60,
                       ) -> str:
  """Apply optimization pass to LLVM-IR bytecode file.
  Args:
    bytecode: The source code to optimize.
    optimization: optimization pass to apply.
  Returns:
    Dictionary with 70-dimensional InstCount feature vector.
  """
  return clang.CompileOptimizerIR(
    bytecode = bytecode,
    suffix   = ".ll",
    optimization = optimization,
  )

@public.benchpress_preprocessor
def Compile(text: str, header_file = None, use_aux_headers = True, extra_args = [], return_diagnostics = False) -> str:
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
    GetClangArgs(use_shim = False, use_aux_headers = use_aux_headers, extra_args = extra_args),# + ["-Werror=implicit-function-declaration"],
    header_file = header_file,
    return_diagnostics = return_diagnostics,
  )

@public.benchpress_preprocessor
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

@public.benchpress_preprocessor
def ExtractStructTypedefs(text: str) -> str:
  """
  Preprocessor extracts all struct type definitions.

  Args:
    text: The text to preprocess.

  Returns:
    The input text, with all whitespaces removed.
  """
  return _ExtractTypedefs(text, 'struct')

@public.benchpress_preprocessor
def ExtractUnionTypedefs(text: str) -> str:
  """
  Preprocessor extracts all union type definitions.

  Args:
    text: The text to preprocess.

  Returns:
    The input text, with all whitespaces removed.
  """
  return _ExtractTypedefs(text, 'union')

@public.benchpress_preprocessor
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

@public.benchpress_preprocessor
def InvertKernelSpecifier(text: str) -> str:
  """
  Inverts 'void kernel' specifier to 'kernel void'.

  Args:
    text: The text to preprocess.

  Returns:
    The input text, with all whitespaces removed.
  """
  return text.replace("void kernel ", "kernel void ")

@public.benchpress_preprocessor
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
          l.logger().warn(chunk)
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

@public.benchpress_preprocessor
def ExtractSingleKernelsHeaders(text: str) -> typing.List[typing.Tuple[str, str]]:
  """
  A preprocessor that splits a single source file
  to tuples of (single kernels, their global space)

  Args:
    text: The text to preprocess.

  Returns:
    List of tuples of kernels, global space (strings).
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
          l.logger().warn(chunk)
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
      actual_kernels.append((kernel_specifier + chunk[:chunk_idx], ''.join(global_space)))
      if ''.join(chunk[chunk_idx:]) != '':
        # All the rest below are appended to global_space
        global_space.append(chunk[chunk_idx:])

  return actual_kernels

@public.benchpress_preprocessor
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
      is_declaration = False
      # Given this preprocessor is called after compile,
      # we are certain that brackets will be paired
      num_lbrack, num_rbrack, chunk_idx = 0, 0, 0
      while ((num_lbrack  == 0 
      or      num_lbrack  != num_rbrack)
      and     chunk_idx   <  len(chunk)):

        try:
          cur_tok = chunk[chunk_idx]
        except IndexError:
          l.logger().warn(chunk)
        if cur_tok == ";" and num_lbrack == 0:
          is_declaration = True
          break
        elif   cur_tok == "{":
          num_lbrack += 1
        elif cur_tok == "}":
          num_rbrack += 1
        chunk_idx += 1
      if not is_declaration:
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

@public.benchpress_preprocessor
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

@public.benchpress_preprocessor
def NormalizeIdentifiers(text: str, extra_args = []) -> str:
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
    text, ".cl", GetClangArgs(use_shim = False, use_aux_headers = True, extra_args = extra_args)
  )

@public.benchpress_preprocessor
def SequentialNormalizeIdentifiers(text: str, extra_args = []) -> str:
  """Normalize identifiers sequentially in OpenCL source code.

  Args:
    text: The source code to rewrite.

  Returns:
    Source code with identifier names normalized.

  Raises:
    RewriterException: If rewriter found nothing to rewrite.
    ClangTimeout: If rewriter fails to complete within timeout_seconds.
  """
  return normalizer.NormalizeIdentifiers(
    text, ".cl", GetClangArgs(use_shim = False, use_aux_headers = True, extra_args = extra_args), sequential_rewrite = True
  )

@public.benchpress_preprocessor
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

@public.benchpress_preprocessor
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


@public.benchpress_preprocessor
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
