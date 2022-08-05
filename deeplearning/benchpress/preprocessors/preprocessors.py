# coding=utf-8
# Copyright 2022 Foivos Tsimpourlas.
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
"""Preprocess source code files for machine learning."""
import importlib
import pathlib
import typing
from importlib import util as importlib_util
from detect_secrets import main as secrets_main
from detect_secrets.plugins.common import initialize as secrets_init

from deeplearning.benchpress.preprocessors import public
from absl import flags
from deeplearning.benchpress.util import logging as l

FLAGS = flags.FLAGS

# Import type alias to public module.
PreprocessorFunction = public.PreprocessorFunction


def _ImportPreprocessorFromFile(module_path: pathlib.Path, function_name: str):
  """Import module from an absolute path to file, e.g. '/foo/bar.py'."""
  if not module_path.is_file():
    raise ValueError(f"File not found: {module_path}")
  try:
    spec = importlib_util.spec_from_file_location("module", str(module_path))
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
  except ImportError as e:
    raise ImportError(f"Failed to import module {module_path}: {e}")
  if not hasattr(module, function_name):
    raise AttributeError(
      f"Function {function_name} not found in module {module_path}"
    )
  return getattr(module, function_name)


def _ImportPreprocessorFromModule(module_name: str, function_name: str):
  """Import module from a fully qualified module name, e.g. 'foo.bar'."""
  try:
    module = importlib.import_module(module_name)
  except (ModuleNotFoundError, AttributeError):
    raise AttributeError(f"Module {module_name} not found.")
  if not hasattr(module, function_name):
    raise AttributeError(
      f"Function {function_name} not found in module {module_name}"
    )
  function_ = getattr(module, function_name)
  if not function_.__dict__.get("is_clgen_preprocessor"):
    raise AttributeError(
      f"Preprocessor {function_name} not decorated with @clgen_preprocessor"
    )
  return function_


def GetPreprocessorFunction(name: str) -> public.PreprocessorFunction:
  """Lookup a preprocess function by name.

  A preprocessor is a function which takes a single argument 'text' of type str,
  and returns a str. The name is in the form <module>:<name>, where <name> is
  the name of a python function, and <module> is either a fully qualified module
  name, or an absolute path to the module file. For example, the name
  'deeplearning.clgen.preprocessors.cxx:Compile' will return the function
  'Compile' in the module 'deeplearning.clgen.preprocessors.cxx'. The name
  '/tmp/my_preprocessors.py:Transform' will return the function Transform() in
  the module defined at '/tmp/my_preprocessors.py'.

  Args:
    name: The name of the preprocessor to get.

  Returns:
    The python preprocessor function.

  Raises:
    UserError: If the requested name cannot be found or is not a
      @clgen_preprocessor decorated function.
  """
  components = name.split(":")
  if len(components) != 2:
    raise ValueError(f"Invalid preprocessor name {name}")
  module_name, function_name = components
  if module_name[0] == "/":
    return _ImportPreprocessorFromFile(pathlib.Path(module_name), function_name)
  else:
    return _ImportPreprocessorFromModule(module_name, function_name)


def Preprocess(text: str, preprocessors: typing.List[str]) -> str:
  """Preprocess a text using the given preprocessor pipeline.

  If preprocessing succeeds, the preprocessed text is returned.

  Args:
    text: The input to be preprocessed.
    preprocessors: The list of preprocessor functions to run. These will be
      passed to GetPreprocessorFunction() to resolve the python implementations.

  Returns:
    Preprocessed source input as a string.

  Raises:
    ValueError, UnicodeError
  """
  preprocessor_functions = [GetPreprocessorFunction(p) for p in preprocessors]

  def PreprocessSingle(text, preprocessors: typing.List[public.clgen_preprocessor]):
    """
      This recursive generator is an elegant way to manage preprocessors that decide to split one text into many,
      without destroying the whole preprocessing pipeline. The generator creates a stream of pre-processed files,
      in case one - or more - preprocessing functions return a list of strings.
    """
    preprocessor_success = True
    for idx, pr in enumerate(preprocessors):
      if isinstance(text, str):
        try:
          text = pr(text)
        except ValueError as e:
          yield str(e), False
          return
        except UnicodeError:
          yield "UnicodeError", False
          return
        except OSError:
          yield "OSError: Memory Allocation", False
          return
      elif isinstance(text, list):
        for item in text:
          for t, pc in PreprocessSingle(item, preprocessors[idx:]):
            yield t, pc
        return
      else:
        raise TypeError("Preprocessor has returned type: {}".format(type(text)))

    if isinstance(text, str):
      yield text, preprocessor_success
    elif isinstance(text, list):
      for i in text:
        yield i, preprocessor_success

  return PreprocessSingle(text, preprocessor_functions)


def PreprocessFile(
  path: str, preprocessors: typing.List[str], inplace: bool
) -> str:
  """Preprocess a file and optionally update it.

  Args:
    text: The input to be preprocessed.
    preprocessors: The list of preprocessor functions to run. These will be
      passed to GetPreprocessorFunction() to resolve the python implementations.
    inplace: If True, the input file is overwritten with the preprocessed code,
      unless the preprocessing fails. If the preprocessing fails, the input
      file is left unmodified.

  Returns:
    Preprocessed source input as a string.

  Raises:
    ValueError
  """
  with open(path) as infile:
    contents = infile.read()
  preprocessed = Preprocess(contents, preprocessors)
  if inplace:
    with open(path, "w") as outfile:
      outfile.write(preprocessed)
  return preprocessed


@public.clgen_preprocessor
def RejectSecrets(text: str) -> str:
  """Test for secrets such as private keys in a text.

  Args:
    text: The text to check.

  Returns:
    The unmodified text.

  Raises:
    ValueError: In case the text contains secrets.
  """

  args = secrets_main.parse_args(["scan"])
  plugins = secrets_init.from_parser_builder(args.plugins, exclude_lines_regex="")
  for plugin in plugins:
    if plugin.analyze_string(text, 0, "does_not_matter"):
      raise ValueError(plugin.__class__.__name__)

  return text
