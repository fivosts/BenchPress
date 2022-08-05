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
"""This file defines the decorator for marking a CLgen preprocessor function."""
import typing


from absl import flags

FLAGS = flags.FLAGS

# Type hint for a preprocessor function. See @clgen_preprocess for details.
PreprocessorFunction = typing.Callable[[str], str]


def clgen_preprocessor(func: PreprocessorFunction) -> PreprocessorFunction:
  """A decorator which marks a function as a CLgen preprocessor.

  A CLgen preprocessor is accessible using GetPreprocessFunction(), and is a
  function which accepts a single parameter 'text', and returns a string.
  Type hinting is used to ensure that any function wrapped with this decorator
  has the appropriate argument and return type. If the function does not, an
  InternalError is raised at the time that the module containing the function
  is imported.

  Args:
    func: The preprocessor function to decorate.

  Returns:
    The decorated preprocessor function.

  Raises:
    InternalError: If the function being wrapped does not have the signature
      'def func(text: str) -> str:'.
  """
  type_hints = typing.get_type_hints(func)
  if not (type_hints == {"text": str, "return": str} or type_hints == {"text": str, "return": typing.List[str]} or type_hints == {"text": str, "return": typing.List[typing.Tuple[str, str]]}):
    raise SystemError(
      f"Preprocessor {func.__name__} does not have signature "
      f'"def {func.__name__}(text: str) -> str".'
      f"or"
      f'"def {func.__name__}(text: str) -> typing.List[str]".'
    )
  func.__dict__["is_clgen_preprocessor"] = True
  return func
