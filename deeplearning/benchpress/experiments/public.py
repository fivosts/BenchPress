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
"""This file defines the decorator for marking an evaluator function."""
import typing

PreprocessorFunction = typing.Callable[[str], str]

def evaluator(func: PreprocessorFunction) -> PreprocessorFunction:
  """A decorator which marks a function as an evaluator.

  Args:
    func: The preprocessor function to decorate.

  Returns:
    The decorated preprocessor function.

  Raises:
    InternalError: If the function being wrapped does not have the signature
      'def func(text: str) -> str:'.
  """
  type_hints = typing.get_type_hints(func)
  if not type_hints == {"return": type(None)}:
    raise SystemError(
      f"Preprocessor {func.__name__} does not have signature "
      f'"def {func.__name__}(text: str) -> str".'
      f"or"
      f'"def {func.__name__}(text: str) -> typing.List[str]".'
    )
  func.__dict__["is_evaluator"] = True
  return func
