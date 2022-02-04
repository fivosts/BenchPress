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
