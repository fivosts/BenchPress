<<<<<<< HEAD:labm8/py/labtypes.py
# Copyright 2014-2020 Chris Cummins <chrisc.101@gmail.com>.
=======
# Copyright 2014-2019 Chris Cummins <chrisc.101@gmail.com>.
>>>>>>> 77b550945... Relicense labm8 under Apache 2.0.:labm8/labtypes.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Python type utilities.
"""
import inspect
import itertools
import sys
import typing
from collections.abc import Mapping

from six import string_types


def is_str(s):
  """
  Return whether variable is string type.

  On python 3, unicode encoding is *not* string type. On python 2, it is.

  Arguments:
      s: Value.

  Returns:
      bool: True if is string, else false.
  """
  return isinstance(s, string_types)


def is_dict(obj):
  """
  Check if an object is a dict.
  """
  return isinstance(obj, dict)


def is_seq(obj):
  """
  Check if an object is a sequence.
  """
<<<<<<< HEAD:labm8/py/labtypes.py
  return (
    not is_str(obj)
    and not is_dict(obj)
    and (hasattr(obj, "__getitem__") or hasattr(obj, "__iter__"))
  )
=======
  return (not is_str(obj) and not is_dict(obj) and
          (hasattr(obj, '__getitem__') or hasattr(obj, '__iter__')))
>>>>>>> 49340dc00... Auto-format labm8 python files.:labm8/labtypes.py


def flatten(lists):
  """
  Flatten a list of lists.
  """
  return [item for sublist in lists for item in sublist]


def update(dst, src):
  """
  Recursively update values in dst from src.

  Unlike the builtin dict.update() function, this method will descend into
  nested dicts, updating all nested values.

  Arguments:
      dst (dict): Destination dict.
      src (dict): Source dict.

  Returns:
      dict: dst updated with entries from src.
  """
  for k, v in src.items():
    if isinstance(v, Mapping):
      r = update(dst.get(k, {}), v)
      dst[k] = r
    else:
      dst[k] = src[k]
  return dst


def dict_values(src):
  """
  Recursively get values in dict.

  Unlike the builtin dict.values() function, this method will descend into
  nested dicts, returning all nested values.

  Arguments:
      src (dict): Source dict.

  Returns:
      list: List of values.
  """
  for v in src.values():
    if isinstance(v, dict):
      for v in dict_values(v):
        yield v
    else:
      yield v


def get_class_that_defined_method(meth):
  """
  Return the class that defines a method.

  Arguments:
      meth (str): Class method.

  Returns:
      class: Class object, or None if not a class method.
  """
  if sys.version_info >= (3, 0):
    # Written by @Yoel http://stackoverflow.com/a/25959545
    if inspect.ismethod(meth):
      for cls in inspect.getmro(meth.__self__.__class__):
        if cls.__dict__.get(meth.__name__) is meth:
          return cls
      meth = meth.__func__  # fallback to __qualname__ parsing
    if inspect.isfunction(meth):
      cls = getattr(
<<<<<<< HEAD:labm8/py/labtypes.py
        inspect.getmodule(meth),
        meth.__qualname__.split(".<locals>", 1)[0].rsplit(".", 1)[0],
      )
=======
          inspect.getmodule(meth),
<<<<<<< HEAD:labm8/py/labtypes.py
          meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0])
>>>>>>> 150d66672... Auto format files.:labm8/labtypes.py
=======
          meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0],
      )
>>>>>>> 49340dc00... Auto-format labm8 python files.:labm8/labtypes.py
      if isinstance(cls, type):
        return cls
  else:
    try:
      # Writted by @Alex Martelli http://stackoverflow.com/a/961057
      for cls in inspect.getmro(meth.im_class):
        if meth.__name__ in cls.__dict__:
          return cls
    except AttributeError:
      return None
  return None


class ReprComparable(object):
  """
  An abstract class which may be inherited from in order to enable __repr__.
  """

  def __lt__(self, other):
    return str(self) < str(other)

  def __le__(self, other):
    return str(self) <= str(other)

  def __eq__(self, other):
    return str(self) == str(other)

  def __ne__(self, other):
    return str(self) != str(other)

  def __gt__(self, other):
    return str(self) > str(other)

  def __ge__(self, other):
    return str(self) >= str(other)


<<<<<<< HEAD:labm8/py/labtypes.py
<<<<<<< HEAD:labm8/py/labtypes.py
def PairwiseIterator(
  iterable: typing.Iterator[typing.Any],
) -> typing.Iterator[typing.Tuple[typing.Any, typing.Any]]:
=======
def PairwiseIterator(iterable: typing.Iterator[typing.Any]
=======
def PairwiseIterator(iterable: typing.Iterator[typing.Any],
>>>>>>> 49340dc00... Auto-format labm8 python files.:labm8/labtypes.py
                    ) -> typing.Iterator[typing.Tuple[typing.Any, typing.Any]]:
>>>>>>> 150d66672... Auto format files.:labm8/labtypes.py
  """Construct a pairwise iterator for a input generator.

  Given an iterator, produces an iterator of overlapping pairs from the input:
  s -> (s0,s1), (s1,s2), (s2, s3), ...

  Args:
    iterable: The input iterable. Once called, this iterable should not be
      used any more.

  Returns:
    An iterator of pairs.
  """
  # Split the iterable into two.
  a, b = itertools.tee(iterable)
  # Advance the second iterable by one.
  next(b, None)
  # Return the pairs.
  return zip(a, b)


def SetDiff(
<<<<<<< HEAD:labm8/py/labtypes.py
  a: typing.Iterator[typing.Any], b: typing.Iterator[typing.Any],
=======
    a: typing.Iterator[typing.Any],
    b: typing.Iterator[typing.Any],
>>>>>>> 49340dc00... Auto-format labm8 python files.:labm8/labtypes.py
) -> typing.List[typing.Any]:
  """Return the set difference between two sequences.

  Args:
    a: An iterable.
    b: An iterable.

  Returns:
    The difference between the elements in the two iterables as a set.
  """
  set_a = set(a)
  set_b = set(b)
  return (set_a - set_b).union(set_b - set_a)


def AllSubclassesOfClass(cls: typing.Type) -> typing.Set[typing.Type]:
  """Return the set of subclasses of a base class.

  This recursively returns all nested subclasses of a base class.

  Example:
    >>> class A(object): pass
    >>> class B(A): pass
    >>> class C(B): pass
    >>> AllSubclassesOfClass(A)
    {B, C}

  Args:
    cls: The class to return subclasses of.

  Returns:
    A set of class types.
  """
  return set(cls.__subclasses__()).union(
<<<<<<< HEAD:labm8/py/labtypes.py
<<<<<<< HEAD:labm8/py/labtypes.py
    [s for c in cls.__subclasses__() for s in AllSubclassesOfClass(c)],
  )


def Chunkify(
  iterable: typing.Iterable[typing.Any], chunk_size: int
) -> typing.Iterable[typing.List[typing.Any]]:
  """Split an iterable into chunks of a given size.

  Args:
    iterable: The iterable to split into chunks.
    chunk_size: The size of the chunks to return.

  Returns:
    An iterator over chunks of the input iterable.
  """
  i = iter(iterable)
  piece = list(itertools.islice(i, chunk_size))
  while piece:
    yield piece
    piece = list(itertools.islice(i, chunk_size))


def DeleteKeys(d, keys):
  """Delete the keys from the given dictionary, if present.

  Args:
    d: The dictionary to remove the keys from.
    keys: The list of keys to remove.

  Returns:
    The dictionary.
  """
  for key in keys:
    if key in d:
      del d[key]
  return d
=======
      [s for c in cls.__subclasses__() for s in AllSubclassesOfClass(c)])
>>>>>>> 8a82495b7... Add labm8.labtypes.AllSubclassesOfClass() method.:labm8/labtypes.py
=======
      [s for c in cls.__subclasses__() for s in AllSubclassesOfClass(c)],)
<<<<<<< HEAD:labm8/py/labtypes.py
>>>>>>> 49340dc00... Auto-format labm8 python files.:labm8/labtypes.py
=======


def Chunkify(iterable: typing.Iterable[typing.Any],
             chunk_size: int) -> typing.Iterable[typing.List[typing.Any]]:
  """Split an iterable into chunks of a given size.

  Args:
    iterable: The iterable to split into chunks.
    chunk_size: The size of the chunks to return.

  Returns:
    An iterator over chunks of the input iterable.
  """
  i = iter(iterable)
  piece = list(itertools.islice(i, chunk_size))
  while piece:
    yield piece
    piece = list(itertools.islice(i, chunk_size))
<<<<<<< HEAD:labm8/py/labtypes.py
>>>>>>> 8e1930394... Add labtypes.Chunkify() function.:labm8/labtypes.py
=======

def DeleteKeys(d, keys):
  """Delete the keys from the given dictionary, if present.

  Args:
    d: The dictionary to remove the keys from.
    keys: The list of keys to remove.

  Returns:
    The dictionary.
  """
  for key in keys:
    if key in d:
      del d[key]
  return d
>>>>>>> 4f357866c... Add two utility functions.:labm8/labtypes.py
