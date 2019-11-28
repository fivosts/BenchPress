<<<<<<< HEAD:labm8/py/io.py
# Copyright 2014-2020 Chris Cummins <chrisc.101@gmail.com>.
=======
# Copyright 2014-2019 Chris Cummins <chrisc.101@gmail.com>.
>>>>>>> 77b550945... Relicense labm8 under Apache 2.0.:labm8/io.py
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
"""Logging interface.
"""
import json

<<<<<<< HEAD:labm8/py/io.py
<<<<<<< HEAD:labm8/py/io.py
from labm8.py import system
=======
from phd.lib.labm8 import system
>>>>>>> 386c66354... Add 'phd' prefix to labm8 imports.:lib/labm8/io.py
=======
from labm8.py import system
>>>>>>> 8be094257... Move //labm8 to //labm8/py.:labm8/py/io.py


def colourise(colour, *args):
  return ''.join([colour] + list(args) + [Colours.RESET])


def printf(colour, *args, **kwargs):
  string = colourise(colour, *args)
  print(string, **kwargs)


def pprint(data, **kwargs):
<<<<<<< HEAD:labm8/py/io.py
<<<<<<< HEAD:labm8/py/io.py
  print(
<<<<<<< HEAD:labm8/py/io.py
    json.dumps(data, sort_keys=True, indent=2, separators=(",", ": ")), **kwargs
  )
=======
      json.dumps(data, sort_keys=True, indent=2, separators=(",", ": ")),
      **kwargs)
>>>>>>> 150d66672... Auto format files.:labm8/io.py
=======
  print(json.dumps(data, sort_keys=True, indent=2, separators=(",", ": ")),
=======
  print(json.dumps(data, sort_keys=True, indent=2, separators=(',', ': ')),
>>>>>>> 49340dc00... Auto-format labm8 python files.:labm8/io.py
        **kwargs)
>>>>>>> bb562b8d7... Refresh labm8 for new deps.:labm8/io.py


def info(*args, **kwargs):
  print('[INFO  ]', *args, **kwargs)


def debug(*args, **kwargs):
  print('[DEBUG ]', *args, **kwargs)


def warn(*args, **kwargs):
  print('[WARN  ]', *args, **kwargs)


def error(*args, **kwargs):
  print('[ERROR ]', *args, **kwargs)


def fatal(*args, **kwargs):
  returncode = kwargs.pop('status', 1)
  error('fatal:', *args, **kwargs)
  system.exit(returncode)


def prof(*args, **kwargs):
  """
  Print a profiling message.

  Profiling messages are intended for printing runtime performance
  data. They are prefixed by the "PROF" title.

  Arguments:

      *args, **kwargs: Message payload.
  """
  print('[PROF  ]', *args, **kwargs)


class Colours:
  """
  Shell escape colour codes.
  """

  RESET = "\033[0m"
  GREEN = "\033[92m"
  YELLOW = "\033[93m"
  BLUE = "\033[94m"
  RED = "\033[91m"
