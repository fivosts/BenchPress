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
"""Pytest fixtures for CLgen unit tests."""
import os
import pathlib
import tarfile
import tempfile

<<<<<<< HEAD:deeplearning/clgen/tests/fixtures.py
=======
import pytest

>>>>>>> 89b790ba9... Merge absl logging, app, and flags modules.:deeplearning/clgen/conftest.py
from deeplearning.clgen.proto import clgen_pb2
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import sampler_pb2
<<<<<<< HEAD:deeplearning/clgen/tests/fixtures.py
<<<<<<< HEAD:deeplearning/clgen/tests/fixtures.py
<<<<<<< HEAD:deeplearning/clgen/tests/fixtures.py
from labm8.py import pbutil
from labm8.py import test

FLAGS = test.FLAGS
=======
from labm8 import pbutil

FLAGS = flags.FLAGS
>>>>>>> 3333e1db6... Auto format files.:deeplearning/clgen/conftest.py
=======
from labm8 import app
from labm8 import pbutil
=======
from labm8.py import app
from labm8.py import pbutil
>>>>>>> 8be094257... Move //labm8 to //labm8/py.:deeplearning/clgen/conftest.py

FLAGS = app.FLAGS
>>>>>>> 89b790ba9... Merge absl logging, app, and flags modules.:deeplearning/clgen/conftest.py


@test.Fixture(scope="function")
def clgen_cache_dir() -> str:
  """Creates a temporary directory and sets CLGEN_CACHE to it.

  This fixture has session scope, meaning that the clgen cache directory
  is shared by all unit tests.

  Returns:
    The location of $CLGEN_CACHE.
  """
  with tempfile.TemporaryDirectory(prefix="clgen_cache_") as d:
    os.environ["CLGEN_CACHE"] = d
    yield d


@test.Fixture(scope="function")
def abc_corpus() -> str:
  """A corpus consisting of three files.

  This fixture has function scope, meaning that a new corpus is created for
  every function which uses this fixture.

  Returns:
    The location of the corpus directory.
  """
  with tempfile.TemporaryDirectory(prefix="clgen_abc_corpus_") as d:
    path = pathlib.Path(d)
    with open(path / "a", "w") as f:
      f.write("The cat sat on the mat.")
    with open(path / "b", "w") as f:
      f.write("Hello, world!")
    with open(path / "c", "w") as f:
      f.write("\nSuch corpus.\nVery wow.")
    yield d


@test.Fixture(scope="function")
def abc_corpus_archive(abc_corpus) -> str:
  """Creates a .tar.bz2 packed version of the abc_corpus.

  Returns:
    Path to the abc_corpus tarball.
  """
  with tempfile.TemporaryDirectory() as d:
    with tarfile.open(d + "/corpus.tar.bz2", "w:bz2") as f:
      f.add(abc_corpus + "/a", arcname="corpus/a")
      f.add(abc_corpus + "/b", arcname="corpus/b")
      f.add(abc_corpus + "/c", arcname="corpus/c")
    yield d + "/corpus.tar.bz2"


@test.Fixture(scope="function")
def abc_corpus_config(abc_corpus):
  """The proto config for a simple Corpus."""
<<<<<<< HEAD:deeplearning/clgen/tests/fixtures.py
  return corpus_pb2.Corpus(
<<<<<<< HEAD:deeplearning/clgen/tests/fixtures.py
    local_directory=abc_corpus,
    ascii_character_atomizer=True,
    contentfile_separator="\n\n",
  )
<<<<<<< HEAD:deeplearning/clgen/tests/fixtures.py
=======
      local_directory=abc_corpus,
      ascii_character_atomizer=True,
      contentfile_separator='\n\n')
>>>>>>> 3333e1db6... Auto format files.:deeplearning/clgen/conftest.py
=======
  return corpus_pb2.Corpus(local_directory=abc_corpus,
                           ascii_character_atomizer=True,
                           contentfile_separator='\n\n')
>>>>>>> 1ae6d8129... Update //deeplearning/clgen/...:deeplearning/clgen/conftest.py
=======
>>>>>>> 8434bf4d8... Add //labm8/py:test wrappers for pytest functions.:deeplearning/clgen/conftest.py


@test.Fixture(scope="function")
def abc_model_config(abc_corpus_config):
  """The proto config for a simple Model."""
  architecture = model_pb2.NetworkArchitecture(
<<<<<<< HEAD:deeplearning/clgen/tests/fixtures.py
    backend=model_pb2.NetworkArchitecture.TENSORFLOW,
    embedding_size=2,
    neuron_type=model_pb2.NetworkArchitecture.LSTM,
    neurons_per_layer=4,
    num_layers=1,
    post_layer_dropout_micros=2000,
  )
  optimizer = model_pb2.AdamOptimizer(
    initial_learning_rate_micros=2000,
    learning_rate_decay_per_epoch_micros=5000,
    beta_1_micros=900000,
    beta_2_micros=999000,
    normalized_gradient_clip_micros=5000000,
  )
=======
      backend=model_pb2.NetworkArchitecture.TENSORFLOW,
      embedding_size=2,
      neuron_type=model_pb2.NetworkArchitecture.LSTM,
      neurons_per_layer=4,
      num_layers=1,
      post_layer_dropout_micros=2000)
  optimizer = model_pb2.AdamOptimizer(initial_learning_rate_micros=2000,
                                      learning_rate_decay_per_epoch_micros=5000,
                                      beta_1_micros=900000,
                                      beta_2_micros=999000,
                                      normalized_gradient_clip_micros=5000000)
>>>>>>> 1ae6d8129... Update //deeplearning/clgen/...:deeplearning/clgen/conftest.py
  training = model_pb2.TrainingOptions(
<<<<<<< HEAD:deeplearning/clgen/tests/fixtures.py
    num_epochs=1,
    sequence_length=10,
    batch_size=5,
    shuffle_corpus_contentfiles_between_epochs=False,
    adam_optimizer=optimizer,
  )
  return model_pb2.Model(
    corpus=abc_corpus_config, architecture=architecture, training=training
  )


@test.Fixture(scope="function")
<<<<<<< HEAD:deeplearning/clgen/tests/fixtures.py
=======
      num_epochs=1,
      sequence_length=10,
      batch_size=5,
      shuffle_corpus_contentfiles_between_epochs=False,
      adam_optimizer=optimizer)
  return model_pb2.Model(corpus=abc_corpus_config,
                         architecture=architecture,
                         training=training)


@pytest.fixture(scope='function')
>>>>>>> 3333e1db6... Auto format files.:deeplearning/clgen/conftest.py
=======
>>>>>>> 8434bf4d8... Add //labm8/py:test wrappers for pytest functions.:deeplearning/clgen/conftest.py
def abc_sampler_config():
  """The sampler config for a simple Sampler."""
  maxlen = sampler_pb2.MaxTokenLength(maximum_tokens_in_sample=5)
  sample_stop = [sampler_pb2.SampleTerminationCriterion(maxlen=maxlen)]
<<<<<<< HEAD:deeplearning/clgen/tests/fixtures.py
  return sampler_pb2.Sampler(
<<<<<<< HEAD:deeplearning/clgen/tests/fixtures.py
    start_text="a",
    batch_size=5,
    sequence_length=10,
    termination_criteria=sample_stop,
    temperature_micros=1000000,
  )


@test.Fixture(scope="function")
def abc_instance_config(
  clgen_cache_dir, abc_model_config, abc_sampler_config
) -> clgen_pb2.Instance:
  """A test fixture that returns an Instance config proto."""
  return clgen_pb2.Instance(
    working_dir=clgen_cache_dir,
    model=abc_model_config,
    sampler=abc_sampler_config,
  )
<<<<<<< HEAD:deeplearning/clgen/tests/fixtures.py
=======
      start_text='a',
      batch_size=5,
      sequence_length=10,
      termination_criteria=sample_stop,
      temperature_micros=1000000)
=======
  return sampler_pb2.Sampler(start_text='a',
                             batch_size=5,
                             sequence_length=10,
                             termination_criteria=sample_stop,
                             temperature_micros=1000000)
>>>>>>> 1ae6d8129... Update //deeplearning/clgen/...:deeplearning/clgen/conftest.py


@pytest.fixture(scope='function')
def abc_instance_config(clgen_cache_dir, abc_model_config,
                        abc_sampler_config) -> clgen_pb2.Instance:
  """A test fixture that returns an Instance config proto."""
<<<<<<< HEAD:deeplearning/clgen/tests/fixtures.py
  return clgen_pb2.Instance(
      working_dir=clgen_cache_dir,
      model=abc_model_config,
      sampler=abc_sampler_config)
>>>>>>> 3333e1db6... Auto format files.:deeplearning/clgen/conftest.py
=======
  return clgen_pb2.Instance(working_dir=clgen_cache_dir,
                            model=abc_model_config,
                            sampler=abc_sampler_config)
>>>>>>> 1ae6d8129... Update //deeplearning/clgen/...:deeplearning/clgen/conftest.py
=======
>>>>>>> 8434bf4d8... Add //labm8/py:test wrappers for pytest functions.:deeplearning/clgen/conftest.py


@test.Fixture(scope="function")
def abc_instance_file(abc_instance_config) -> str:
  """A test fixture that returns a path to an Instance config file."""
  with tempfile.NamedTemporaryFile() as f:
    pbutil.ToFile(abc_instance_config, pathlib.Path(f.name))
    yield f.name
