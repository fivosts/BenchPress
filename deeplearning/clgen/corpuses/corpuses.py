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
"""This file defines the logic and training corpuses.

A training corpus is a set of one or more "contentfiles", where each contentfile
is a file containing text to train over.
"""
import os
import pathlib
import random
import subprocess
import tempfile
import time
import typing
import json
import gdown
import humanize
import checksumdir
import numpy as np
from sqlalchemy.sql.expression import func

from deeplearning.clgen.util import cache
from deeplearning.clgen.util import crypto
from deeplearning.clgen.util import commit
from deeplearning.clgen.util import pbutil
from deeplearning.clgen.util import distrib
from deeplearning.clgen.util import environment
from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.corpuses import encoded
from deeplearning.clgen.corpuses import preprocessed
from deeplearning.clgen.preprocessors import preprocessors
from deeplearning.clgen.proto import corpus_pb2

from absl import flags

from deeplearning.clgen.util import sqlutil

from deeplearning.clgen.util import logging as l

FLAGS = flags.FLAGS

flags.DEFINE_string(
  "clgen_local_path_prefix",
  None,
  "An optional prefix to use when resolving the path to a local directory "
  "or archive. For example, given a corpus which is configured for a "
  'local_directory with value "foo/bar" and a --clgen_local_path_prefix of '
  '"/tmp/", the absolute path of the corpus will resolve to "/tmp/foo/bar". '
  "If the --clgen_local_path_prefix is a directory, the trailing slash must "
  "not be omitted.",
)

def AssertConfigIsValid(config: typing.Union[corpus_pb2.Corpus, corpus_pb2.PreTrainCorpus]
                       ) -> typing.Union[corpus_pb2.Corpus, corpus_pb2.PreTrainCorpus]:
  """Assert that config proto is valid.

  Args:
    config: A Corpus proto.

  Returns:
    The Corpus proto.

  Raises:
    UserError: If the config is invalid.
  """
  try:
    # Early-exit to support corpuses derived from databases of pre-encoded
    # content files.
    # TODO(github.com/ChrisCummins/clgen/issues/130): Refactor after splitting
    # Corpus class.
    if config.HasField("pre_encoded_corpus_url"):
      return config

    pbutil.AssertFieldIsSet(config,          "contentfiles")
    if isinstance(config, corpus_pb2.Corpus):
      pbutil.AssertFieldIsSet(config,          "tokenizer")
      pbutil.AssertFieldIsSet(config.tokenizer, "token_type")
      pbutil.AssertFieldConstraint(config.tokenizer, 
                                   "token_type", 
                                   lambda x: x == "character" or x == "word" or x == "ast",
                                   "tokenizer is either character or word based."
                                   )
      if config.tokenizer.token_type == "word":
        pbutil.AssertFieldConstraint(config.tokenizer,
                                    "token_list",
                                    lambda x: os.path.isfile(str(ExpandConfigPath(x, path_prefix=FLAGS.clgen_local_path_prefix))),
                                    "Invalid token_list file"
                                    )
    else:
      if config.HasField("tokenizer"):
        raise ValueError("Pre-train corpus cannot have a distinct tokenizer.")
    pbutil.AssertFieldIsSet(config,          "contentfile_separator")
    # Check that the preprocessor pipeline resolves to preprocessor functions.
    [preprocessors.GetPreprocessorFunction(p) for p in config.preprocessor]

    return config
  except pbutil.ProtoValueError as e:
    raise e


class Corpus(object):
  """Representation of a training corpus.

  Please note corpus instances should be treated as immutable. Upon
  instantiation, a corpus's properties are used to determine its hash. If you
  modify a property after instantiation, the hash will be out of date, which
  can lead to bad things happening.
  """

  def __init__(self, config: typing.Union[corpus_pb2.Corpus, corpus_pb2.PreTrainCorpus]):
    """Instantiate a corpus from a proto config.

    If this is a new corpus, a number of files will be created, which may
    take some time.

    Args:
      config: A Corpus message.

    Raises:
      TypeError: If the config argument is not a Sampler proto.
      UserError: In case the corpus is not found, or config contains invalid
        options.
      EmptyCorpusException: In case the corpus contains no data.
    """
    if not isinstance(config, corpus_pb2.Corpus) and not isinstance(config, corpus_pb2.PreTrainCorpus):
      raise TypeError(f"Config must be a Corpus proto. Received: '{type(config).__name__}'")

    # Make a local copy of the configuration.
    if isinstance(config, corpus_pb2.Corpus):
      self.config    = corpus_pb2.Corpus()
      self.pre_train = False
    else:
      self.config    = corpus_pb2.PreTrainCorpus()
      self.pre_train = True

    self.config.CopyFrom(AssertConfigIsValid(config))
    self._tokenizer = None
    self._created = False

    # An in-memory cache of the encoded contentfiles indices arrays.
    # Set and used in GetTrainingData().
    self._indices_arrays: typing.Optional[typing.List[np.array]] = None

    if environment.WORLD_RANK == 0:
      cache.cachepath("corpus").mkdir(parents=True, exist_ok=True)
    distrib.barrier()
    self.content_id = ResolveContentId(self.config)
    # Database of pre-processed files.
    preprocessed_id = ResolvePreprocessedId(self.content_id, self.config)
    if environment.WORLD_RANK == 0
      cache.cachepath("corpus", "preprocessed", preprocessed_id).mkdir(exist_ok=True, parents=True)
    distrib.barrier()
    preprocessed_db_path = cache.cachepath("corpus", "preprocessed",
                                           preprocessed_id, "preprocessed.db")

    if self.config.HasField("content_id") and not preprocessed_db_path.is_file():
      raise ValueError(f"Content ID not found: '{self.content_id}'")
    self.preprocessed = preprocessed.PreprocessedContentFiles(
      f"sqlite:///{preprocessed_db_path}"
    )
    # Create symlink to contentfiles.
    if environment.WORLD_RANK == 0:
      symlink = (pathlib.Path(self.preprocessed.url[len("sqlite:///") :]).parent / "contentfiles")
      if not symlink.is_symlink():
        if config.HasField("local_directory"):
          os.symlink(
            str(ExpandConfigPath(config.local_directory,   path_prefix=FLAGS.clgen_local_path_prefix)),
            symlink,
          )
        elif config.HasField("local_tar_archive"):
          os.symlink(
            str(ExpandConfigPath(config.local_tar_archive, path_prefix=FLAGS.clgen_local_path_prefix)),
            symlink,
          )
        elif config.HasField("bq_database"):
          os.symlink(
            str(ExpandConfigPath(config.bq_database, path_prefix=FLAGS.clgen_local_path_prefix)),
            symlink,
          )  
        # elif config.HasField("fetch_github"):
        #   os.symlink(
        #     str(ExpandConfigPath(config.fetch_github, path_prefix=FLAGS.clgen_local_path_prefix)),
        #     symlink,
        #   )
    distrib.barrier()
    # Data of encoded pre-preprocessed files.
    encoded_id = ResolveEncodedId(self.content_id, self.config)
    if environment.WORLD_RANK == 0:
      cache.cachepath("corpus", "encoded", encoded_id).mkdir(exist_ok=True, parents=True)
    distrib.barrier()
    db_path = cache.cachepath("corpus", "encoded", encoded_id, "encoded.db")
    if self.config.HasField("pre_encoded_corpus_url"):
      self.encoded = encoded.EncodedContentFiles(config.pre_encoded_corpus_url, self.pre_train)
    else:
      self.encoded = encoded.EncodedContentFiles(f"sqlite:///{db_path}", self.pre_train)
    self.tokenizer_path = cache.cachepath(
      "corpus", "encoded", encoded_id, "tokenizer.pkl"
    )
    if not self.config.HasField("pre_encoded_corpus_url"):
      symlink = (pathlib.Path(self.encoded.url[len("sqlite:///") :]).parent / "preprocessed")
      if not symlink.is_symlink():
        os.symlink(
          os.path.relpath(
            pathlib.Path(self.preprocessed.url[len("sqlite:///") :]).parent,
            pathlib.Path(self.encoded.url[len("sqlite:///") :]).parent,
            ),
          symlink,
        )
    self.hash = encoded_id
    self.cache = cache.mkcache("corpus", "encoded", encoded_id)
    if environment.WORLD_RANK == 0:
      commit.saveCommit(self.cache.path)
      commit.saveCommit(self.cache.path.parent.parent / "preprocessed" / preprocessed_id)
    distrib.barrier()
    l.logger().info("Initialized {}train corpus in {}".format("pre_" if self.pre_train else "", self.cache.path))
    return

  def GetShortSummary(self) -> str:
    try:
      corpus_size = humanize.naturalsize(self.encoded.token_count)
      return (
        f"{corpus_size} token corpus with {self.vocab_size}-element vocabulary"
      )
    except Exception:
      return ""


  def Create(self, tokenizer = None) -> None:
    """Create the corpus files.
  
    Args:
      tokenizer: In case of pre-training ONLY the tokenizer of fine-tuned corpus,
      is provided as-is in the pre-training corpus as they must have the same vocab.

    Raises:
      EmptyCorpusException: If there are no content files, or no successfully
        pre-processed files.
    """
    if self.pre_train and not tokenizer:
      raise ValueError("Tokenizer must be specified when encoding pre-training corpus.")
    self._created = True

    self.preprocessed.Create(self.config)
    if not self.preprocessed.size and not FLAGS.override_preprocessing:
      raise ValueError(
        f"Pre-processed corpus contains no files: '{self.preprocessed.url}'"
      )
    l.logger().info("Pre-processed {}train corpus in corpuses/{}".format("pre_" if self.pre_train else "", pathlib.Path(self.preprocessed.url).parent.stem))

    start_time      = time.time()
    self._tokenizer = tokenizer
    tokenizer       = self.tokenizer
    if not self.pre_train:
      l.logger().info(
        "{}: {} tokens".format(
            type(tokenizer).__name__,
            humanize.intcomma(tokenizer.vocab_size),
          )
      )
    self.encoded.Create(
      self.preprocessed, tokenizer, self.config.contentfile_separator
    )
    l.logger().info("Encoded {}train corpus in corpuses/{}".format("pre_" if self.pre_train else "", pathlib.Path(self.encoded.url).parent.stem))
    return

  def GetTextCorpus(self, shuffle: bool) -> str:
    """Concatenate the entire corpus into a string.

    Args:
      shuffle: If true, randomize order of contentfiles.

    Returns:
      A concatenated corpus string.
    """
    with self.preprocessed.Session() as session:
      query = session.query(preprocessed.PreprocessedContentFile.text).filter(
        preprocessed.PreprocessedContentFile.preprocessing_succeeded == True
      )
      if shuffle:
        query = query.order_by(func.random())
      return self.config.contentfile_separator.join([x[0] for x in query])

  def GetTrainingDataGenerator(self, offset: int = None):
    with self.encoded.Session() as session:
      if offset is None:
        for x in session.query(encoded.EncodedContentFile).yield_per(1000000):
          yield list(x.indices_array)
      else:
        for x in session.query(encoded.EncodedContentFile).offset(offset).yield_per(1000000):
          yield list(x.indices_array)
    return

  def GetTrainingData(self,
                      shuffle: bool = False,
                      sequence_length: int = False,
                      ) -> np.ndarray:
    """Concatenate the entire encoded corpus into an array.

    Args:
      shuffle: If true, randomize order of encoded contentfiles.
      sequence_length: If set, query is optimized to bring only fitting sequences.

    Returns:
      The encoded corpus.
    """
    # Load all indices from the database into memory, and keep them there.
    # This is to remove the latency from reading the contents from a
    # database.
    #
    # TODO(https://github.com/ChrisCummins/clgen/issues/128): Storing the
    # entire corpus in memory like this prevents training on corpuses larger
    # than system memory. Replace this method with an interface for streaming
    # data from the encoded database.
    if self._indices_arrays is None:
      with self.encoded.Session() as session:
        if sequence_length:
          query = session.query(encoded.EncodedContentFile).filter(encoded.EncodedContentFile.tokencount <= sequence_length).yield_per(1000000)
        else:
          query = session.query(encoded.EncodedContentFile).yield_per(1000000)
        self._indices_arrays = np.array([x.indices_array for x in query])

    if shuffle:
      random.shuffle(self._indices_arrays)

    return self._indices_arrays

  def GetTrainingFeatures(self, sequence_length: int) -> typing.List[typing.Dict[str, typing.Dict[str, float]]]:
    """
    Get feature vectors of training instances within the specified sequence length.
    """
    with self.encoded.Session() as session:
      query = session.query(encoded.EncodedContentFile).filter(encoded.EncodedContentFile.tokencount <= sequence_length)
      return [x.features for x in query]

  def getFeaturesContents(self, sequence_length: int = None) -> typing.List[typing.Tuple[np.array, typing.Dict[str, float]]]:
    """
    Get tuple of contents accompanied by feature vectors.
    """
    with self.encoded.Session() as session:
      if sequence_length:
        query = session.query(encoded.EncodedContentFile).filter(encoded.EncodedContentFile.tokencount <= sequence_length)
      else:
        query = session.query(encoded.EncodedContentFile)
      return [(self.tokenizer.ArrayToCode(x.indices_array, with_formatting = False), x.features) for x in query]

  def GetNumContentFiles(self) -> int:
    """Get the number of contentfiles which were pre-processed."""
    with self.preprocessed.Session() as session:
      return session.query(preprocessed.PreprocessedContentFile).count()

  def GetNumPreprocessedFiles(self) -> int:
    """The number of succesfully pre-processed content files."""
    with self.preprocessed.Session() as session:
      return (
        session.query(preprocessed.PreprocessedContentFile.text)
        .filter(
          preprocessed.PreprocessedContentFile.preprocessing_succeeded == True
        )
        .count()
      )

  @property
  def tokenizer(self) -> tokenizers.TokenizerBase:
    """Must call Create() first."""
    if not self._created:
      raise ValueError("Must call Create() before accessing tokenizer property.")
    if self._tokenizer is None:
      if self.tokenizer_path.is_file():
        self._tokenizer = tokenizers.TokenizerBase.FromFile(self.tokenizer_path)
      else:
        if environment.WORLD_RANK == 0:
          self._tokenizer = self._CreateTokenizer()
        distrib.barrier()
        if environment.WORLD_RANK != 0:
          self._tokenizer = tokenizers.TokenizerBase.FromFile(self.tokenizer_path)
        distrib.barrier()
    return self._tokenizer

  def _CreateTokenizer(self) -> tokenizers.TokenizerBase:
    """Creates and caches an tokenizer."""
    corpus_txt = self.GetTextCorpus(shuffle=False)

    if self.config.HasField("pre_encoded_corpus_url"):
      encoded_db = encoded.EncodedContentFiles(
        self.config.pre_encoded_corpus_url, self.pre_train
      )
      tokenizer = WordTokenizerFromEncodedDb(self.config.tokenizer, encoded_db)
    else:
      tokenizer = tokenizers.FromText(self.config.tokenizer, self.config.contentfile_separator, corpus_txt)

    tokenizer.ToFile(self.tokenizer_path)
    return tokenizer

  @property
  def vocab_size(self) -> int:
    """Get the number of elements in the corpus vocabulary."""
    return self.tokenizer.vocab_size

  @property
  def size(self) -> int:
    """Return the size of the atomized corpus."""
    with self.encoded.Session() as session:
      return session.query(
        sql.func.sum(encoded.EncodedContentFile.tokencount)
      ).one()

  def __eq__(self, rhs) -> bool:
    if not isinstance(rhs, Corpus):
      return False
    return rhs.hash == self.hash

  def __ne__(self, rhs) -> bool:
    return not self.__eq__(rhs)


def GetVocabFromMetaTable(session: sqlutil.Session) -> typing.Dict[str, int]:
  """Read a vocabulary dictionary from the 'Meta' table of a database."""
  return encoded.EncodedContentFiles.GetVocabFromMetaTable(session)


def StoreVocabInMetaTable(
  session: sqlutil.Session, vocabulary: typing.Dict[str, int]
) -> None:
  """Store a vocabulary dictionary in the 'Meta' table of a database."""
  return encoded.EncodedContentFiles.StoreVocabInMetaTable(session, vocabulary)


def WordTokenizerFromEncodedDb(encoded_db: encoded.EncodedContentFiles):
  raise NotImplementedError
  """Create a greedy tokenizer for the vocabulary of a given encoded_db."""
  # TODO(github.com/ChrisCummins/clgen/issues/130): This should be a method of
  # a concrete `DatabaseCorpus` class.
  with encoded_db.Session() as s:
    vocab = GetVocabFromMetaTable(s)
  l.logger().info("Loaded vocabulary of {} tokens from meta table".format(len(vocab)))
  return tokenizers.WordTokenizer(vocab)


def ExpandConfigPath(path: str, path_prefix: str = None) -> pathlib.Path:
  """Resolve an absolute path from a config proto string field.

  This performs shell-style expansion of $VARS, and prefixes the
  --clgen_local_path_prefix flag value, if it is set.

  Args:
    path: The string value as it appears in the proto.
    path_prefix: An optional string to prepend to the resolved path.

  Returns:
    An absolute path.
  """
  # Set a useful variable for expansion.
  if "HOME" not in os.environ:
    os.environ["HOME"] = str(pathlib.Path("~").expanduser())
  return (
    pathlib.Path(os.path.expandvars((path_prefix or "") + path))
    .expanduser()
    .absolute()
  )


def ResolveContentId(config: typing.Union[corpus_pb2.Corpus, corpus_pb2.PreTrainCorpus]) -> str:
  """Compute the hash of the input contentfiles.

  This function resolves the unique sha1 checksum of a set of content files.

  Args:
    config: The corpus config proto.

  Returns:
    A hex encoded sha1 string.
  """
  # We can take a massive shortcut if the content ID is already set in the
  # config proto.
  if config.HasField("content_id"):
    # TODO(github.com/ChrisCummins/clgen/issues/130): Refactor this after splitting
    # out Corpus class.
    return config.content_id
  elif config.HasField("pre_encoded_corpus_url"):
    # TODO(github.com/ChrisCummins/clgen/issues/130): Refactor this after splitting
    # out Corpus class.
    return crypto.sha1_str(config.pre_encoded_corpus_url)

  start_time = time.time()
  if config.HasField("local_directory"):
    local_directory = ExpandConfigPath(
      config.local_directory, path_prefix=FLAGS.clgen_local_path_prefix
    )

    # After the first time we compute the hash of a directory, we write it into
    # a file. This is a shortcut to work around the fact that computing the
    # directory checksum is O(n) with respect to the number of files in the
    # directory (even if the directory is already cached by the hash cache).
    # This means that it is the responsibility of the user to delete this cached
    # file if the directory is changed.
    hash_file_path = pathlib.Path(str(local_directory) + ".sha1.txt")
    if hash_file_path.is_file():
      l.logger().info("Reading directory hash: '{}'.".format(hash_file_path))
      with open(hash_file_path) as f:
        content_id = f.read().rstrip()
    else:
      # No hash file, so compute the directory hash and create it.
      try:
        # content_id = hc.GetHash(local_directory)
        content_id = crypto.sha256_str(str(local_directory))
      except FileNotFoundError as e:
        raise ValueError(e)
      # Create the hash file in the directory so that next time we don't need
      # to reference the hash cache.
      with open(hash_file_path, "w") as f:
        print(content_id, file=f)
      l.logger().info("Wrote directory hash: '{}'.".format(hash_file_path))
  elif config.HasField("local_tar_archive"):
    # This if not an efficient means of getting the hash, as it requires always
    # unpacking the archive and reading the entire contents. It would be nicer
    # to maintain a cache which maps the mtime of tarballs to their content ID,
    # similart to how local_directory is implemented.
    content_id = GetHashOfArchiveContents(
      ExpandConfigPath(config.local_tar_archive, path_prefix=FLAGS.clgen_local_path_prefix)
    )
  elif config.HasField("bq_database"):
    content_id = crypto.sha256_str(str(config.bq_database))
  # elif config.HasField("fetch_github"):

  #   gitfile_path = ExpandConfigPath(
  #     config.fetch_github, path_prefix=FLAGS.clgen_local_path_prefix
  #   )
  #   gitfile_path.mkdir(exist_ok=True, parents=True)
  #   github_fetcher = github.GithubFetcher(gitfile_path)

  #   github_fetcher.fetch()
  #   hash_file_path = pathlib.Path(str(gitfile_path) + ".sha1.txt")
  #   if hash_file_path.is_file():
  #     l.logger().info("Reading directory hash: '{}'.".format(hash_file_path))
  #     with open(hash_file_path) as f:
  #       content_id = f.read().rstrip()
  #   else:
  #     # No hash file, so compute the directory hash and create it.
  #     try:
  #       content_id = hc.GetHash(gitfile_path)
  #     except FileNotFoundError as e:
  #       raise ValueError(e)
  #     # Create the hash file in the directory so that next time we don't need
  #     # to reference the hash cache.
  #     with open(hash_file_path, "w") as f:
  #       print(content_id, file=f)
  #     l.logger().info("Wrote directory hash: '{}'.".format(hash_file_path))
  else:
    raise NotImplementedError("Unsupported Corpus.contentfiles field value")
  return content_id


def ResolvePreprocessedId(content_id: str,
                          config: typing.Union[corpus_pb2.Corpus, corpus_pb2.PreTrainCorpus]
                          ) -> str:
  """Compute the hash of a corpus of preprocessed contentfiles.

  The hash is computed from the ID of the input files and the serialized
  representation of the preprocessor pipeline.
  """
  # TODO(github.com/ChrisCummins/clgen/issues/130): Refactor this after splitting
  # out Corpus class.
  if config.pre_encoded_corpus_url:
    return "null"
  return crypto.sha1_list(content_id, *config.preprocessor)


def ResolveEncodedId(content_id: str,
                     config: typing.Union[corpus_pb2.Corpus, corpus_pb2.PreTrainCorpus]
                     ) -> str:
  """Compute the hash of a corpus of preprocessed and encoded contentfiles.

  The hash is computed from the ID of the input files and the serialized
  representation of the config proto.
  """
  if isinstance(config,  corpus_pb2.Corpus):
    config_without_contentfiles = corpus_pb2.Corpus()
  else:
    config_without_contentfiles = corpus_pb2.PreTrainCorpus()
  config_without_contentfiles.CopyFrom(config)
  # Clear the contentfiles field, since we use the content_id to uniquely
  # identify the input files. This means that corpuses with the same content
  # files delivered through different means (e.g. two separate but identical
  # directories) have the same hash.
  config_without_contentfiles.ClearField("contentfiles")
  return crypto.sha1_list(
    content_id, config_without_contentfiles.SerializeToString()
  )


def GetHashOfArchiveContents(archive: pathlib.Path) -> str:
  """Compute the checksum of the contents of a directory.

  Args:
    archive: Path of the archive.

  Returns:
    Checksum of the archive.

  Raises:
    UserError: If the requested archive does not exist, or cannot be unpacked.
  """
  if not (archive.parent / "corpus_registry.json").exists():
    raise FileNotFoundError("corpus_registry.json file not found.")

  with open(archive.parent / "corpus_registry.json", 'r') as js:
    reg = json.load(js)

  if archive.name not in reg:
    raise FileNotFoundError("Corpus {} is not registered in corpus_registry".format(archive.name))

  if not archive.is_file():
    l.logger().info("Corpus found in registry. Downloading from Google Drive...")
    if environment.WORLD_RANK == 0:
      gdown.download("https://drive.google.com/uc?id={}".format(reg[archive.name]['url']), str(archive))
    distrib.barrier()

  if 'hash' in reg[archive.name]:
    return reg[archive.name]['hash']
  else:
    with tempfile.TemporaryDirectory(prefix="clgen_corpus_", dir = FLAGS.local_filesystem) as d:
      pv  = ["pv", str(archive)]
      tar = ["tar", "xfj", "-", "-C", d]
      try:
        pv_proc = subprocess.Popen(pv, stdout = subprocess.PIPE)
        subprocess.check_call(tar, stdin = pv_proc.stdout)
      except subprocess.CalledProcessError:
        raise ValueError(f"Archive unpack failed: '{archive}'")
      return checksumdir.dirhash(d, "sha1")
