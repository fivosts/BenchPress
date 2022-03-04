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
"""Samplers for CLgen language models.

A Sampler is an object which, when passed to a mode's Sample() method,
determines the shape of the generated samples.
"""
import os
import datetime
import typing
import pathlib
import pickle
from absl import flags
from sqlalchemy.ext import declarative

from deeplearning.clgen.util import cache
from deeplearning.clgen.util import pbutil
from deeplearning.clgen.util import crypto
from deeplearning.clgen.util import environment
from deeplearning.clgen.util import distrib
from deeplearning.clgen.util import commit
from deeplearning.clgen.features import extractor
from deeplearning.clgen.features import feature_sampler
from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.proto import sampler_pb2
from deeplearning.clgen.proto import internal_pb2
from deeplearning.clgen.models import lm_data_generator
from deeplearning.clgen.models import active_models

from deeplearning.clgen.util import logging as l
from deeplearning.clgen.util import sqlutil

FLAGS = flags.FLAGS

Base = declarative.declarative_base()

def AssertConfigIsValid(config: sampler_pb2.Sampler) -> sampler_pb2.Sampler:
  """Assert that a sampler configuration contains no invalid values.

  Args:
    config: A sampler configuration proto.

  Returns:
    The sampler configuration proto.

  Raises:
    UserError: If there are configuration errors.
  """
  try:
    if config.HasField("start_text"):
      pbutil.AssertFieldConstraint(
        config,
        "start_text",
        lambda s: len(s),
        "Sampler.start_text must be a string",
      )
    elif config.HasField("sample_corpus"):
      if config.sample_corpus.HasField("corpus_config"):
        if config.sample_corpus.corpus_config.HasField("normal"):
          pbutil.AssertFieldIsSet(config.sample_corpus.corpus_config, "normal")
        elif config.sample_corpus.corpus_config.HasField("online"):
          pbutil.AssertFieldIsSet(config.sample_corpus.corpus_config, "online")
        elif config.sample_corpus.corpus_config.HasField("active"):
          pbutil.AssertFieldIsSet(config.sample_corpus.corpus_config.active, "active_limit_per_feed")
          pbutil.AssertFieldIsSet(config.sample_corpus.corpus_config.active, "active_search_depth")
          pbutil.AssertFieldIsSet(config.sample_corpus.corpus_config.active, "active_search_width")
          pbutil.AssertFieldConstraint(
            config.sample_corpus.corpus_config.active,
            "batch_size_per_feed",
            lambda x : config.batch_size % x == 0,
            "batch_size {} must be a multiple of batch_size_per_feed".format(
              config.sample_corpus.corpus_config.active,
              config.batch_size
            )
          )
          pbutil.AssertFieldConstraint(
            config.sample_corpus.corpus_config.active,
            "feature_space",
            lambda x : x in set(extractor.extractors.keys()),
            "feature_space can only be one of {}".format(', '.join(list(extractor.extractors.keys())))
          )
          if config.sample_corpus.corpus_config.active.HasField("target"):
            pbutil.AssertFieldConstraint(
              config.sample_corpus.corpus_config.active,
              "target",
              lambda x : x in set(feature_sampler.targets.keys()),
              "target can only be one of {}".format(', '.join(list(feature_sampler.targets.keys())))
            )
          elif config.sample_corpus.corpus_config.active.HasField("active_learner"):
            active_models.AssertConfigIsValid(config.sample_corpus.corpus_config.active.active_learner)
          else:
            raise ValueError(config.sample_corpus.corpus_config.active)
        else:
          raise ValueError("Sampling type is undefined: {}".format(config.sample_corpus.corpus_config))

        pbutil.AssertFieldIsSet(config.sample_corpus.corpus_config, "max_predictions_per_seq")
        pbutil.AssertFieldIsSet(config.sample_corpus.corpus_config, "masked_lm_prob")

        pbutil.AssertFieldIsSet(config.sample_corpus.corpus_config, "mask_technique")
        if config.sample_corpus.corpus_config.HasField("mask"):
          pbutil.AssertFieldIsSet(
            config.sample_corpus.corpus_config.mask,
            "random_placed_mask",
          )
        elif config.sample_corpus.corpus_config.HasField("hole"):
          if config.sample_corpus.corpus_config.hole.HasField("absolute_length"):
            pbutil.AssertFieldConstraint(
              config.sample_corpus.corpus_config.hole,
              "absolute_length",
              lambda x : x > 0,
              "absolute length is the upper bound range of a hole's length. Therefore should be > 0."
            )
          else:
            pbutil.AssertFieldConstraint(
              config.sample_corpus.corpus_config.hole,
              "relative_length",
              lambda x : 0.0 < x <= 1.0,
              "relative length must be between 0 and 100% of a kernel's actual length."
            )
          if config.sample_corpus.corpus_config.hole.HasField("normal_distribution"):
            pbutil.AssertFieldIsSet(
              config.sample_corpus.corpus_config.hole.normal_distribution,
              "mean",
            )
            pbutil.AssertFieldIsSet(
              config.sample_corpus.corpus_config.hole.normal_distribution,
              "variance",
            )
          elif not config.sample_corpus.corpus_config.hole.HasField("uniform_distribution"):
            raise ValueError("Hole length distribution has not been set.")
        elif config.sample_corpus.corpus_config.HasField("mask_seq"):
          if config.sample_corpus.corpus_config.mask_seq.HasField("absolute_length"):
            pbutil.AssertFieldConstraint(
              config.sample_corpus.corpus_config.mask_seq,
              "absolute_length",
              lambda x : x > 0,
              "absolute length is the upper bound range of a mask_seq's length. Therefore should be > 0."
            )
          else:
            pbutil.AssertFieldConstraint(
              config.sample_corpus.corpus_config.mask_seq,
              "relative_length",
              lambda x : 0.0 < x <= 1.0,
              "relative length must be between 0 and 100% of a kernel's actual length."
            )
          if config.sample_corpus.corpus_config.mask_seq.HasField("normal_distribution"):
            pbutil.AssertFieldIsSet(
              config.sample_corpus.corpus_config.mask_seq.normal_distribution,
              "mean",
            )
            pbutil.AssertFieldIsSet(
              config.sample_corpus.corpus_config.mask_seq.normal_distribution,
              "variance",
            )
          elif not config.sample_corpus.corpus_config.mask_seq.HasField("uniform_distribution"):
            raise ValueError("Hole length distribution has not been set.")
      else:
        raise ValueError("sample_corpus has no corpus_config field.")

      if config.sample_corpus.HasField("corpus"):
        corpuses.AssertConfigIsValid(config.sample_corpus.corpus)        
      else:
        pbutil.AssertFieldIsSet(
          config.sample_corpus,
          "start_text"
        )
    elif ((not config.HasField("train_set"))
      and (not config.HasField("validation_set"))
      and (not config.HasField("sample_set"))
      and (not config.HasField("live_sampling"))):
      raise ValueError(config)
    pbutil.AssertFieldConstraint(
      config, "batch_size", lambda x: 0 < x, "Sampler.batch_size must be > 0"
    )
    pbutil.AssertFieldConstraint(
      config,
      "sequence_length",
      lambda x: 0 < x,
      "Sampler.sequence_length must be > 0",
    )
    pbutil.AssertFieldConstraint(
      config,
      "temperature_micros",
      lambda x: 0 < x,
      "Sampler.temperature_micros must be > 0",
    )
    return config
  except pbutil.ProtoValueError as e:
    raise ValueError(e)


class TerminationCriterionBase(object):
  """Base class for TerminationCriterion objects.

  A TerminationCriterion is an object with a single public function
  SampleIsComplete(), which accepts as its sole argument a sample-in-progress,
  and returns whether to stop sampling.
  """

  def Specialize(self, tokenizer: tokenizers.TokenizerBase) -> None:
    """Specialize a termination criteria to a vocabulary.

    This enables the termination criteria to set state specialized to a specific
    encoding vocabulary. This is guaranteed to be called before
    SampleIsComplete(), and ensures that the vocabulary used for all sample
    arguments to SampleIsComplete() is from this vocabulary.

    Args:
      tokenizer: An tokenizer to specialize to.
    """
    pass

  def SampleIsComplete(self, sample_in_progress: typing.List[str]) -> bool:
    """Determine whether to stop sampling.

    Args:
      sample_in_progress: A sample in progress, as a sequence of decoded tokens.

    Returns:
      True if the sample is "complete", else False to continue sampling.
    """
    raise NotImplementedError("abstract class")


class MaxlenTerminationCriterion(TerminationCriterionBase):
  """A termination criterion which limits the maximum length of a sample."""

  def __init__(self, config: sampler_pb2.MaxTokenLength):
    try:
      self.max_len = pbutil.AssertFieldConstraint(
        config,
        "maximum_tokens_in_sample",
        lambda x: x > 1,
        "MaxTokenLength.maximum_tokens_in_sample must be > 0",
      )
    except pbutil.ProtoValueError as e:
      raise ValueError(e)

  def SampleIsComplete(self, sample_in_progress: typing.List[str]) -> bool:
    """Determine whether to stop sampling."""
    return len(sample_in_progress) >= self.max_len


class SymmetricalTokenDepthCriterion(TerminationCriterionBase):
  """A termination criterion which counts symmetrical token depth.

  This is a generalization of bracked (i.e. { }) depth counting for C-syntax
  programming languages. When sampling to generate a C function, the sample
  is not "started" until the first { token is reached, and it is complete once
  the final } token has been emitted to close the function. In between those
  two tokens, there may be additional { } characters which increase and decrease
  the "depth" of the scope, respectively.
  """

  def __init__(self, config: sampler_pb2.SymmetricalTokenDepth):
    try:
      self.left_token = pbutil.AssertFieldConstraint(
        config,
        "depth_increase_token",
        lambda s: len(s) > 0,
        "SymmetricalTokenDepth.depth_increase_token must be a string",
      )
      self.right_token = pbutil.AssertFieldConstraint(
        config,
        "depth_decrease_token",
        lambda s: len(s) > 0,
        "SymmetricalTokenDepth.depth_decrease_token must be a string",
      )
    except pbutil.ProtoValueError as e:
      raise ValueError(e)
    if self.left_token == self.right_token:
      raise ValueError("SymmetricalTokenDepth tokens must be different")

  def Specialize(self, tokenizer: tokenizers.TokenizerBase) -> None:
    """Specialize a termination criteria to a vocabulary.

    This enables the termination criteria to set state specialized to a specific
    encoding vocabulary. This is guaranteed to be called before
    SampleIsComplete(), and ensures that the vocabulary used for all sample
    arguments to SampleIsComplete() is from this vocabulary.

    Args:
      tokenizer: An tokenizer to specialize to.

    Raises:
      InvalidSymtokTokens: If the depth tokens can't be encoded, or they encode
        to more than one token.
    """
    try:
      left = tokenizer.TokenizeString(self.left_token)
      right = tokenizer.TokenizeString(self.right_token)
      if len(left) > 1 or len(right) > 1:
        raise ValueError(
          "Sampler symmetrical depth tokens do not encode to a single "
          "token using the corpus vocabulary"
        )
    except ValueError:
      raise ValueError(
        "Sampler symmetrical depth tokens cannot be encoded using the "
        "corpus vocabulary"
      )

  def SampleIsComplete(self, sample_in_progress: typing.List[str]) -> bool:
    """Determine whether to stop sampling."""
    if len(sample_in_progress) == 0:
      return False
    if not sample_in_progress[-1] == self.right_token:
      return False
    return self.GetTokenDepth(sample_in_progress) == 0

  def GetTokenDepth(self, sample_in_progress: typing.List[str]) -> int:
    """Calculate the symmetrical token depth.

    The symmetrical token depth is the difference between the left and right
    token counts, provided that the last token is the right, left token count
    is nonzero, the right token count is less than the left token count. If
    either of those constraints are not met, the returned value is negative.
    """
    left_token_count = sample_in_progress.count(self.left_token)
    right_token_count = sample_in_progress.count(self.right_token)
    # We have descending into negative depth, so abort.
    if right_token_count and not left_token_count:
      return 0
    # We haven't started balancing the tokens yet.
    if not left_token_count:
      return -1
    return left_token_count - right_token_count


def GetTerminationCriteria(
  config: typing.List[sampler_pb2.SampleTerminationCriterion],
) -> typing.List[TerminationCriterionBase]:
  """Build a list of termination criteria from config protos.

  Args:
    config: A list of SampleTerminationCriterion protos.

  Returns:
    A list of TerminationCriterion instances.

  Raises:
    UserError: In case of invalid configs.
    InternalError: If any of the termination criteria are unrecognized.
  """
  terminators = []
  for criterion in config:
    if criterion.HasField("maxlen"):
      terminators.append(MaxlenTerminationCriterion(criterion.maxlen))
    elif criterion.HasField("symtok"):
      terminators.append(SymmetricalTokenDepthCriterion(criterion.symtok))
    else:
      raise SystemError("Unknown Sampler.termination_criteria")
  return terminators

class Sampler(object):
  """CLgen sampler for models.

  Please note sampler instances should be treated as immutable. Upon
  instantiation, a sampler's properties are used to determine its hash. If you
  modify a property after instantiation, the hash will be out of date, which
  can lead to bad things happening.
  """

  @property
  def is_active(self):
    if self.config.HasField("sample_corpus"):
      return self.config.sample_corpus.corpus_config.HasField("active")
    else:
      return False

  @property
  def has_active_learning(self):
    if not self.is_active:
      return False
    return self.config.sample_corpus.corpus_config.active.HasField("active_learner")

  @property
  def is_online(self):
    if self.config.HasField("sample_corpus"):
      return self.config.sample_corpus.corpus_config.HasField("online")
    else:
      return False

  @property
  def is_live(self):
    return self.config.HasField("live_sampling")

  @property
  def isFixedStr(self):
    if self.config.HasField("sample_corpus"):
      return self.config.sample_corpus.HasField("start_text")
    else:
      return self.config.HasField("start_text") and not (
            self.config.HasField("train_set") or
            self.config.HasField("validation_set") or
            self.config.HasField("sample_set") or
            self.config.HasField("sample_corpus")
          )

  def __init__(self, config: sampler_pb2.Sampler, sample_db_name = "samples.db"):
    """Instantiate a sampler.

    Args:
      config: A Sampler message.

    Raises:
      TypeError: If the config argument is not a Sampler proto.
      UserError: If the config contains invalid values.
    """
    if not isinstance(config, sampler_pb2.Sampler):
      t = type(config).__name__
      raise TypeError(f"Config must be a Sampler proto. Received: '{t}'")
    self.config = sampler_pb2.Sampler()
    self.config.CopyFrom(AssertConfigIsValid(config))
    self.hash = self._ComputeHash(self.config)
    self.terminators = GetTerminationCriteria(self.config.termination_criteria)
    if config.HasField("start_text"):
      self.start_text = self.config.start_text
    else:
      self.start_text = ""

    if self.has_active_learning:
      self.active_learner = active_models.Model(config.sample_corpus.corpus_config.active.active_learner)

    self.temperature = self.config.temperature_micros / 1e6
    self.batch_size = self.config.batch_size
    self.sequence_length = self.config.sequence_length
    self.sample_db_name = sample_db_name

    # Create the necessary cache directories.
    distrib.lock()
    self.cache = cache.mkcache("sampler", self.hash)
    distrib.unlock()
    self.samples_directory = self.cache.path / "samples"
    if environment.WORLD_RANK == 0:
      self.samples_directory.mkdir(exist_ok = True)
    self.corpus_directory = None
    self.sample_corpus    = None
    if self.config.HasField("sample_corpus"):
      self.corpus_directory = self.cache.path / "sample_corpus"
      if environment.WORLD_RANK == 0:
        self.corpus_directory.mkdir(exist_ok = True)
      if self.config.sample_corpus.HasField("corpus"):
        self.sample_corpus = corpuses.Corpus(self.config.sample_corpus.corpus)
        self.sample_corpus.Create()
        self.symlinkSampleCorpus(
          pathlib.Path(self.sample_corpus.encoded.url[len("sqlite:///") :]).parent
        )
        text_data = [
          self.sample_corpus.tokenizer.tokensToString(x) for x in self.sample_corpus.GetTrainingData()
        ]
      else:
        self.start_text = self.config.sample_corpus.start_text
        text_data = [self.start_text]
      # Text data is dumped in order to specialize with all different model tokenizers.
      if environment.WORLD_RANK == 0:
        with open(self.cache.path / "sample_corpus" / "text_corpus.pkl", 'wb') as outf:
          pickle.dump(text_data, outf)

    if environment.WORLD_RANK == 0:
      meta = internal_pb2.SamplerMeta()
      meta.config.CopyFrom(self.config)
      pbutil.ToFile(meta, path = self.cache.path / "META.pbtxt")
      commit.saveCommit(self.cache.path)

    # Set in Specialize().
    self.encoded_start_text = None
    self.tokenized_start_text = None

  def setStartText(self, start_text: str):
    """
      Assign current start_text used to sample. This function lazily assigns self.start_text and
      is used when sampling from tf_record dataset instead of a simple fixed string. This
      function is usedin conjunction with BERT Data generator.
    """
    self.start_text = start_text
    return

  def Create(self) -> None:
    if not self.has_active_learning:
      return None
    else:
      self.active_learner.Train()
      return

  def Specialize(self, tokenizer: tokenizers.TokenizerBase) -> None:
    """Specialize a sampler a vocabulary.

    This enables the sampler to set state specialized to a specific encoding
    vocabulary. This is guaranteed to be called before SampleIsComplete(), and
    ensures that the vocabulary used for all sample arguments to
    SampleIsComplete() is from this vocabulary.

    Args:
      tokenizer: An tokenizer to specialize to.

    Raises:
      InvalidStartText: If the start_text cannot be encoded using the
        vocabulary.
      UserError: In case the sampler cannot be specialized to this vocabulary.
    """
    try:
      self.encoded_start_text = tokenizer.TokenizeString(self.start_text)
      self.tokenized_start_text = tokenizer.AtomizeString(self.start_text)
    except ValueError:
      raise ValueError(
        "Sampler start text cannot be encoded using the corpus vocabulary: "
        f"'{self.start_text}'"
      )

    if len(self.encoded_start_text) > self.sequence_length:
      raise ValueError(
        "Encoded sampler start text must be less than sampler sequence "
        f"length. Sampler sequence length={self.sequence_length}, encoded "
        f"start text length={len(self.encoded_start_text)}"
      )
    l.logger().info("Sampling: '{}'\n".format(self.start_text))
    [terminator.Specialize(tokenizer) for terminator in self.terminators]

  def symlinkModelDB(self,
                     db_path   : pathlib.Path,
                     model_hash: int,
                     ) -> None:
    """
    Create symbolic link entry in sampler workspace. In one 
    model's workspace, there is one sampler.db for each different
    sampler. Each sampler holds a directory of all models it has 
    sampled with symbolic links created in this function.
    """
    assert os.path.isdir(db_path), "Parent path of database is not an existing path!"
    if environment.WORLD_RANK == 0:
      (self.samples_directory / model_hash).mkdir(exist_ok = True)

      for file in db_path.iterdir():
        symlink = self.samples_directory / model_hash / file.name
        if not symlink.is_symlink():
          os.symlink(
            os.path.relpath(
              db_path / file.name,
              self.samples_directory / model_hash
            ),
            symlink
          )
    return

  def symlinkSampleCorpus(self,
                          corpus_path : pathlib.Path,
                          ) -> None:
    """
    When sample corpus has been selected, creates a symlink
    of the sampled encoded corpus to the dataset 'sample_corpus'
    directory of the sampler.
    """
    assert os.path.isdir(corpus_path), "Parent path of database is not an existing path!"
    symlink = self.corpus_directory / "corpus"
    if not symlink.is_symlink():
      os.symlink(
        os.path.relpath(
          corpus_path,
          self.corpus_directory,
        ),
        symlink,
      )
    return

  def SampleIsComplete(self, sample_in_progress: typing.List[str]) -> bool:
    """Determine whether to stop sampling.

    Args:
      sample_in_progress: A sample in progress, as a sequence of decoded tokens.

    Returns:
      True if the sample is "complete", else False to continue sampling.
    """
    return any(t.SampleIsComplete(sample_in_progress) for t in self.terminators)

  @staticmethod
  def _ComputeHash(config: sampler_pb2.Sampler) -> str:
    """Compute sampler hash.

    The hash is computed from the serialized representation of the config
    proto.
    """
    return crypto.sha1(config.SerializeToString())

  def __eq__(self, rhs) -> bool:
    if not isinstance(rhs, Sampler):
      return False
    return rhs.hash == self.hash

  def __ne__(self, rhs) -> bool:
    return not self.__eq__(rhs)
