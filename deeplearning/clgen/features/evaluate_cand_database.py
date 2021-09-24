"""A module for databases of CLgen samples."""
import contextlib
import math
import datetime
import typing
import sqlite3
import numpy as np
import sqlalchemy as sql
from sqlalchemy.ext import declarative
from absl import flags

from deeplearning.clgen.util import crypto
from deeplearning.clgen.util import sqlutil

FLAGS = flags.FLAGS

Base = declarative.declarative_base()

class SearchCandidate(Base, sqlutil.ProtoBackedMixin):
  """A database row representing a CLgen sample.

  This is the clgen.Sample protocol buffer in SQL format.
  """
  __tablename__    = "search_candidates"
  # entry id
  id               : int    = sql.Column(sql.Integer,    primary_key = True)
  # unique hash of sample text
  sha256           : str    = sql.Column(sql.String(64), nullable = False, index = True)
  # sample_hash
  sample_sha256     : str   = sql.Column(sql.String(64), nullable = False)
  # generation id of sample.
  generation_id     : int   = sql.Column(sql.Integer, nullable = False)
  # Frequency of specific sample.
  frequency         : int   = sql.Column(sql.Integer, nullable = False)
  # hole length in terms of actual tokens hidden
  abs_hole_lengths  : str   = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # hole length in terms of percentage of kernel's actual length.
  rel_hole_lengths  : str   = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # how many tokens were used to fill that hole
  hole_ind_length   : int   = sql.Column(sql.Integer, nullable = False)
  # Original input feed pre-masking
  input_feed        : str   = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Input Ids with the hole placed.
  input_ids         : str   = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Encoded original input
  encoded_input_ids : str   = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Feature vector of input_feed
  input_features    : str   = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Score-distance of input from target benchmark
  input_score       : float = sql.Column(sql.Float, nullable = False)
  # Output sample
  sample            : str   = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # indices filled in the hole.
  sample_indices    : str   = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Actual length of sample, excluding pads.
  num_tokens        : int   = sql.Column(sql.Integer, nullable = False)
  # Sample's vector of features.
  output_features   : str   = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Sample distance from target benchmark.
  sample_score      : float = sql.Column(sql.Float,  nullable = False)
  # Name and contents of target benchmark specified.
  target_benchmark  : str   = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Feature vector of target benchmark.
  target_features   : str   = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Whether sample compiles or not.
  compile_status    : bool  = sql.Column(sql.Boolean, nullable = False)
  # Percentage delta of output score compared to input score.
  score_delta       : float = sql.Column(sql.Float, nullable = False)
  # Delta between feature of sample - input.
  features_delta    : str   = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Date
  date_added        : datetime.datetime = sql.Column(sql.DateTime, nullable = False)

  @classmethod
  def FromArgs(cls,
               tokenizer,
               id               : int,
               input_feed       : np.array,
               input_ids        : np.array,
               input_features   : typing.Dict[str, float],
               input_score      : float,
               hole_lengths     : typing.List[int],
               sample           : np.array,
               sample_indices   : np.array,
               output_features  : typing.Dict[str, float],
               sample_score     : float,
               target_benchmark : typing.Tuple[str, str],
               target_features  : typing.Dict[str, float],
               compile_status   : bool,
               generation_id    : int,
               # timestep         : int,
               ) -> typing.TypeVar("SearchCandidate"):
    """Construt SearchCandidate table entry from argumentns."""
    str_input_feed = tokenizer.tokensToString(input_ids, ignore_token = tokenizer.padToken, with_formatting = True)
    str_sample     = tokenizer.ArrayToCode(sample, with_formatting = True)
    len_indices    = len(sample_indices)
    sample_indices = tokenizer.tokensToString(sample_indices, ignore_token = tokenizer.padToken)

    num_tokens = len(sample)
    if tokenizer.padToken in sample:
      num_tokens = np.where(sample == tokenizer.padToken)[0][0]

    actual_length = len(input_ids) - 3
    if tokenizer.padToken in input_ids:
      actual_length = np.where(input_ids == tokenizer.padToken)[0][0] - 3

    return SearchCandidate(
      id                = id,
      sha256            = crypto.sha256_str(str_input_feed + str_sample + str(hole_lengths)),
      sample_sha256     = crypto.sha256_str(str_sample),
      generation_id     = generation_id,
      frequency         = 1,
      abs_hole_lengths  = ','.join([str(hl) for hl in hole_lengths if hl >= 0]),
      rel_hole_lengths  = ','.join([str(hl / (hl + actual_length)) for hl in hole_lengths if hl >= 0]),
      hole_ind_length   = len_indices,
      input_feed        = tokenizer.ArrayToCode(input_feed, with_formatting = True),
      input_ids         = str_input_feed,
      encoded_input_ids = ','.join([str(x) for x in input_ids]),
      input_features    = '\n'.join(["{}:{}".format(k, v) for k, v in input_features.items()]) if input_features else "None",
      input_score       = input_score,
      sample            = str_sample,
      sample_indices    = sample_indices,
      num_tokens        = int(num_tokens),
      output_features   = '\n'.join(["{}:{}".format(k, v) for k, v in output_features.items()]) if output_features else "None",
      sample_score      = sample_score,
      target_benchmark  = "// {}\n{}".format(target_benchmark[0], target_benchmark[1]),
      target_features   = '\n'.join(["{}:{}".format(k, v) for k, v in target_features.items()]) if target_features else "None",
      compile_status    = compile_status,
      score_delta       = (sample_score - input_score) / input_score if not math.isinf(input_score) else math.inf,
      features_delta    = '\n'.join(["{}:{}".format(k, output_features[k] - input_features[k]) for k in input_features.keys() if (output_features[k] - input_features[k] != 0)]) if input_features and output_features else math.inf,
      date_added        = datetime.datetime.utcnow(),
    )

class SearchCandidateDatabase(sqlutil.Database):
  """A database for analysis of search generations and candidates."""

  def __init__(self, url: str, must_exist: bool = False):
    super(SearchCandidateDatabase, self).__init__(url, Base, must_exist = must_exist)

  @property
  def count(self):
    """Number of input feeds in DB."""
    with self.Session() as s:
      count = s.query(SearchCandidate).count()
    return count
