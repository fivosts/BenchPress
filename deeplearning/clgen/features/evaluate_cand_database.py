"""A module for databases of CLgen samples."""
import contextlib
import math
import pathlib
import datetime
import typing
import sqlite3
import numpy as np
import sqlalchemy as sql
from sqlalchemy.ext import declarative
from absl import app, flags

from deeplearning.clgen.util import crypto
from deeplearning.clgen.util import sqlutil
from deeplearning.clgen.util import plotter as plt

from eupy.native import logger as l

FLAGS = flags.FLAGS

Base = declarative.declarative_base()

flags.DEFINE_string(
  "eval_cand_db",
  "",
  "Set path of candidatae Database to evaluate."
)

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

  @property
  def get_data(self):
    """Return all database in list format"""
    with self.Session() as s:
      return s.query(SearchCandidate).all()

  @property
  def freq_gen_col(self):
    """Return frequency-generation columns of databse."""
    with self.Session() as s:
      return s.query(SearchCandidate.generation_id, SearchCandidate.frequency).all()

  @property
  def rel_hole_lengths(self):
    """Return column of relative hole lengths"""
    with self.Session() as s:
      return s.query(SearchCandidate.rel_hole_lengths).all()

  @property
  def gen_lenind_abshole(self):
    with self.Session() as s:
      return s.query(
        SearchCandidate.generation_id,
        SearchCandidate.abs_hole_lengths,
        SearchCandidate.hole_ind_length
      ).all()

  @property
  def holes_scores(self):
    """Return hole length, hole filled length, score delta"""
    with self.Session() as s:
      return s.query(
        SearchCandidate.abs_hole_lengths,
        SearchCandidate.hole_ind_length,
        SearchCandidate.score_delta,
        SearchCandidate.compile_status
      ).all()

def run_db_evaluation(db: SearchCandidateDatabase) -> None:


  data = db.get_data
  gen_ids          = []
  frequencies      = []
  token_deltas     = []
  len_indices      = []
  rel_hole_lengths = []
  abs_hole_lengths = []
  score_deltas     = []
  compile_status   = []
  feed_len         = []
  for dp in data:
    gen_ids.append(dp.generation_id)
    frequencies.append(dp.frequency)
    token_deltas.append(dp.hole_ind_length - sum([int(x) for x in dp.abs_hole_lengths.split(',') if x]))
    len_indices.append(dp.hole_ind_length)
    if dp.rel_hole_lengths:
      rel_hole_lengths.append(float(dp.rel_hole_lengths))
    if dp.abs_hole_lengths:
      abs_hole_lengths.append(int(dp.abs_hole_lengths))
    if not math.isinf(dp.score_delta):
      score_deltas.append(dp.score_delta)
    compile_status.append(dp.compile_status)
    feed_len.append(len([int(x) for x in dp.encoded_input_ids.split(',') if x]))

  # 1) Frequency per generation.
  #   x-axis: times occured, y-axis: how many samples did hit these freq.
  #   One group of these distributions per generation.
  data  = db.freq_gen_col
  freqd = {}
  for gen, f in zip(gen_ids, frequencies):
    if gen in freqd:
      if f in freqd[gen]:
        freqd[gen][f] += 1
      else:
        freqd[gen][f] = 1
    else:
      freqd[gen] = {}
      freqd[gen][f] = 1
  for k, v in freqd.items():
    freqd[k] = (list(v.keys()), list(v.values()))
  plt.GrouppedBars(
    groups = freqd, # Dict[Dict[int, int]]
    title = "Frequency of samples per generation",
    x_name = "# of repetitions",
    plot_name = "freq_samples_per_gen"
  )

  # 2) Relative hole length distribution.
  rhl_dist = {}
  for rhl in rel_hole_lengths:
    rounded = int(100*rhl)
    if rounded not in rhl_dist:
      rhl_dist[rounded] = 1
    else:
      rhl_dist[rounded] += 1
  plt.FrequencyBars(
    x = list(rhl_dist.keys()),
    y = list(rhl_dist.values()),
    title = "% hole length distribution",
    x_name = "percentile",
    plot_name = "perc_hole_length_distribution",
  )

  # 3) Per generation: delta of (filled_tokens - hole_length)
  print("Filled tokens - hole length will be wrong for multiple holes!")
  print("For now, I am assigning every hole to the total of sample indices length.")
  gen_hole_deltas = {} # gen -> list of deltas.
  for gen, ahl, lind in zip(gen_ids, abs_hole_lengths, len_indices):
    if gen not in gen_hole_deltas:
      gen_hole_deltas[gen] = []
    gen_hole_deltas[gen].append(lind - int(ahl))

  raise NotImplementedError("The plot below is wrong ? Check. Also remove the V things. Keep the rectangles.")
  plt.CategoricalViolin(
    x = list(gen_hole_deltas.keys()),
    y = list(gen_hole_deltas.values()),
    title = "Hole delta vs generation",
    x_name = "Generation id",
    plot_name = "hole_delta_vs_gen"
  )

  # 4) 2D scatter: token delta vs score delta.
  plt.ScatterPlot(
    x = token_deltas,
    y = score_deltas,
    title = "Token Delta VS Score Delta",
    x_name = "Token Delta",
    y_name = "Score Delta",
    plot_name = "Token Delta VS Score Delta",
  )

  # 5) Bar plot: 6 linear combinations of sign of token delta and score delta (neg, pos, 0.0).
  groups = {
    'token delta > 0, score_delta > 0': 0,
    'token delta > 0, score_delta < 0': 0,
    'token delta > 0, score_delta == 0': 0,
    'token delta < 0, score_delta > 0': 0,
    'token delta < 0, score_delta < 0': 0,
    'token delta < 0, score_delta == 0': 0,
    'token delta == 0, score_delta > 0': 0,
    'token delta == 0, score_delta < 0': 0,
    'token delta == 0, score_delta == 0': 0,
  }
  for td, sd in zip(token_deltas, score_deltas):
    if td > 0:
      if sd > 0:
        groups['token delta > 0, score_delta > 0'] += 1
      elif sd < 0:
        groups['token delta > 0, score_delta < 0'] += 1
      else:
        groups['token delta > 0, score_delta == 0'] += 1
    elif td < 0:
      if sd > 0:
        groups['token delta < 0, score_delta > 0'] += 1
      elif sd < 0:
        groups['token delta < 0, score_delta < 0'] += 1
      else:
        groups['token delta < 0, score_delta == 0'] += 1
    else:
      if sd > 0:
        groups['token delta == 0, score_delta > 0'] += 1
      elif sd < 0:
        groups['token delta == 0, score_delta < 0'] += 1
      else:
        groups['token delta == 0, score_delta == 0'] += 1
  raise NotImplementedError("Move to groupped bars, switch to percentiles")
  plt.FrequencyBars(
    x = list(groups.keys()),
    y = list(groups.values()),
    title = "Sample Frequency VS token & score delta",
    x_name = "category",
    plot_name = "token_score_deltas",
  )

  # 6) Bar plot: 4 linear combinations of compilability and token delta.
  groups = {
    'token delta > 0, compile': 0,
    'token delta > 0, not-compile': 0,
    'token delta < 0, compile': 0,
    'token delta < 0, not-compile': 0,
    'token delta == 0, compile': 0,
    'token delta == 0, not-compile': 0,
  }
  for td, cs in zip(token_deltas, compile_status):
    if td > 0:
      if cs == 1:
        groups['token delta > 0, compile'] += 1
      else:
        groups['token delta > 0, not-compile'] += 1
    elif td < 0:
      if cs == 1:
        groups['token delta < 0, compile'] += 1
      else:
        groups['token delta < 0, not-compile'] += 1
    else:
      if cs == 1:
        groups['token delta == 0, compile'] += 1
      else:
        groups['token delta == 0, not-compile'] += 1

  plt.FrequencyBars(
    x = list(groups.keys()),
    y = list(groups.values()),
    title = "Compilability VS token & score delta",
    x_name = "category",
    plot_name = "comp_score_token_delta",
  )

  # 7) 2D scatter per generation: rel hole length vs score delta.
  plt.ScatterPlot(
    x = [int(100*x) for x in rel_hole_lengths],
    y = score_deltas,
    x_name = "Relative Hole Length",
    y_name = "Score Delta",
    title = "Relative Hole Length VS Score Delta",
    plot_name = "rel_hl_score_delta",
  )

  raise NotImplementedError("Is this correct ? Check.")
  # 8) token delta vs len_input_feed.
  plt.ScatterPlot(
    x = feed_len,
    y = token_deltas,
    x_name = "Input Feed Length",
    y_name = "Token Delta",
    title = "Input Length VS Token Delta",
    plot_name = "feed_len_token_delta",
  )
  return

def initMain(*args, **kwargs):
  # l.initLogger(name = "eval_cand_db")
  db_path = pathlib.Path(FLAGS.eval_cand_db).absolute()
  if not db_path.exists():
    raise FileNotFoundError(str(db_path))
  db = SearchCandidateDatabase(url = "sqlite:///{}".format(str(db_path)), must_exist = True)
  run_db_evaluation(db)
  return

if __name__ == "__main__":
  app.run(initMain)
  sys.exit(0)