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
"""A module for databases of BenchPress samples."""
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

from deeplearning.benchpress.util import crypto
from deeplearning.benchpress.util import sqlutil
from deeplearning.benchpress.util import plotter as plt

from deeplearning.benchpress.util import logging as l

FLAGS = flags.FLAGS

Base = declarative.declarative_base()

flags.DEFINE_string(
  "eval_cand_db",
  "",
  "Set path of candidatae Database to evaluate."
)

class SearchCandidate(Base, sqlutil.ProtoBackedMixin):
  """A database row representing a BenchPress sample.

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
  # The runtime features it must achieve.
  runtime_features  : str   = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
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
               runtime_features : typing.Dict[str, float],
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
      sha256            = crypto.sha256_str(str_input_feed + str_sample + str(hole_lengths) + target_benchmark[0]),
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
      runtime_features  = '\n'.join(["{}:{}".format(k, v) for k, v in runtime_features.items()]) if runtime_features else "None",
      sample_score      = sample_score,
      target_benchmark  = "// {}\n{}".format(target_benchmark[0], target_benchmark[1]),
      target_features   = '\n'.join(["{}:{}".format(k, v) for k, v in target_features.items()]) if target_features else "None",
      compile_status    = compile_status,
      score_delta       = (sample_score - input_score) / input_score if not math.isinf(input_score) and input_score > 0 else math.inf,
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

def input_samples_distribution(data) -> None:
  # 1) Frequency per generation.
  #   x-axis: times occured, y-axis: how many samples did hit these freq.
  #   One group of these distributions per generation.
  freqd = {}
  for dp in data:
    gen, f = dp.generation_id, dp.frequency
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
    plot_name = "freq_input_samples_per_gen",
    path = pathlib.Path(FLAGS.eval_cand_db).absolute().parent,
    title = "Repetition of input/samples pair per generation",
    x_name = "# of repetitions",
  )
  return

def samples_distribution(data) -> None:
  freqd = {}
  for dp in data:
    gen, sam = dp.generation_id, dp.sample
    hsm = crypto.sha256_str(sam)
    if gen in freqd:
      if hsm in freqd[gen]:
        freqd[gen][hsm] += 1
      else:
        freqd[gen][hsm] = 1
    else:
      freqd[gen] = {}
      freqd[gen][hsm] = 1
  for k, v in freqd.items():
    gdict = {}
    for samp, freq in v.items():
      if freq in gdict:
        gdict[freq] += 1
      else:
        gdict[freq] = 1
    freqd[k] = (list(gdict.keys()), list(gdict.values()))
  plt.GrouppedBars(
    groups = freqd, # Dict[Dict[int, int]]
    plot_name = "freq_samples_per_gen",
    path = pathlib.Path(FLAGS.eval_cand_db).absolute().parent,
    title = "Repetition of samples per generation",
    x_name = "# of repetitions",
  )

  return

def rel_length_distribution(data) -> None:
  # 2) Relative hole length distribution.
  rhl_dist = {}
  for dp in data:
    rhl = dp.rel_hole_lengths
    try:
      rounded = int(100*float(rhl))
      if rounded not in rhl_dist:
        rhl_dist[rounded] = 1
      else:
        rhl_dist[rounded] += 1
    except Exception:
      continue
  plt.FrequencyBars(
    x = list(rhl_dist.keys()),
    y = list(rhl_dist.values()),
    plot_name = "perc_hole_length_distribution",
    path = pathlib.Path(FLAGS.eval_cand_db).absolute().parent,
    title = "% hole length distribution",
    x_name = "percentile",
  )
  return

def token_delta_per_gen(data) -> None:
  # 3) Per generation: delta of (filled_tokens - hole_length)
  logger.warn("Filled tokens - hole length will be wrong for multiple holes!")
  logger.warn("For now, I am assigning every hole to the total of sample indices length.")
  gen_hole_deltas = {} # gen -> list of deltas.
  for dp in data:
    gen, ahl, lind = dp.generation_id, dp.abs_hole_lengths, dp.hole_ind_length
    try:
      ahl = sum([int(x) for x in ahl.split(',') if x])
      if gen not in gen_hole_deltas:
        gen_hole_deltas[gen] = []
      gen_hole_deltas[gen].append(lind - int(ahl))
    except Exception:
      continue

  plt.CategoricalViolin(
    x = list(gen_hole_deltas.keys()),
    y = list(gen_hole_deltas.values()),
    plot_name = "hole_delta_vs_gen",
    path = pathlib.Path(FLAGS.eval_cand_db).absolute().parent,
    title = "Hole delta vs generation",
    x_name = "Generation id",
  )
  return

def comp_token_delta_per_gen(data) -> None:
  # 3) Per generation: delta of (filled_tokens - hole_length)
  gen_hole_deltas = {} # gen -> list of deltas.
  for dp in data:
    if dp.compile_status == 1:
      gen, ahl, lind = dp.generation_id, dp.abs_hole_lengths, dp.hole_ind_length
      try:
        ahl = sum([int(x) for x in ahl.split(',') if x])
        if gen not in gen_hole_deltas:
          gen_hole_deltas[gen] = []
        gen_hole_deltas[gen].append(lind - int(ahl))
      except Exception:
        continue

  plt.CategoricalViolin(
    x = list(gen_hole_deltas.keys()),
    y = list(gen_hole_deltas.values()),
    plot_name = "comp_hole_delta_vs_gen",
    path = pathlib.Path(FLAGS.eval_cand_db).absolute().parent,
    title = "Hole delta vs generation for compilable samples",
    x_name = "Generation id",
  )
  return

def nohole_in_end_token_delta_per_gen(data) -> None:
  # 3) Per generation: delta of (filled_tokens - hole_length)
  gen_hole_deltas = {} # gen -> list of deltas.
  for dp in data:
    if "[HOLE][END]" not in dp.input_ids.replace('\n', '').replace(' ', ''):
      gen, ahl, lind = dp.generation_id, dp.abs_hole_lengths, dp.hole_ind_length
      try:
        ahl = sum([int(x) for x in ahl.split(',') if x])
        if gen not in gen_hole_deltas:
          gen_hole_deltas[gen] = []
        gen_hole_deltas[gen].append(lind - int(ahl))
      except Exception:
        continue

  plt.CategoricalViolin(
    x = list(gen_hole_deltas.keys()),
    y = list(gen_hole_deltas.values()),
    plot_name = "nohole_end_token_delta_vs_gen",
    path = pathlib.Path(FLAGS.eval_cand_db).absolute().parent,
    title = "Hole delta vs generation for input ids with no hole before end",
    x_name = "Generation id",
  )
  return

def token_score_delta_scatter(data) -> None:
  # 4) 2D scatter: token delta vs score delta.
  tds, sds = [], []
  for dp in data:
    td = dp.hole_ind_length - sum([int(x) for x in dp.abs_hole_lengths.split(',') if x])
    sd = dp.score_delta if not math.isinf(dp.score_delta) else None
    if sd is not None and td is not None:
      tds.append(td)
      sds.append(sd)
  plt.ScatterPlot(
    x = tds,
    y = sds,
    plot_name = "token_delta_vs_score_delta",
    path = pathlib.Path(FLAGS.eval_cand_db).absolute().parent,
    title = "Token Delta VS Score Delta",
    x_name = "Token Delta",
    y_name = "Score Delta",
  )
  return  

def score_vs_token_delta(data) -> None:
  # 5) Bar plot: 6 linear combinations of sign of token delta and score delta (neg, pos, 0.0).
  groups = {
    'better score' : [['token delta > 0', 'token delta < 0', 'token delta == 0'], [0, 0, 0]],
    'worse score'  : [['token delta > 0', 'token delta < 0', 'token delta == 0'], [0, 0, 0]],
    'same score'   : [['token delta > 0', 'token delta < 0', 'token delta == 0'], [0, 0, 0]],
  }
  nsum = 0
  for dp in data:
    td = dp.hole_ind_length - sum([int(x) for x in dp.abs_hole_lengths.split(',') if x])
    sd = dp.score_delta if not math.isinf(dp.score_delta) else None
    if sd is not None and td is not None:
      nsum += 1
      if sd < 0:
        if td > 0:
          groups['better score'][1][0] += 1
        elif td < 0:
          groups['better score'][1][1] += 1
        else:
          groups['better score'][1][2] += 1
      elif sd > 0:
        if td > 0:
          groups['worse score'][1][0] += 1
        elif td < 0:
          groups['worse score'][1][1] += 1
        else:
          groups['worse score'][1][2] += 1
      else:
        if td > 0:
          groups['same score'][1][0] += 1
        elif td < 0:
          groups['same score'][1][1] += 1
        else:
          groups['same score'][1][2] += 1
  for k, v in groups.items():
    for idx, nv in enumerate(v[1]):
      groups[k][1][idx] = 100 * (nv / nsum)
  plt.GrouppedBars(
    groups = groups,
    plot_name = "token_score_deltas",
    path = pathlib.Path(FLAGS.eval_cand_db).absolute().parent,
    title = "Sample Frequency % VS token & score delta",
    x_name = "category",
  )
  return

def comp_vs_token_delta(data) -> None:
  # 6) Bar plot: 4 linear combinations of compilability and token delta.
  groups = {
    'token delta > 0': [['compile', 'not-compile'], [0, 0]],
    'token delta < 0': [['compile', 'not-compile'], [0, 0]],
    'token delta == 0': [['compile', 'not-compile'], [0, 0]],
  }
  nsum = 0
  for dp in data:
    td = dp.hole_ind_length - sum([int(x) for x in dp.abs_hole_lengths.split(',') if x])
    cs = dp.compile_status
    if td is not None and cs is not None:
      nsum += 1
      if td > 0:
        if cs == 1:
          groups['token delta > 0'][1][0] += 1
        else:
          groups['token delta > 0'][1][1] += 1
      elif td < 0:
        if cs == 1:
          groups['token delta < 0'][1][0] += 1
        else:
          groups['token delta < 0'][1][1] += 1
      else:
        if cs == 1:
          groups['token delta == 0'][1][0] += 1
        else:
          groups['token delta == 0'][1][1] += 1
  for k, v in groups.items():
    for idx, nv in enumerate(v[1]):
      groups[k][1][idx] = 100 * (nv / nsum)

  plt.GrouppedBars(
    groups = groups,
    plot_name = "comp_token_delta",
    path = pathlib.Path(FLAGS.eval_cand_db).absolute().parent,
    title = "Sample Frequency % VS Compilability & token delta",
    x_name = "category",
  )
  return

def rel_length_score(data) -> None:
  # 7) 2D scatter per generation: rel hole length vs score delta.
  rhl_lens = []
  scd      = []
  for dp in data:
    try:
      rhl = int(100*float(dp.rel_hole_lengths))
      sd = dp.score_delta
      if rhl is not None and not math.isinf(sd):
        rhl_lens.append(rhl)
        scd.append(sd)
    except Exception:
      continue

  plt.ScatterPlot(
    x = rhl_lens,
    y = scd,
    plot_name = "rel_hl_score_delta",
    path = pathlib.Path(FLAGS.eval_cand_db).absolute().parent,
    x_name = "Relative Hole Length",
    y_name = "Score Delta",
    title = "Relative Hole Length VS Score Delta",
  )
  return

def token_delta_vs_len_input(data) -> None:
  # 8) token delta vs len_input_feed.
  feed_len = []
  token_deltas = []
  for dp in data:
    td = dp.hole_ind_length - sum([int(x) for x in dp.abs_hole_lengths.split(',') if x])
    if td is not None:
      token_deltas.append(td)
      feed_len.append(len([int(x) for x in dp.encoded_input_ids.split(',') if x]))

  plt.ScatterPlot(
    x = feed_len,
    y = token_deltas,
    plot_name = "feed_len_token_delta",
    path = pathlib.Path(FLAGS.eval_cand_db).absolute().parent,
    x_name = "Input Feed Length",
    y_name = "Token Delta",
    title = "Input Length VS Token Delta",
  )
  return

def token_vs_rel_len(data) -> None:
  # Token Delta vs Relative hole length percentile.
  tds = []
  rhl_list = []
  for dp in data:
    try:
      rhl = dp.rel_hole_lengths
      rounded = int(100*float(rhl))
      td = dp.hole_ind_length - sum([int(x) for x in dp.abs_hole_lengths.split(',') if x])
      if td is not None:
        tds.append(td)
        rhl_list.append(rhl)
    except Exception:
      continue
  plt.ScatterPlot(
    x = rhl_list,
    y = tds,
    plot_name = "rel_hl_token_delta",
    path = pathlib.Path(FLAGS.eval_cand_db).absolute().parent,
    x_name = "Relative Hole length %",
    y_name = "Token Delta",
    title = "Rel. Hole length VS Token Delta",
  )
  return

def score_per_abs_hlen(data) -> None:
  """
  Groupped bars of a) better, b) same, c) worse score, per absolute hole length unit.
  """
  abshl = []
  score_ds = []
  groups = {
    'better score' : {},
    'worse score'  : {},
    'same score'   : {},
  }
  max_abs = 0
  for dp in data:
    try:
      ahl = sum([int(x) for x in dp.abs_hole_lengths.split(',') if x])
      max_abs = max(max_abs, ahl)
      sd = dp.score_delta
      if not math.isinf(sd):
        if sd > 0:
          k = 'worse score'
        elif sd < 0:
          k = 'better score'
        else:
          k = 'same score'
        if str(ahl) not in groups[k]:
          groups[k][str(ahl)] = 1
        else:
          groups[k][str(ahl)] += 1
    except Exception as e:
      logger.error(e)
      continue
  for l in range(0, max_abs):
    total = 0
    for k, v in groups.items():
      if str(l) in v:
        total += v[str(l)]
    for k, v in groups.items():
      if str(l) in v:
        groups[k][str(l)] = 100 * (v[str(l)] / total)
  for k, v in groups.items():
    groups[k] = (list(v.keys()), list(v.values()))

  plt.GrouppedBars(
    groups = groups, # Dict[Dict[int, int]]
    plot_name = "score_per_abs_hlen",
    path = pathlib.Path(FLAGS.eval_cand_db).absolute().parent,
    title = "Score Direction (%) per Absolute Hole Length",
    x_name = "Size of Hole",
  )

def comp_vs_len_indices(data) -> None:
  """
  Groupped bars of a) better, b) same, c) worse score, per absolute hole length unit.
  """
  abshl = []
  score_ds = []
  groups = {
    'compile' : {},
    'not-compile'   : {},
  }
  max_len_ind = 0
  for dp in data:
    try:
      len_ind = dp.hole_ind_length
      max_len_ind = max(max_len_ind, len_ind)
      cs = dp.compile_status
      if cs ==1:
        k = 'compile'
      else:
        k = 'not-compile'
      if str(len_ind) not in groups[k]:
        groups[k][str(len_ind)] = 1
      else:
        groups[k][str(len_ind)] += 1
    except Exception as e:
      logger.error(e)
      continue
  # for l in range(0, max_len_ind):
  #   total = 0
  #   for k, v in groups.items():
  #     if str(l) in v:
  #       total += v[str(l)]
    # for k, v in groups.items():
    #   if str(l) in v:
    #     groups[k][str(l)] = 100 * (v[str(l)] / total)
  for k, v in groups.items():
    groups[k] = (list(v.keys()), list(v.values()))

  plt.GrouppedBars(
    groups = groups, # Dict[Dict[int, int]]
    plot_name = "comp_per_len_indices",
    path = pathlib.Path(FLAGS.eval_cand_db).absolute().parent,
    title = "Compilability VS Length of Indices",
    x_name = "Length of Indices",
  )
  return

def comp_vs_len_indices_over_len_input(data) -> None:
  """
  Groupped bars of a) better, b) same, c) worse score, per absolute hole length unit.
  """
  abshl = []
  score_ds = []
  groups = {
    'compile' : {},
    'not-compile'   : {},
  }
  max_len_ind = 0.0
  for dp in data:
    try:
      len_ratio = dp.hole_ind_length / len([int(x) for x in dp.encoded_input_ids.split(',') if x])
      len_ratio = round(len_ratio, 1)
      max_len_ind = max(max_len_ind, len_ratio)
      cs = dp.compile_status
      if cs ==1:
        k = 'compile'
      else:
        k = 'not-compile'
      if str(len_ratio) not in groups[k]:
        groups[k][str(len_ratio)] = 1
      else:
        groups[k][str(len_ratio)] += 1
    except Exception as e:
      logger.error(e)
      continue
  # for l in range(0, max_len_ind):
  #   total = 0
  #   for k, v in groups.items():
  #     if str(l) in v:
  #       total += v[str(l)]
    # for k, v in groups.items():
    #   if str(l) in v:
    #     groups[k][str(l)] = 100 * (v[str(l)] / total)
  for k, v in groups.items():
    groups[k] = (list(v.keys()), list(v.values()))

  plt.GrouppedBars(
    groups = groups, # Dict[Dict[int, int]]
    plot_name = "comp_per_indices_input_len_ratio",
    path = pathlib.Path(FLAGS.eval_cand_db).absolute().parent,
    title = "Compilability VS (Length of Indices / Length of Input)",
    x_name = "Length of Indices / Length of Input",
  )
  return

def comp_vs_num_tokens(data) -> None:
  """
  Groupped bars of a) better, b) same, c) worse score, per absolute hole length unit.
  """
  abshl = []
  score_ds = []
  groups = {
    'compile' : {},
    'not-compile'   : {},
  }
  max_numtok = 0
  for dp in data:
    try:
      numtok = dp.num_tokens
      max_numtok = max(max_numtok, numtok)
      cs = dp.compile_status
      if cs ==1:
        k = 'compile'
      else:
        k = 'not-compile'
      if str(numtok) not in groups[k]:
        groups[k][str(numtok)] = 1
      else:
        groups[k][str(numtok)] += 1
    except Exception as e:
      logger.error(e)
      continue
  for l in range(0, max_numtok):
    total = 0
    for k, v in groups.items():
      if str(l) in v:
        total += v[str(l)]
    for k, v in groups.items():
      if str(l) in v:
        groups[k][str(l)] = 100 * (v[str(l)] / total)
  for k, v in groups.items():
    groups[k] = (list(v.keys()), list(v.values()))

  plt.GrouppedBars(
    groups = groups, # Dict[Dict[int, int]]
    plot_name = "comp_per_len_sample",
    path = pathlib.Path(FLAGS.eval_cand_db).absolute().parent,
    title = "Compilability % VS Length of Sample",
    x_name = "Length of Sample",
  )
  return

def score_per_rel_hlen(data) -> None:
  """
  Groupped bars of a) better, b) same, c) worse score, per absolute hole length unit.
  """
  abshl = []
  score_ds = []
  groups = {
    'better score' : {},
    'worse score'  : {},
    'same score'   : {},
  }
  max_abs = 0
  for dp in data:
    try:
      rhl = dp.rel_hole_lengths
      rounded = int(100*float(rhl))
      max_abs = max(max_abs, rounded)
      sd = dp.score_delta
      if not math.isinf(sd):
        if sd > 0:
          k = 'worse score'
        elif sd < 0:
          k = 'better score'
        else:
          k = 'same score'
        if str(rounded) not in groups[k]:
          groups[k][str(rounded)] = 1
        else:
          groups[k][str(rounded)] += 1
    except Exception as e:
      continue
  for l in range(0, max_abs):
    total = 0
    for k, v in groups.items():
      if str(l) in v:
        total += v[str(l)]
    for k, v in groups.items():
      if str(l) in v:
        groups[k][str(l)] = 100 * (v[str(l)] / total)
  for k, v in groups.items():
    groups[k] = (list(v.keys()), list(v.values()))

  plt.GrouppedBars(
    groups = groups, # Dict[Dict[int, int]]
    plot_name = "score_per_rel_hlen",
    path = pathlib.Path(FLAGS.eval_cand_db).absolute().parent,
    title = "Score Direction (%) per Relative Hole Length",
    x_name = "Size of Hole %",
  )

def score_direction_categorical_distrib_per_rel_hlen(data) -> None:
  """
  Groupped bars of a) better, b) same, c) worse score, per absolute hole length unit.
  """
  abshl = []
  score_ds = []
  groups = {
    'better score' : {},
    'worse score'  : {},
    'same score'   : {},
  }
  normalizers = {
    'better score' : 0,
    'worse score'  : 0,
    'same score'   : 0,
  }
  max_abs = 0
  for dp in data:
    try:
      rhl = dp.rel_hole_lengths
      rounded = int(100*float(rhl))
      max_abs = max(max_abs, rounded)
      sd = dp.score_delta
      if not math.isinf(sd):
        if sd > 0:
          k = 'worse score'
          normalizers[k] += 1
        elif sd < 0:
          k = 'better score'
          normalizers[k] += 1
        else:
          k = 'same score'
          normalizers[k] += 1
        if str(rounded) not in groups[k]:
          groups[k][str(rounded)] = 1
        else:
          groups[k][str(rounded)] += 1
    except Exception as e:
      continue
  for k, v in groups.items():
    for rhlk, rhlv in v.items():
      groups[k][rhlk] = groups[k][rhlk] / normalizers[k]

  for k, v in groups.items():
    groups[k] = (list(v.keys()), list(v.values()))

  plt.GrouppedBars(
    groups = groups, # Dict[Dict[int, int]]
    plot_name = "score_cat_distrib_rel_hlen",
    path = pathlib.Path(FLAGS.eval_cand_db).absolute().parent,
    title = "Score Direction Distribution per Category VS Relative Hole Length",
    x_name = "Size of Hole %",
  )

def run_db_evaluation(db: SearchCandidateDatabase) -> None:

  data = db.get_data
  input_samples_distribution(data)
  samples_distribution(data)
  rel_length_distribution(data)
  comp_token_delta_per_gen(data)
  nohole_in_end_token_delta_per_gen(data)
  token_delta_per_gen(data)
  token_score_delta_scatter(data)
  score_vs_token_delta(data)
  comp_vs_num_tokens(data)
  comp_vs_token_delta(data)
  comp_vs_len_indices(data) ######## I also need per (len_indices / len input feed) ratio.
  comp_vs_len_indices_over_len_input(data)
  rel_length_score(data)
  token_delta_vs_len_input(data)
  token_vs_rel_len(data)
  score_per_abs_hlen(data)
  score_per_rel_hlen(data)
  score_direction_categorical_distrib_per_rel_hlen(data)
  # raise NotImplementedError("bars of better, same, worse score per rel hole length.")
  return

def initMain(*args, **kwargs):
  l.initLogger(name = "eval_cand_db")
  db_path = pathlib.Path(FLAGS.eval_cand_db).absolute()
  if not db_path.exists():
    raise FileNotFoundError(str(db_path))
  db = SearchCandidateDatabase(url = "sqlite:///{}".format(str(db_path)), must_exist = True)
  run_db_evaluation(db)
  return

if __name__ == "__main__":
  app.run(initMain)
  sys.exit(0)