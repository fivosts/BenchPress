"""A module for databases of search-based generation."""
import contextlib
import math
import copy
import tqdm
import pathlib
import multiprocessing
import datetime
import typing
import sqlite3
import numpy as np
import sqlalchemy as sql
from sqlalchemy.ext import declarative
from absl import app, flags

from deeplearning.clgen.features import extractor
from deeplearning.clgen.samplers import samples_database
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.util import crypto
from deeplearning.clgen.util import sqlutil
from deeplearning.clgen.util import environment
from deeplearning.clgen.util import distrib

FLAGS = flags.FLAGS

flags.DEFINE_string(
  "active_mergeable_databases",
  None,
  "Comma separated paths of ActiveFeedDatabase to merge into one."
)

flags.DEFINE_string(
  "output_active_db",
  None,
  "Specify output of active merged database."
)

flags.DEFINE_string(
  "output_samples_db",
  None,
  "Specify output of samples merged database."
)

flags.DEFINE_string(
  "active_feed_mode",
  None,
  "Select module's operation. Choices: \"active_to_samples\" and \"merge_active\""
)

Base = declarative.declarative_base()

class ActiveSamplingSpecs(Base):
  __tablename__ = "specifications"
  """
    DB Table for concentrated online/active sampling results.
  """
  sha256                : str = sql.Column(sql.String(1024), primary_key=True)
  active_search_depth   : int = sql.Column(sql.Integer, nullable = False)
  active_search_width   : int = sql.Column(sql.Integer, nullable = False)
  feature_space         : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)

  @classmethod
  def FromArgs(cls,
               act_s_dep  : int,
               act_s_wid  : int,
               feat_space : str
               ) -> typing.TypeVar("ActiveSamplingSpecs"):
    return ActiveSamplingSpecs(
      sha256                = crypto.sha256_str(str(act_s_dep) + str(act_s_wid) + feat_space),
      active_search_depth   = act_s_dep,
      active_search_width   = act_s_wid,
      feature_space         = feat_space,
    )

class ActiveInput(Base, sqlutil.ProtoBackedMixin):
  """
  A database for all original inputs used for active learning.
  """
  __tablename__    = "input_feeds"
  # entry id
  id             : int = sql.Column(sql.Integer,    primary_key = True)
  # unique hash of sample text
  sha256         : str = sql.Column(sql.String(64), nullable = False, index = True)
  # Text original input
  input_feed     : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Encoded original input
  encoded_feed   : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Feature vector of input_feed
  input_features : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Actual length of sample, excluding pads.
  num_tokens     : int = sql.Column(sql.Integer, nullable = False)
  # Date
  date_added     : datetime.datetime = sql.Column(sql.DateTime, nullable = False)

  @classmethod
  def FromArgs(cls,
               tokenizer,
               id             : int,
               input_feed     : np.array,
               input_features : typing.Dict[str, float],
               ) -> typing.TypeVar("ActiveInput"):
    """Construt ActiveFeed table entry from argumentns."""
    str_input_feed = tokenizer.tokensToString(input_feed, ignore_token = tokenizer.padToken)
    if tokenizer.padToken in input_feed:
      num_tokens = np.where(input_feed == tokenizer.padToken)[0][0]
    else:
      num_tokens = len(input_feed)

    return ActiveInput(
      id             = id,
      sha256         = crypto.sha256_str(str_input_feed),
      input_feed     = str_input_feed,
      encoded_feed   = ','.join([str(x) for x in input_feed]),
      input_features = '\n'.join(["{}:{}".format(k, v) for k, v in input_features.items()]),
      num_tokens     = int(num_tokens),
      date_added     = datetime.datetime.utcnow(),
    )

class ActiveFeed(Base, sqlutil.ProtoBackedMixin):
  """
  A database row representing a search-based generational sample.
  """
  __tablename__    = "active_feeds"
  # entry id
  id               : int   = sql.Column(sql.Integer,    primary_key = True)
  # unique hash of sample text
  sha256           : str   = sql.Column(sql.String(64), nullable = False, index = True)
  # Text original input
  input_feed       : str   = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Encoded original input
  encoded_feed     : str   = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Feature vector of input_feed
  input_features   : str   = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Output sample
  sample           : str   = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Actual length of sample, excluding pads.
  num_tokens       : int   = sql.Column(sql.Integer, nullable = False)
  # Sample's vector of features.
  output_features  : str   = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Whether the generated sample is of good quality or not.
  sample_quality   : float = sql.Column(sql.Float,  nullable = False)
  # Name and contents of target benchmark specified.
  target_benchmark : str   = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Feature vector of target benchmark.
  target_features  : str   = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Whether sample compiles or not.
  compile_status   : bool  = sql.Column(sql.Boolean, nullable = False)
  # Number of generation for sample
  generation_id    : int   = sql.Column(sql.Integer, nullable = False)
  # Date
  date_added       : datetime.datetime = sql.Column(sql.DateTime, nullable = False)

  @classmethod
  def FromArgs(cls,
               tokenizer,
               id               : int,
               input_feed       : np.array,
               input_features   : typing.Dict[str, float],
               sample           : np.array,
               output_features  : typing.Dict[str, float],
               sample_quality   : float,
               target_benchmark : typing.Tuple[str, str],
               target_features  : typing.Dict[str, float],
               compile_status   : bool,
               generation_id    : int,
               ) -> typing.TypeVar("ActiveFeed"):
    """Construt ActiveFeed table entry from argumentns."""
    str_input_feed       = tokenizer.tokensToString(input_feed,       ignore_token = tokenizer.padToken, with_formatting = True)
    str_sample           = tokenizer.ArrayToCode(sample, with_formatting = True)

    num_tokens = len(sample)
    if tokenizer.padToken in sample:
      num_tokens = np.where(sample == tokenizer.padToken)[0][0]

    return ActiveFeed(
      id               = id,
      sha256           = crypto.sha256_str(str_input_feed + str_sample),
      input_feed       = str_input_feed,
      encoded_feed     = ','.join([str(x) for x in input_feed]),
      input_features   = '\n'.join(["{}:{}".format(k, v) for k, v in input_features.items()]),
      sample           = str_sample,
      num_tokens       = int(num_tokens),
      output_features  = '\n'.join(["{}:{}".format(k, v) for k, v in output_features.items()]) if output_features else "None",
      target_benchmark = "// {}\n{}".format(target_benchmark[0], target_benchmark[1]),
      target_features  = '\n'.join(["{}:{}".format(k, v) for k, v in target_features.items()]) if target_features else "None",
      sample_quality   = sample_quality,
      compile_status   = compile_status,
      generation_id    = generation_id,
      date_added       = datetime.datetime.utcnow(),
    )

  @classmethod
  def FromActiveFeed(cls,
                     id : int,
                     sha256           : str,
                     input_feed       : str = "",
                     encoded_feed     : str = "",
                     input_features   : str = "",
                     sample           : str = "",
                     num_tokens       : int = -1,
                     output_features  : str = "",
                     target_benchmark : str = "",
                     target_features  : str = "",
                     sample_quality   : float = -1,
                     compile_status   : bool = False,
                     generation_id    : int = -1,
                     date_added : datetime.datetime = datetime.datetime.utcnow()
                     ) -> typing.TypeVar("ActiveFeed"):
    return ActiveFeed(
      id = id,
      sha256           = sha256,
      input_feed       = input_feed,
      encoded_feed     = encoded_feed,
      input_features   = input_features,
      sample           = sample,
      num_tokens       = num_tokens,
      output_features  = output_features,
      target_benchmark = target_benchmark,
      target_features  = target_features,
      sample_quality   = sample_quality,
      compile_status   = compile_status,
      generation_id    = generation_id,
      date_added       = date_added,
    )

class ActiveFeedDatabase(sqlutil.Database):
  """A database monitoring search-based generation process."""

  def __init__(self, url: str, must_exist: bool = False, is_replica = False):
    if environment.WORLD_RANK == 0 or is_replica:
      super(ActiveFeedDatabase, self).__init__(url, Base, must_exist = must_exist)
    if environment.WORLD_SIZE > 1 and not is_replica:
      # Conduct engine connections to replicated preprocessed chunks.
      self.base_path = pathlib.Path(url.replace("sqlite:///", "")).resolve().parent
      hash_id = self.base_path.name
      try:
        tdir = pathlib.Path(FLAGS.local_filesystem).resolve() / hash_id / "node_active_feed"
      except Exception:
        tdir = pathlib.Path("/tmp").resolve() / hash_id / "node_active_feed"
      distrib.lock()
      tdir.mkdir(parents = True, exist_ok = True)
      distrib.unlock()
      self.replicated_path = tdir / "active_feeds_{}.db".format(environment.WORLD_RANK)
      self.replicated = ActiveFeedDatabase(
        url = "sqlite:///{}".format(str(self.replicated_path)),
        must_exist = must_exist,
        is_replica = True
      )
      distrib.barrier()
    return

  @property
  def input_count(self):
    """Number of input feeds in DB."""
    with self.get_session() as s:
      count = s.query(ActiveInput).count()
    return count

  @property
  def get_data(self):
    """Return all database in list format"""
    with self.get_session() as s:
      return s.query(ActiveFeed).all()

  @property
  def get_features(self):
    """Return all feature vectors of compiling samples."""
    with self.get_session() as s:
      return [x.output_features for x in s.query(ActiveFeed).yield_per(1000)]

  @property
  def active_count(self):
    """Number of active samples in DB."""
    with self.get_session() as s:
      count = s.query(ActiveFeed).count()
    return count

  @property
  def get_session(self):
    """
    get proper DB session.
    """
    if environment.WORLD_SIZE == 1 or environment.WORLD_RANK == 0:
      return self.Session
    else:
      return self.replicated.Session

def merge_databases(dbs: typing.List[ActiveFeedDatabase], out_db: ActiveFeedDatabase) -> None:
  """
  Merges a list  of active_feed_databases to a single one, specified in out_db.

  Arguments:
    dbs: List of active feed databases.
    out_db: Exported output database.

  Returns:
    None
  """
  sdir = {}
  new_id = out_db.active_count
  existing = [dp.sha256 for dp in out_db.get_data]
  for db in dbs:
    data = db.get_data
    for dp in data:
      if dp.sha256 not in sdir and dp.sha256 not in existing:
        sdir[dp.sha256] = ActiveFeed.FromActiveFeed(
          id = new_id,
          sha256 = dp.sha256,
          input_feed       = dp.input_feed,
          encoded_feed     = dp.encoded_feed,
          input_features   = dp.input_features,
          sample           = dp.sample,
          num_tokens       = dp.num_tokens,
          output_features  = dp.output_features,
          target_benchmark = dp.target_benchmark,
          target_features  = dp.target_features,
          sample_quality   = dp.sample_quality,
          compile_status   = dp.compile_status,
          generation_id    = dp.generation_id,
          date_added       = dp.date_added,
        )
        new_id += 1
  with out_db.Session() as s:
    bar = tqdm.tqdm(total = len(sdir.values()), desc = "Merged DB")
    for dp in bar(sdir.values()):
      s.add(s.merge(dp))
    s.commit()
  return

def ToProto(dp: ActiveFeed) -> samples_database.Sample:
  return samples_database.Sample(
           **samples_database.Sample.FromProto(0, model_pb2.Sample(
             train_step             = -1,
             text                   = dp.sample,
             sample_indices         = "",
             encoded_sample_indices = "",
             original_input         = "",
             sample_feed            = dp.input_feed,
             encoded_text           = "",
             sample_time_ms         = 0,
             feature_vector         = extractor.ExtractRawFeatures(dp.sample),
             num_tokens             = dp.num_tokens,
             compile_status         = dp.compile_status,
             categorical_sampling   = 1,
             date_added             = dp.date_added.strftime("%m/%d/%Y, %H:%M:%S"),
            )
          )
        )

def active_convert_samples(dbs: typing.List[ActiveFeedDatabase], out_db: samples_database.SamplesDatabase) -> None:
  """
  Merges a list  of active_feed_databases to a SamplesDatabase db.

  Arguments:
    dbs: List of active feed databases.
    out_db: Exported output samples database.

  Returns:
    None
  """
  sdir = {}
  new_id = out_db.count
  existing = [dp.sha256 for dp in out_db.get_data]
  for db in dbs:
    data = []
    pool = multiprocessing.Pool()
    for dp in tqdm.tqdm(pool.imap_unordered(ToProto, db.get_data), total = db.active_count, desc = "{}".format(pathlib.Path(db.url).name)):
      data.append(dp)
    for dp in data:
      if dp.sha256 not in sdir and dp.sha256 not in existing:
        dp.id = new_id
        sdir[dp.sha256] = dp
        new_id += 1
  with out_db.Session() as s:
    for dp in tqdm.tqdm(sdir.values(), total = len(sdir.values()), desc = "Output DB"):
      s.add(dp)
    s.commit()
  return

def initMain(*args, **kwargs):
  """
  Setup module's operations.
  """
  if not FLAGS.active_mergeable_databases:
    raise ValueError("Please input active feed databases to merge as a comma separated list.")
  db_paths = [pathlib.Path(p).absolute() for p in FLAGS.active_mergeable_databases.replace(" ", "").split(",")]
  for p in db_paths:
    if not p.exists():
      raise FileNotFoundError(p)
  dbs = [ActiveFeedDatabase(url = "sqlite:///{}".format(str(p)), must_exist = True) for p in db_paths]

  if FLAGS.active_feed_mode == "merge_active":
    if not FLAGS.output_active_db:
      raise ValueError("Specify out path for merged database")

    out_path = pathlib.Path(FLAGS.output_active_db).absolute()
    if out_path.suffix != '.db':
      raise ValueError("output_active_db must end in a valid database name (.db extension): {}".format(out_path))
    out_path.parent.mkdir(exist_ok = True, parents = True)
    out_db = ActiveFeedDatabase(url = "sqlite:///{}".format(str(out_path)), must_exist = False)

    merge_databases(dbs, out_db)
  elif FLAGS.active_feed_mode == "active_to_samples":

    out_path = pathlib.Path(FLAGS.output_samples_db).absolute()
    if out_path.suffix != '.db':
      raise ValueError("output_samples_db must end in a valid database name (.db extension)")
    out_path.parent.mkdir(exist_ok = True, parents = True)
    out_db = samples_database.SamplesDatabase(url = "sqlite:///{}".format(str(out_path)), must_exist = False)

    active_convert_samples(dbs, out_db)
  else:
    raise ValueError("Invalid value for FLAGS.active_feed_mode: {}".format(FLAGS.active_feed_mode))

  return

if __name__ == "__main__":
  app.run(initMain)
  sys.exit(0)
