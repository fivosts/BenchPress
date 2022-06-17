"""A module for databases of CLgen samples."""
import contextlib
import datetime
import typing
import multiprocessing
import progressbar
import sqlite3
import functools
import pathlib
import tqdm

import sqlalchemy as sql
from sqlalchemy.ext import declarative
from absl import app, flags

from deeplearning.clgen.features import extractor
from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.util import crypto
from deeplearning.clgen.util import distrib
from deeplearning.clgen.util import environment
from deeplearning.clgen.util import sqlutil
from deeplearning.clgen.util import logging as l

FLAGS = flags.FLAGS

flags.DEFINE_string(
  "sample_mergeable_databases",
  None,
  "Comma separated paths of SamplesDatabase to merge into one."
)

flags.DEFINE_string(
  "sample_merged_path",
  None,
  "Specify output of merged database."
)

flags.DEFINE_string(
  "tokenizer_path",
  None,
  "Specify path of tokenizer to update database."
)

Base = declarative.declarative_base()

class SampleResults(Base):
  __tablename__ = "sampling_results"
  """
    DB Table for concentrated validation results.
  """
  key     : str = sql.Column(sql.String(1024), primary_key=True)
  results : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)

class Sample(Base, sqlutil.ProtoBackedMixin):
  """A database row representing a CLgen sample.

  This is the clgen.Sample protocol buffer in SQL format.
  """
  __tablename__    = "samples"
  # entry id
  id                     : int = sql.Column(sql.Integer,    primary_key = True)
  # unique hash of sample text
  sha256                 : str = sql.Column(sql.String(64), nullable = False, index = True)
  # model's train step that generated the sample
  train_step             : int = sql.Column(sql.Integer,    nullable = False)
  # Original input where the feed came from
  original_input         : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Starting feed of model
  sample_feed            : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # String-format generated text
  text                   : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Array of the actual generated tokens
  sample_indices         : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # encoded sample text
  encoded_text           : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Encoded generated tokens
  encoded_sample_indices : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Whether the generated sample compiles or not.
  compile_status         : bool = sql.Column(sql.Boolean,  nullable = False)
  # Sample's vector of features.
  feature_vector         : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Length of total sequence in number of tokens
  num_tokens             : int = sql.Column(sql.Integer,   nullable = False)
  # If Bernoulli distribution was used during samplinng
  categorical_sampling   : str = sql.Column(sql.String(8), nullable = False)
  # Time
  sample_time_ms         : int = sql.Column(sql.Integer,   nullable = False)
  # Date
  date_added             : datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromProto(cls, id: int, proto: model_pb2.Sample) -> typing.Dict[str, typing.Any]:
    return {
      "id"                     : id,
      "sha256"                 : crypto.sha256_str(proto.text),
      "train_step"             : proto.train_step,
      "encoded_text"           : proto.encoded_text,
      "original_input"         : proto.original_input,
      "sample_feed"            : proto.sample_feed,
      "text"                   : proto.text,
      "sample_indices"         : proto.sample_indices,
      "encoded_sample_indices" : proto.encoded_sample_indices,
      "feature_vector"         : proto.feature_vector,
      "num_tokens"             : proto.num_tokens,
      "compile_status"         : proto.compile_status,
      "categorical_sampling"   : proto.categorical_sampling,
      "sample_time_ms"         : proto.sample_time_ms,
      "date_added"             : datetime.datetime.strptime(proto.date_added, "%m/%d/%Y, %H:%M:%S"),
    }

  @classmethod
  def FromArgsLite(cls, id: int, text: str, feature_vector: str, compiles: bool) -> "Sample":
    """
    Do you want to use SamplesDatabase as a means to store only code
    without much fuss ? This function is for you!
    """
    return Sample(**{
      "id"                     : id,
      "sha256"                 : crypto.sha256_str(text),
      "train_step"             : -1,
      "encoded_text"           : "",
      "original_input"         : "",
      "sample_feed"            : "",
      "text"                   : text,
      "sample_indices"         : "",
      "encoded_sample_indices" : "",
      "compile_status"         : compiles,
      "feature_vector"         : feature_vector,
      "num_tokens"             : 0,
      "categorical_sampling"   : "False",
      "sample_time_ms"         : 0,
      "date_added"             : datetime.datetime.utcnow(),
    })

class SamplesDatabase(sqlutil.Database):
  """A database of CLgen samples."""

  def __init__(self, url: str, must_exist: bool = False, is_replica: bool = False):
    self.base_url = url
    if environment.WORLD_RANK == 0 or is_replica:
      super(SamplesDatabase, self).__init__(url, Base, must_exist = must_exist)
    if environment.WORLD_SIZE > 1 and not is_replica:
      # Conduct engine connections to replicated preprocessed chunks.
      self.base_path = pathlib.Path(url.replace("sqlite:///", "")).resolve().parent
      hash_id = self.base_path.name
      try:
        tdir = pathlib.Path(FLAGS.local_filesystem).resolve() / hash_id / "lm_samples"
      except Exception:
        tdir = pathlib.Path("/tmp").resolve() / hash_id / "lm_samples"
      try:
        tdir.mkdir(parents = True, exist_ok = True)
      except Exception:
        pass
      self.replicated_path = tdir / "samples_{}.db".format(environment.WORLD_RANK)
      self.replicated = SamplesDatabase(
        url = "sqlite:///{}".format(str(self.replicated_path)),
        must_exist = must_exist,
        is_replica = True
      )
      distrib.barrier()

  @property
  def url(self):
    """
    Return Database URL
    """
    if environment.WORLD_RANK == 0:
      return self.base_url
    else:
      return self.replicated.base_url

  @property
  def get_session(self):
    """
    get proper DB session.
    """
    if environment.WORLD_RANK == 0:
      return self.Session
    else:
      return self.replicated.Session

  @property
  def count(self):
    """Number of samples in DB."""
    with self.get_session() as s:
      count = s.query(Sample).count()
    return count

  @property
  def get_data(self):
    """Return all database in list format"""
    with self.get_session() as s:
      return s.query(Sample).all()

  @property
  def get_hash_entries(self):
    """Return all unique hash entries found in DB."""
    with self.get_session() as s:
      return s.query(Sample.sha256).all()

  @property
  def samples(self) -> typing.List[Sample]:
    """Get a list of all files in database."""
    with self.get_session() as s:
      return s.query(Sample).yield_per(1000)

  @property
  def correct_samples(self) -> typing.Set[str]:
    """Get samples that compile from SamplesDatabase."""
    with self.get_session() as s:
      return s.query(Sample).filter(Sample.compile_status == True).yield_per(1000).enable_eagerloads(False)

  @property
  def get_features(self) -> typing.List[typing.Dict[str, float]]:
    """Return all feature vectors of compiling samples."""
    with self.get_session() as s:
      return [x.feature_vector for x in s.query(Sample).filter(Sample.compile_status == True).yield_per(1000)]

  @property
  def get_data_features(self) -> typing.List[typing.Tuple[str, typing.Dict[str, float]]]:
    """Return tuple of code + feature vectors"""
    with self.get_session() as s:
      return [(x.text, x.feature_vector) for x in s.query(Sample).filter(Sample.compile_status == True).yield_per(1000)]

  @property
  def get_samples_features(self) -> typing.List[typing.Tuple[str, typing.Dict[str, float]]]:
    """Return compiling samples with feature vectors"""
    with self.get_session() as s:
      return [(x.text, extractor.RawToDictFeats(x.feature_vector)) for x in s.query(Sample).filter(Sample.compile_status == True).yield_per(1000)]

  @property
  def get_compilable_num_tokens(self) -> typing.List[int]:
    """Return num_tokens column."""
    with self.get_session() as s:
      return [int(x[0]) for x in s.query(Sample.num_tokens).filter(Sample.compile_status == True)]

  def get_by_ids(self, ids):
    """Index and return sample by ID."""
    with self.get_session() as s:
      return [s.query(Sample).filter(Sample.id == i).first() for i in ids]

def merge_databases(dbs: typing.List[SamplesDatabase], out_db: SamplesDatabase) -> None:
  sdir = {}
  new_id = 0
  for db in dbs:
    data = db.get_data
    for dp in data:
      if dp.hash not in sdir:
        dp.id = new_id
        sdir[dp.hash] = dp
        new_id += 1
  with out_db.Session() as s:
    for dp in sdir.values():
      s.add(dp)
    s.commit()
  return

def run_extractors(sample: Sample) -> Sample:
  if sample.compile_status:
    return Sample(
             **Sample.FromProto(0, model_pb2.Sample(
               train_step             = sample.train_step,
               text                   = sample.text,
               sample_indices         = sample.sample_indices,
               encoded_sample_indices = sample.encoded_sample_indices,
               original_input         = sample.original_input,
               sample_feed            = sample.sample_feed,
               encoded_text           = sample.encoded_text,
               sample_time_ms         = sample.sample_time_ms,
               feature_vector         = extractor.ExtractRawFeatures(sample.text),
               num_tokens             = sample.num_tokens,
               compile_status         = sample.compile_status,
               categorical_sampling   = int(sample.categorical_sampling),
               date_added             = sample.date_added.strftime("%m/%d/%Y, %H:%M:%S"),
              )
            )
          )
  else:
    return Sample(
             **Sample.FromProto(0, model_pb2.Sample(
               train_step             = sample.train_step,
               text                   = sample.text,
               sample_indices         = sample.sample_indices,
               encoded_sample_indices = sample.encoded_sample_indices,
               original_input         = sample.original_input,
               sample_feed            = sample.sample_feed,
               encoded_text           = sample.encoded_text,
               sample_time_ms         = sample.sample_time_ms,
               feature_vector         = "",
               num_tokens             = sample.num_tokens,
               compile_status         = sample.compile_status,
               categorical_sampling   = int(sample.categorical_sampling),
               date_added             = sample.date_added.strftime("%m/%d/%Y, %H:%M:%S"),
              )
            )
          )

def get_sample(sample: Sample) -> Sample:
    return Sample(
             **Sample.FromProto(0, model_pb2.Sample(
               train_step             = sample.train_step,
               text                   = sample.text,
               sample_indices         = sample.sample_indices,
               encoded_sample_indices = sample.encoded_sample_indices,
               original_input         = sample.original_input,
               sample_feed            = sample.sample_feed,
               encoded_text           = sample.encoded_text,
               sample_time_ms         = sample.sample_time_ms,
               feature_vector         = sample.feature_vector,
               num_tokens             = sample.num_tokens,
               compile_status         = sample.compile_status,
               categorical_sampling   = int(sample.categorical_sampling),
               date_added             = sample.date_added.strftime("%m/%d/%Y, %H:%M:%S"),
              )
            )
          )

def modernize_samples_db(db: SamplesDatabase, out_db: SamplesDatabase) -> None:
  """
  Re-run feature extractors to update old db.
  """
  pool = multiprocessing.Pool()
  inp_data = db.get_data
  bar = progressbar.ProgressBar(max_value = len(inp_data))

  with out_db.Session(commit = True) as s:
    for idx, dp in bar(enumerate(pool.imap_unordered(run_extractors, inp_data))):
      dp.id = idx
      s.add(dp)
      if idx+1 % 5000:
        s.commit()
    s.commit()
  pool.close()
  return

def update_tokenizer(sample: Sample, tokenizer) -> Sample:
  encoded = tokenizer.TokenizeString(sample.text)
  return Sample(
           **Sample.FromProto(0, model_pb2.Sample(
             train_step             = sample.train_step,
             text                   = sample.text,
             sample_indices         = sample.sample_indices,
             encoded_sample_indices = sample.sample_indices,
             original_input         = sample.original_input,
             sample_feed            = sample.sample_feed,
             encoded_text           = ','.join([str(x) for x in encoded]),
             sample_time_ms         = sample.sample_time_ms,
             feature_vector         = sample.feature_vector,
             num_tokens             = len(encoded),
             compile_status         = sample.compile_status,
             categorical_sampling   = int(sample.categorical_sampling),
             date_added             = sample.date_added.strftime("%m/%d/%Y, %H:%M:%S"),
            )
          )
        )

def modernize_clgen_tokenizer(db: SamplesDatabase, out_db: SamplesDatabase, tokenizer) -> None:
  """
  Re-run feature extractors to update old db.
  """
  pool = multiprocessing.Pool()
  inp_data = db.get_data
  bar = progressbar.ProgressBar(max_value = len(inp_data))

  f = functools.partial(update_tokenizer, tokenizer = tokenizer)

  with out_db.Session(commit = True) as s:
    for idx, dp in bar(enumerate(pool.imap_unordered(f, inp_data))):
      dp.id = idx
      s.add(dp)
      if idx+1 % 5000:
        s.commit()
    s.commit()
  pool.close()
  return

def ContentHash_worker(sample: Sample) -> typing.Tuple[str, Sample]:
  """
  Return new sample along with content hash of code.
  """
  try:
    return opencl.ContentHash(sample.text), sample
  except Exception as e:
    l.logger().warn(e)
    return None

def to_unique_samples(db: SamplesDatabase, out_db: SamplesDatabase) -> None:
  """
  Read input database, pass through deterministic re-writer and keep only unique samples.
  """
  pool     = multiprocessing.Pool()
  inp_data = [x for x in db.get_data]
  visited  = set()
  data     = []
  try:
    for sha, sample in tqdm.tqdm(pool.imap_unordered(ContentHash_worker, inp_data), total = len(inp_data), desc = "Unique-fy samples database"):
      if sha not in visited:
        visited.add(sha)
        data.append(sample)
  except Exception as e:
    l.logger().error(e)
    pool.terminate()
    raise e
  pool.close()
  with out_db.Session() as s:
    idx = 0
    for dp in tqdm.tqdm(data, total = len(data), desc = "Adding to DB"):
      new_dp = get_sample(dp)
      new_dp.id = idx
      idx += 1
      s.add(new_dp)
    s.commit()
  return

def initMain(*args, **kwargs):
  l.initLogger("samples_database")

  if not FLAGS.sample_merged_path:
    raise ValueError("Specify out path for merged database")

  out_path = pathlib.Path(FLAGS.sample_merged_path).absolute()
  if out_path.suffix != '.db':
    raise ValueError("sample_merged_path must end in a valid database name (.db extension): {}")
  out_path.parent.mkdir(exist_ok = True, parents = True)
  out_db = SamplesDatabase(url = "sqlite:///{}".format(str(out_path)), must_exist = False)

  db_paths = [pathlib.Path(p).absolute() for p in FLAGS.sample_mergeable_databases.replace(" ", "").split(",")]
  for p in db_paths:
    if not p.exists():
      raise FileNotFoundError(p)
  dbs = [SamplesDatabase(url = "sqlite:///{}".format(str(p)), must_exist = True) for p in db_paths]

  # tokenizer_path = pathlib.Path(FLAGS.tokenizer_path).resolve()
  # if not tokenizer_path.exists():
  #   raise FileNotFoundError(tokenizer_path)
  # tokenizer = tokenizers.TokenizerBase.FromFile(tokenizer_path)

  # merge_databases(dbs, out_db)
  # modernize_samples_db(dbs[0], out_db)
  # modernize_clgen_tokenizer(dbs[0], out_db, tokenizer)
  to_unique_samples(dbs[0], out_db)
  return

if __name__ == "__main__":
  app.run(initMain)
  sys.exit(0)
