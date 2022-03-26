"""
Module handling Structs Database.

This database captures and handles the struct/class/union/enum
dependencies found during raw corpus preprocessing.
"""
import datetime
import sqlite3
import pathlib
import typing
import multiprocessing
import hashlib
# import clang.cindex

import sqlalchemy as sql
from sqlalchemy.ext import declarative

from deeplearning.clgen.util import environment
from deeplearning.clgen.util import distrib
from deeplearning.clgen.util import crypto
from deeplearning.clgen.util import logging as l
from deeplearning.clgen.preprocessors import c

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
  "datatypes_bq_db",
  None,
  "Set path for BQ database to parse datatypes."
)

flags.DEFINE_string(
  "datatypes_db",
  None,
  "Set path for output datatypes database."
)

Base = declarative.declarative_base()

class Data(Base):
  __tablename__ = "data"
  """
  DB Table holding struct meta-data and stats.
  """
  key     : str = sql.Column(sql.String(1024), primary_key=True)
  results : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)

class Struct(Base, sqlutil.ProtoBackedMixin):
  """
  A database row representing a parsed struct.
  """
  __tablename__    = "structs"
  # entry id.
  id            : int = sql.Column(sql.Integer, primary_key = True)
  # Relative path of original bq entry.
  input_relpath : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Sha256 of input bq entry
  input_sha256  : str = sql.Column(sql.String(64), nullable = False)
  # unique, indexable content has of struct.
  sha256        : str = sql.Column(sql.String(64), nullable = False, index = True)
  # Struct contents.
  contents      : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Struct name.
  name          : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Struct fields.
  fields        : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Number of fields a struct has.
  num_fields    : int = sql.Column(sql.Integer, nullable = False)
  # Repo name where struct was found.
  repo_name     : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Repo ref.
  ref           : str = sql.Column(sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable = False)
  # Wall time
  wall_time_ms  : int = sql.Column(Integer, nullable = False)
  # Date
  date_added : datetime.datetime = sql.Column(sql.DateTime, nullable=False)

  @classmethod
  def FromArgs(cls,
               id         : int,
               contents   : str,
               num_fields : int,
               repo_name  : str,
               ref        : str,
               ) -> 'Struct':
    return Struct

class DatatypeDB(sqlutil.Database):
  """A database directory of C/OpenCL composite types and functions."""
  @classmethod
  def FromBQ(entry: bqdb.bqMainFile):
    start_time = time.time()
    preprocessing_succeeded = False
    try:
      input_text = entry.content
      structs = preprocessors.Preprocess(input_text, preprocessors_)
      # preprocessing_succeeded = True
    except Exception as e:
      raise("Unexpected exception: {}".format(e))

    end_time = time.time()
    preprocess_time_ms = int((end_time - start_time) * 1000)
    input_text_stripped = input_text.strip()
    return [ Struct(
      input_relpath           = "main_files/{}".format(entry.id),
      input_sha256            = entry.id,
      sha256                  = hashlib.sha256(struct['text'].encode("utf-8")).hexdigest(),
      contents                = struct['text'],
      name                    = struct['name'],
      fields                  = struct['fields'],
      num_fields              = len(struct['fields']),
      preprocessing_succeeded = success,
      repo_name               = entry.repo_name,
      ref                     = entry.ref,
      wall_time_ms            = preprocess_time_ms,
      date_added              = datetime.datetime.utcnow(),
    ) for (struct, success) in structs]
    return

  def __init__(self, url: str, must_exist: bool = False):
    super(DatatypeDB, self).__init__(url, Base, must_exist = must_exist)
    return

def CollectStructsBQ(db, session):
  total = db.mainfile_count                        # Total number of files in BQ database.
  total_per_node = total // environment.WORLD_SIZE # In distributed nodes, this is the total files to be processed per node.
  if total == 0:
    raise ValueError("Input BQ database {} is empty!".format(contentfile_root))

  # Set of IDs that have been completed.
  done = set(
    [x[0].replace("main_files/", "") for x in session.query(Struct.input_relpath)]
  )

  chunk, idx = min(total_per_node, 100000), environment.WORLD_RANK * total_per_node
  limit = (environment.WORLD_RANK + 1) * total_per_node + (total % total_per_node if environment.WORLD_RANK == environment.WORLD_SIZE - 1 else 0)

  if environment.WORLD_SIZE > 1:
    bar = distrib.ProgressBar(total = total, offset = idx, decs = "Preprocessing DB")
  else:
    bar = tqdm.tqdm(total = total, desc = "Preprocessing DB", leave = True)

  last_commit     = time.time()
  wall_time_start = time.time()

  while idx < limit:
    try:
      chunk = min(chunk, limit - idx)
      batch = db.main_files_batch(chunk, idx, exclude_id = done)
      idx += chunk - len(batch) # This difference will be the number of already done files.
      flush_queue = set()
      pool = multiprocessing.Pool()
      for structs_list in pool.imap_unordered(DatatypeDB.FromBQ, batch):
        for struct in structs_list:
          wall_time_end = time.time()
          struct.wall_time_ms = int(
            (wall_time_end - wall_time_start) * 1000
          )
          wall_time_start = wall_time_end
          exists = session.query(Struct).filter_by(Struct.sha256 == struct.sha256).first()
          if not exists and struct.sha256 not in flush_queue:
            session.add(struct)
            flush_queue.add(struct.sha256)
          if wall_time_end - last_commit > 10:
            session.commit()
            flush_queue = set()
            last_commit = wall_time_end
        idx += 1
        bar.update(idx - bar.n)
      pool.close()
    except KeyboardInterrupt as e:
      pool.terminate()
      raise e
    except Exception as e:
      l.logger().error(e, ddp_nodes = True)
      pool.terminate()
      raise e
  session.commit()
  flush_queue = set()
  if environment.WORLD_SIZE > 1:
    bar.finalize(idx)

def main():
  try:
    tdir = FLAGS.local_filesystem
  except Exception:
    tdir = None

  if FLAGS.datatypes_bq_db is None:
    raise ValueError("Set path for BQ database.")
  if FLAGS.datatypes_db is None:
    raise ValueError("Set path for output datatypes database.")

  with tempfile.TemporaryDirectory(prefix="locks_", dir = tdir) as d:
    distrib.init(str(d))
    db  = bqdb.bqDatabase("sqlite:///{}".format(str(pathlib.Path(FLAGS.datatypes_bq_db).resolve())), must_exist = True)
    structs_db = DatatypeDB("sqlite:///{}".format(str(pathlib.Path(FLAGS.datatypes_db).resolve())), must_exist = False)
    with structs_db.get_session(commit = True) as session:
      CollectStructsBQ(db, session)
  return

if __name__ == "__main__":
  app.run(main)
  exit()
