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
"""This file defines a database for pre-preprocessed content files."""
import contextlib
import datetime
import hashlib
import json
import multiprocessing
import os
import glob
import pathlib
import subprocess
import tempfile
import shutil
import time
import typing
import functools
import humanize
import tqdm
import sqlalchemy as sql
from sqlalchemy.ext import declarative
from sqlalchemy.sql import func

from deeplearning.clgen.preprocessors import preprocessors
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.proto import internal_pb2
from deeplearning.clgen.github import bigQuery_database as bqdb
from deeplearning.clgen.util import fs
from deeplearning.clgen.util import sqlutil
from deeplearning.clgen.util import environment
from deeplearning.clgen.util import distrib

from deeplearning.clgen.util import logging as l
from eupy.hermes import client

from absl import app, flags

FLAGS = flags.FLAGS

# flags.DEFINE_list(
#   "local_dir_file_ext",
#   None,
#   "If local_directory corpus has been selected and only specific file types are required,"
#   "pass a list of acceptable extensions here.",
# )

flags.DEFINE_boolean(
  "override_preprocessing",
  False,
  "Set to override incomplete pre-processing. Does not set DB value to 'done'"
)

flags.DEFINE_string(
  "preprocessed_databases",
  None,
  "Comma-separated list of paths for input preprocessed databases."
)

flags.DEFINE_string(
  "merged_preprocessed_database",
  None,
  "Path for merged output preprocessed database"
)

Base = declarative.declarative_base()


class Meta(Base):
  __tablename__ = "meta"

  key: str = sql.Column(sql.String(1024), primary_key=True)
  value: str = sql.Column(sql.String(1024), nullable=False)

class PreprocessedContentFile(Base):
  __tablename__ = "preprocessed_contentfiles"

  id: int = sql.Column(sql.Integer, primary_key=True)
  # Relative path of the input file within the content files.
  input_relpath: str = sql.Column(sql.String(3072), nullable=False, unique=False)
  # Checksum of the input file.
  input_sha256: str = sql.Column(sql.String(64), nullable=False)
  input_charcount = sql.Column(sql.Integer, nullable=False)
  input_linecount = sql.Column(sql.Integer, nullable=False)
  # Checksum of the preprocessed file.
  sha256: str = sql.Column(sql.String(64), nullable=False, index=True)
  charcount = sql.Column(sql.Integer, nullable=False)
  linecount = sql.Column(sql.Integer, nullable=False)
  text: str = sql.Column(
    sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable=False
  )
  # True if pre-processing succeeded, else False.
  preprocessing_succeeded: bool = sql.Column(sql.Boolean, nullable=False)
  # The number of milliseconds pre-preprocessing took.
  preprocess_time_ms: int = sql.Column(sql.Integer, nullable=False)
  # Pre-processing is parallelizable, so the actual wall time of pre-processing
  # may be much less than the sum of all preprocess_time_ms. This column
  # counts the effective number of "real" milliseconds during pre-processing
  # between the last pre-processed result and this result coming in. The idea
  # is that summing this column provides an accurate total of the actual time
  # spent pre-processing an entire corpus. Will be <= preprocess_time_ms.
  wall_time_ms: int = sql.Column(sql.Integer, nullable=False)
  date_added: datetime.datetime = sql.Column(
    sql.DateTime, nullable=False, default=datetime.datetime.utcnow
  )

  @classmethod
  def FromPreprocessedContentFile(
    cls,
    preprocessed_file: "PreprocessedContentFile",
    idx: int = None,
  ) -> "PreprocessedContentFile":
    """
    Replicate PreprocessedContentFile
    """
    return PreprocessedContentFile(
      id                      = idx if idx else preprocessed_file.id,
      input_relpath           = preprocessed_file.input_relpath,
      input_sha256            = preprocessed_file.input_sha256,
      input_charcount         = preprocessed_file.input_charcount,
      input_linecount         = preprocessed_file.input_linecount,
      sha256                  = preprocessed_file.sha256,
      charcount               = preprocessed_file.charcount,
      linecount               = preprocessed_file.linecount,
      text                    = preprocessed_file.text,
      preprocessing_succeeded = preprocessed_file.preprocessing_succeeded,
      preprocess_time_ms      = preprocessed_file.preprocess_time_ms,
      wall_time_ms            = preprocessed_file.wall_time_ms,
      date_added              = datetime.datetime.utcnow(),
    )

  @classmethod
  def FromContentFile(
    cls,
    contentfile_root: pathlib.Path,
    relpath: pathlib.Path,
    preprocessors_: typing.List[str],
  ) -> "PreprocessedContentFile":
    """Instantiate a PreprocessedContentFile."""
    start_time = time.time()
    input_text = ""
    preprocessing_succeeded = False
    try:
      with open(contentfile_root / relpath) as f:
        try:
          input_text = f.read()
        except UnicodeDecodeError:
          input_text = "/*corrupted file format*/"
        except UnicodeError:
          input_text = "/*corrupted file format*/"
        except Exception:
          input_text = "/*corrupted file format*/"
      text_generator = preprocessors.Preprocess(input_text, preprocessors_)
      # preprocessing_succeeded = True
    except Exception as e:
      raise("Unexpected exception: {}".format(e))

    end_time = time.time()
    preprocess_time_ms = int((end_time - start_time) * 1000)
    input_text_stripped = input_text.strip()
    return [ cls(
      input_relpath           = relpath,
      input_sha256            = GetFileSha256(contentfile_root / (relpath)),
      input_charcount         = len(input_text_stripped),
      input_linecount         = len(input_text_stripped.split("\n")),
      sha256                  = hashlib.sha256(text.encode("utf-8")).hexdigest(),
      charcount               = len(text),
      linecount               = len(text.split("\n")),
      text                    = text,
      preprocessing_succeeded = success,
      preprocess_time_ms      = preprocess_time_ms,
      wall_time_ms            = preprocess_time_ms,  # The outer-loop may change this.
      date_added              = datetime.datetime.utcnow(),
    ) for (text, success) in text_generator ]

  @classmethod
  def FromBQFile(
    cls,
    file: bqdb.bqMainFile,
    preprocessors_: typing.List[str],
  ) -> "PreprocessedContentFile":
    """Instantiate a PreprocessedContentFile."""
    start_time = time.time()
    preprocessing_succeeded = False
    try:
      if file.size is not None and int(str(file.size)) < (10**6):
        input_text = file.content
        text_generator = preprocessors.Preprocess(input_text, preprocessors_)
        # preprocessing_succeeded = True
      else:
        input_text = "<Redacted due to massive size>"
        text_generator = [("File is exceptionally large", 0)]
    except Exception as e:
      l.logger().warn("Unexpected exception: {}".format(e), ddp_nodes = True)
      return []

    end_time = time.time()
    preprocess_time_ms = int((end_time - start_time) * 1000)
    input_text_stripped = input_text.strip()
    return [ cls(
      input_relpath           = "main_files/{}".format(file.id),
      input_sha256            = file.id,
      input_charcount         = len(input_text_stripped),
      input_linecount         = len(input_text_stripped.split("\n")),
      sha256                  = hashlib.sha256(text.encode("utf-8")).hexdigest(),
      charcount               = len(text),
      linecount               = len(text.split("\n")),
      text                    = text,
      preprocessing_succeeded = success,
      preprocess_time_ms      = preprocess_time_ms,
      wall_time_ms            = preprocess_time_ms,  # The outer-loop may change this.
      date_added              = datetime.datetime.utcnow(),
    ) for (text, success) in text_generator ]

def PreprocessorWorker(job: str,
                       contentfile_root: pathlib.Path,
                       preprocessors: typing.List[str]
                       ) -> PreprocessedContentFile:
  """The inner loop of a parallelizable pre-processing job."""
  return PreprocessedContentFile.FromContentFile(
    contentfile_root, job, preprocessors
  )

def BQPreprocessorWorker(file: bqdb.bqMainFile, preprocessors: typing.List[str]) -> PreprocessedContentFile:
  """The inner loop of a parallelizable pre-processing job."""
  ret = PreprocessedContentFile.FromBQFile(file, preprocessors)
  return ret

class PreprocessedContentFiles(sqlutil.Database):
  """A database of pre-processed contentfiles."""

  def __init__(self, url: str, must_exist: bool = False, is_replica = False):
    if environment.WORLD_RANK == 0 or is_replica:
      super(PreprocessedContentFiles, self).__init__(
        url, Base, must_exist=must_exist
      )
    if environment.WORLD_SIZE > 1 and not is_replica:
      # Conduct engine connections to replicated preprocessed chunks.
      self.base_path = pathlib.Path(url.replace("sqlite:///", "")).resolve().parent
      hash_id = self.base_path.name
      try:
        tdir = pathlib.Path(FLAGS.local_filesystem).resolve() / hash_id / "node_preprocessed"
      except Exception:
        tdir = pathlib.Path("/tmp").resolve() / hash_id / "node_preprocessed"
      try:
        tdir.mkdir(parents = True, exist_ok = True)
      except Exception:
        pass
      self.replicated_path = tdir / "preprocessed_{}.db".format(environment.WORLD_RANK)
      self.replicated = PreprocessedContentFiles(
        url = "sqlite:///{}".format(str(self.replicated_path)),
        must_exist = must_exist,
        is_replica = True
      )
      distrib.barrier()
    return

  def Create(self, config: corpus_pb2.Corpus):
    """
    Create preprocessed database of raw corpus.
    """
    ## Check if main preprocessed.db is done.
    if environment.WORLD_SIZE > 1:
      if environment.WORLD_RANK == 0:
        ## If done, broadcast true message and return.
        with self.Session() as session:
          status = self.IsDone(session)
          _ = distrib.broadcast(str(status))
          if status:
            return
      else:
        status = distrib.broadcast()
        if status == "True":
          return
        if status != "False":
          raise OSError("Broken distributed message: '{}'".format(status))

    ## For DDP only: If main DB not done, preprocess the replicas here.
    sessmaker = self.Session if environment.WORLD_SIZE == 1 else self.replicated.Session
    with sessmaker() as session:
      if not self.IsDone(session):
        self.Import(session, config)
        self.SetDone(session)
        session.commit()

    ## For DDP only: Merge replicas into main DB.
    if environment.WORLD_SIZE > 1:
      self.MergeReplicas()

      # Logging output.
    #   num_input_files = session.query(PreprocessedContentFile).count()
    #   num_files = (
    #     session.query(PreprocessedContentFile)
    #     .filter(PreprocessedContentFile.preprocessing_succeeded == True)
    #     .count()
    #   )
    #   input_chars, input_lines, total_walltime, total_time, = session.query(
    #     func.sum(PreprocessedContentFile.charcount),
    #     func.sum(PreprocessedContentFile.linecount),
    #     func.sum(PreprocessedContentFile.wall_time_ms),
    #     func.sum(PreprocessedContentFile.preprocess_time_ms),
    #   ).first()
    #   char_count, line_count = (
    #     session.query(
    #       func.sum(PreprocessedContentFile.charcount),
    #       func.sum(PreprocessedContentFile.linecount),
    #     )
    #     .filter(PreprocessedContentFile.preprocessing_succeeded == True)
    #     .first()
    #   )
    # set_mail = "Content files: {} chars, {} lines, {} files.\n".format(
    #           humanize.intcomma(input_chars),
    #           humanize.intcomma(input_lines),
    #           humanize.intcomma(num_input_files),
    #         )
    # l.logger().info(
    #   "Content files: {} chars, {} lines, {} files.".format(
    #           humanize.intcomma(input_chars),
    #           humanize.intcomma(input_lines),
    #           humanize.intcomma(num_input_files),
    #         ), mail_level = 4
    # )
    # set_mail += "Pre-processed {} files in {} ({:.2f}x speedup).\n".format(
    #           humanize.intcomma(num_input_files),
    #           humanize.naturaldelta((total_walltime or 0) / 1000),
    #           (total_time or 1) / (total_walltime or 1),
    #       )
    # l.logger().info(
    #   "Pre-processed {} files in {} ({:.2f}x speedup).".format(
    #           humanize.intcomma(num_input_files),
    #           humanize.naturaldelta((total_walltime or 0) / 1000),
    #           (total_time or 1) / (total_walltime or 1),
    #       ), mail_level = 4
    # )
    # set_mail += "Pre-processing discard rate: {:.1f}% ({} files).\n".format(
    #           (1 - (num_files / max(num_input_files, 1))) * 100,
    #           humanize.intcomma(num_input_files - num_files),
    #       )
    # l.logger().info(
    #   "Pre-processing discard rate: {:.1f}% ({} files).".format(
    #           (1 - (num_files / max(num_input_files, 1))) * 100,
    #           humanize.intcomma(num_input_files - num_files),
    #       ), mail_level = 4
    # )
    # set_mail += "Pre-processed corpus: {} chars, {} lines, {} files.\n".format(
    #           humanize.intcomma(char_count),
    #           humanize.intcomma(line_count),
    #           humanize.intcomma(num_files),
    #       )
    # l.logger().info(
    #   "Pre-processed corpus: {} chars, {} lines, {} files.".format(
    #           humanize.intcomma(char_count),
    #           humanize.intcomma(line_count),
    #           humanize.intcomma(num_files),
    #       ), mail_level = 4
    # )
    # if FLAGS.notify_me:
    #   client.getClient().send_message("clgen:preprocessed", set_mail)
    return

  def IsDone(self, session: sqlutil.Session):
    if session.query(Meta).filter(Meta.key == "done").first():
      return True
    elif FLAGS.override_preprocessing:
      l.logger().warn("Overriding incomplete pre-processed DB.")
      return True
    else:
      return False

  def SetDone(self, session: sqlutil.Session):
    session.add(Meta(key="done", value="yes"))
    return

  def Import(self, session: sqlutil.Session, config: corpus_pb2.Corpus) -> None:
    with self.GetContentFileRoot(config) as contentfile_root:
      if not config.HasField("bq_database"):
        if environment.WORLD_RANK == 0:
          relpaths = set(self.GetImportRelpaths(contentfile_root))
          done = set(
            [x[0] for x in session.query(PreprocessedContentFile.input_relpath)]
          )
          todo = relpaths - done
          l.logger().info(
            "Preprocessing {} of {} content files".format(
                    humanize.intcomma(len(todo)),
                    humanize.intcomma(len(relpaths)),
                )
          )
          chunk_size = 100000
          jobs, total = [], 0
          for idx, t in enumerate(todo):
            if idx % chunk_size == 0:
              jobs.append([t])
            else:
              jobs[-1].append(t)
            total += 1
          bar = tqdm.tqdm(total = total, desc = "Preprocessing", leave = True)
          c = 0
          last_commit = time.time()
          wall_time_start = time.time()
          for job_chunk in jobs:
            try:
              pool = multiprocessing.Pool()
              for preprocessed_list in pool.imap_unordered(
                                         functools.partial(
                                           PreprocessorWorker,
                                           contentfile_root = contentfile_root,
                                           preprocessors = list(config.preprocessor)
                                         ),
                                         job_chunk
                                        ):
                for preprocessed_cf in preprocessed_list:
                  wall_time_end = time.time()
                  preprocessed_cf.wall_time_ms = int(
                    (wall_time_end - wall_time_start) * 1000
                  )
                  wall_time_start = wall_time_end
                  session.add(preprocessed_cf)
                  if wall_time_end - last_commit > 10:
                    session.commit()
                    last_commit = wall_time_end
                c += 1
                bar.update(1)
              pool.close()
            except KeyboardInterrupt as e:
              pool.terminate()
              raise e
            except Exception as e:
              pool.terminate()
              raise e
          session.commit()
      else:
        db  = bqdb.bqDatabase("sqlite:///{}".format(contentfile_root), must_exist = True)
        total = db.mainfile_count                        # Total number of files in BQ database.
        total_per_node = total // environment.WORLD_SIZE # In distributed nodes, this is the total files to be processed per node.
        if total == 0:
          raise ValueError("Input BQ database {} is empty!".format(contentfile_root))

        # Set of IDs that have been completed.
        done = set(
          [x[0].replace("main_files/", "") for x in session.query(PreprocessedContentFile.input_relpath)]
        )

        chunk, idx = min(total_per_node, 10**8), environment.WORLD_RANK * total_per_node
        limit = (environment.WORLD_RANK + 1) * total_per_node + (total % total_per_node if environment.WORLD_RANK == environment.WORLD_SIZE - 1 else 0)

        if environment.WORLD_SIZE > 1:
          bar = distrib.ProgressBar(total = total, offset = idx, desc = "Preprocessing DB")
        else:
          bar = tqdm.tqdm(total = total, desc = "Preprocessing DB", leave = True)

        last_commit     = time.time()
        wall_time_start = time.time()

        pool = multiprocessing.Pool(maxtasksperchild = 8192)
        try:
          while idx < limit:
            chunk = min(chunk, limit - idx) # This is equivalent to l447/l448 but needed for last node that gets a bit more.
            batch = db.main_files_batch(chunk, idx, exclude_id = done)
            idx += chunk - len(batch) # This difference will be the number of already done files.
            for preprocessed_list in pool.imap_unordered(
                                      functools.partial(
                                        BQPreprocessorWorker,
                                        preprocessors = list(config.preprocessor)
                                    ), batch):
              for preprocessed_cf in preprocessed_list:
                wall_time_end = time.time()
                preprocessed_cf.wall_time_ms = int(
                  (wall_time_end - wall_time_start) * 1000
                )
                wall_time_start = wall_time_end
                session.add(preprocessed_cf)
                if wall_time_end - last_commit > 1000:
                  session.commit()
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
        if environment.WORLD_SIZE > 1:
          bar.finalize(idx)
    return

  def MergeReplicas(self) -> None:
    """
    When distributed nodes work for the same preprocessed DB
    this function moves finalized preprocessed chunks back into the AFS
    and master node merges them into the final preprocessed.db
    """
    shutil.copy(
      self.replicated_path, self.base_path / "preprocessed_{}.db".format(environment.WORLD_RANK)
    )
    distrib.barrier()
    if environment.WORLD_RANK == 0:
      db_chunks = glob.glob(str(self.base_path / "preprocessed_*.db"))
      dbs = [PreprocessedContentFiles(url = "sqlite:///{}".format(p), must_exist = True, is_replica = True) for p in db_chunks]
      merge_db(dbs, self)
      for p in db_chunks:
        os.remove(p)
    distrib.barrier()
    # Once merging has been complete, cleanup the mess left at local clusters' filesystems.
    if (self.replicated_path.parent / "bq_database_replica_{}.db".format(environment.WORLD_RANK)).exists():
      os.remove(self.replicated_path.parent / "bq_database_replica_{}.db".format(environment.WORLD_RANK))
    else:
      l.logger().warn("I didn't find my local BQ replica at {}".format(self.replicated_path.parent / "bq_database_replica_{}.db".format(environment.WORLD_RANK)), ddp_nodes = True)
    distrib.barrier()
    return

  @contextlib.contextmanager
  def GetContentFileRoot(self, config: corpus_pb2.Corpus) -> pathlib.Path:
    """Get the path of the directory containing content files.

    If the corpus is a local directory, this simply returns the path. Otherwise,
    this method creates a temporary copy of the files which can be used within
    the scope of this context.

    Args:
      config: The corpus config proto.

    Returns:
      The path of a directory containing content files.
    """
    if config.HasField("local_directory"):
      yield pathlib.Path(ExpandConfigPath(config.local_directory))
    elif config.HasField("local_tar_archive"):
      with tempfile.TemporaryDirectory(prefix="clgen_corpus_", dir = FLAGS.local_filesystem) as d:
        l.logger().info("Unpacking {}...".format(ExpandConfigPath(config.local_tar_archive).name))
        start_time = time.time()
        if environment.WORLD_RANK == 0:
          cmd = [
            "tar",
            "-xf",
            str(ExpandConfigPath(config.local_tar_archive)),
            "-C",
            d,
          ]
          subprocess.check_call(cmd)
        distrib.barrier()
        l.logger().info(
          "Unpacked {} in {} ms".format(
                  ExpandConfigPath(config.local_tar_archive).name,
                  humanize.intcomma(int((time.time() - start_time) * 1000)),
              )
        )
        yield pathlib.Path(d)
    elif config.HasField("bq_database"):
      input_bq = pathlib.Path(ExpandConfigPath(config.bq_database))
      if environment.WORLD_SIZE > 1:
        target_bq = self.replicated_path.parent / "bq_database_replica_{}.db".format(environment.WORLD_RANK)
        if not target_bq.exists():
          shutil.copy(input_bq, target_bq)
        yield target_bq
      else:
        yield input_bq
    else:
      raise NotImplementedError

  @property
  def size(self) -> int:
    """Return the total number of files in the pre-processed corpus.

    This excludes contentfiles which did not pre-process successfully.
    """
    with self.Session() as session:
      return (
        session.query(PreprocessedContentFile)
        .filter(PreprocessedContentFile.preprocessing_succeeded == True)
        .count()
      )

  @property
  def input_size(self) -> int:
    """Return the total number of files in the pre-processed corpus.

    This *includes* contentfiles which did not pre-process successfully.
    """
    with self.Session() as session:
      return session.query(PreprocessedContentFile).count()

  @property
  def char_count(self) -> int:
    """Get the total number of characters in the pre-processed corpus.

    This excludes contentfiles which did not pre-process successfully.
    """
    with self.Session() as session:
      return (
        session.query(func.sum(PreprocessedContentFile.charcount))
        .filter(PreprocessedContentFile.preprocessing_succeeded == True)
        .scalar()
      )

  @property
  def line_count(self) -> int:
    """Get the total number of lines in the pre-processed corpus.

    This excludes contentfiles which did not pre-process successfully.
    """
    with self.Session() as session:
      return (
        session.query(func.sum(PreprocessedContentFile.linecount))
        .filter(PreprocessedContentFile.preprocessing_succeeded == True)
        .scalar()
      )

  @property
  def input_char_count(self) -> int:
    """Get the total number of characters in the input content files."""
    with self.Session() as session:
      return session.query(
        func.sum(PreprocessedContentFile.input_charcount)
      ).scalar()

  @property
  def input_line_count(self) -> int:
    """Get the total number of characters in the input content files."""
    with self.Session() as session:
      return session.query(
        func.sum(PreprocessedContentFile.input_linecount)
      ).scalar()

  def GetImportRelpaths(
    self, contentfile_root: pathlib.Path
  ) -> typing.List[str]:
    """Get relative paths to all files in the content files directory.

    Args:
      contentfile_root: The root of the content files directory.

    Returns:
      A list of paths relative to the content files root.

    Raises:
      EmptyCorpusException: If the content files directory is empty.
    """

    find_output = []
    queue       = [contentfile_root]
    cpus        = os.cpu_count()
    multi_thr   = min(cpus**2, 1600)

    while queue:
      if len(queue) >= multi_thr:
        break
      cur = queue.pop(0)
      try:
        for f in cur.iterdir():
          if f.is_symlink():
            continue
          elif f.is_file():
            if f.suffix in {'.c', '.cl'}:
              find_output.append(str(f))
          elif f.is_dir():
            queue.append(f)
          else:
            continue
      except PermissionError:
        pass
      except NotADirectoryError:
        pass
      except FileNotFoundError:
        pass
      except OSError:
        pass

    if queue:
      p = multiprocessing.Pool(cpus)
      for batch in p.imap_unordered(path_worker, queue):
        find_output += batch
    return find_output

def path_worker(base_path) -> typing.List[str]:
  paths = []
  queue = [base_path]
  while queue:
    cur = queue.pop(0)
    try:
      for f in cur.iterdir():
        if f.is_symlink():
          continue
        elif f.is_file():
          if f.suffix in {'.c', '.cl'}:
            paths.append(str(f))
        elif f.is_dir():
          queue.append(f)
        else:
          continue
    except PermissionError:
      pass
    except NotADirectoryError:
      pass
    except FileNotFoundError:
      pass
    except OSError:
      pass
  return paths

def ExpandConfigPath(path: str) -> pathlib.Path:
  return pathlib.Path(os.path.expandvars(path)).expanduser().absolute()


def GetFileSha256(path: pathlib.Path) -> str:
  with open(path, "rb") as f:
    return hashlib.sha256(f.read()).hexdigest()

def merge_db(dbs: typing.List[PreprocessedContentFiles], out_db: typing.List[PreprocessedContentFiles]) -> None:
  """
  Collect data from a list of preprocessed databases and merge them.
  """
  for db in dbs:
    l.logger().info("Loading {}...".format(db.url))
    pkey = out_db.input_size
    with db.Session() as ses:
      data = ses.query(PreprocessedContentFile).all()
    with out_db.Session() as ses:
      bar = tqdm.tqdm(total = len(data), desc = "DB Merging", leave = True)
      for df in data:
        ses.add(PreprocessedContentFile.FromPreprocessedContentFile(df, idx = pkey + df.id))
        bar.update(1)
      ses.commit()
  with out_db.Session() as ses:
    out_db.SetDone(ses)
    ses.commit()
  return

def compiling_text_to_huggingface_json(db_path: str, json_out: str) -> None:
  """
  Converts preprocessed.db into json file with compiling samples
  that can be read by huggingface Datasets.
  """
  out_data = []
  p = pathlib.Path(db_path).resolve()
  out_p = pathlib.Path(json_out).resolve()
  if not p.exists():
    raise FileNotFoundError("{} does not exist!".format(db_path))
  db = PreprocessedContentFiles(url = "sqlite:///{}".format(str(p)), must_exist = True)
  with db.Session() as s:
    data = [x for x in s.query(PreprocessedContentFile).all() if x.preprocessing_succeeded == True]
    with open(str(out_p), 'w') as outf:
      for dp in data:
        json_el = json.dumps({
            'file_name'         : "{}.cl".format(dp.sha256),
            'github_id'         : str(dp.sha256),
            'id'                : dp.id,
            'license'           : "mit",
            'path'              : dp.input_relpath,
            'repo_and_filename' : "",
            'repo_name'         : "",
            'signature'         : "",
            'size'              : dp.charcount,
            'text'              : dp.text
          })
        outf.write(json_el)
        outf.write('\n')
  return

def initMain(*args, **kwargs):
  """
  Setup module's operations.
  """
  l.initLogger(name = "bigQuery_database")

  if not FLAGS.preprocessed_databases:
    raise ValueError("Please input preprocessed databases to merge as a comma separated list.")
  db_paths = [pathlib.Path(p).absolute() for p in FLAGS.preprocessed_databases.replace(" ", "").split(",")]
  for p in db_paths:
    if not p.exists():
      raise FileNotFoundError(p)
  dbs = [PreprocessedContentFiles(url = "sqlite:///{}".format(str(p)), must_exist = True) for p in db_paths]

  if not FLAGS.merged_preprocessed_database:
    raise ValueError("You must set a path for merged_preprocessed_database")
  out_db_path = pathlib.Path(FLAGS.merged_preprocessed_database).resolve()
  out_db_path.parent.mkdir(exist_ok = True, parents = True)
  out_db = PreprocessedContentFiles(url = "sqlite:///{}".format(str(out_db_path)), must_exist = False)
  merge_db(dbs, out_db)
  return

if __name__ == "__main__":
  app.run(initMain)
  sys.exit(0)
