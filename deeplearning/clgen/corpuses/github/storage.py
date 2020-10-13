"""BigQuery Dataset structures"""
import os
import typing
import shutil
import pathlib
import progressbar
import humanize
from google.cloud import bigquery

from deeplearning.clgen.proto import github_pb2
from deeplearning.clgen.corpuses.github import bigQuery_database
from eupy.native import logger as l

class Storage(object):

  @classmethod
  def FromArgs(cls,
               path: pathlib.Path,
               name: str,
               extension: str,
               data_format: int
               ):
    storage = {
      github_pb2.GithubMiner.DataFormat.zip   : zipStorage,
      github_pb2.GithubMiner.DataFormat.folder: fileStorage,
      github_pb2.GithubMiner.DataFormat.sql   : dbStorage,
      github_pb2.GithubMiner.DataFormat.bq    : bqStorage,
    }[data_format](path, name, extension)
    return

  def __init__(self,
               path: pathlib.Path,
               name: str,
               extension: str):
    self.cache_path = path
    self.cache_path.mkdir(exist_ok = True)
    self.name       = name
    self.extension  = extension
    return

  def __enter__(self):
    return self

  def __exit__(self):
    return

  def save(self):
    raise NotImplementedError("Abstract Class")

class zipStorage(Storage):
  def __init__(self,
               path: pathlib.Path,
               name: str,
               extension: str
               ):
    super(zipStorage, self).__init__(path, name, extension)
    self.cached_content = []
    self.flush_counter  = 20000
    self.current_file   = 0

  def save(self,
           contentfile: bigQuery_database.bqFile
           ) -> None:
    if not isinstance(contentfile, bigQuery_database.bqFile):
      return
    if contentfile.content is not None:
      self.cached_content.append(contentfile.content)
      self.current_file += 1
      if self.current_file >= self.flush_counter:
        self.zipFiles()
        self.current_file = 0
    else:
      raise ValueError("Wrong format of input contentfile.")
    return

  def zipFiles(self) -> None:
    tmp_root = pathlib.Path("/tmp/bqZipStorageTMP/corpus").mkdir(exist_ok = True)
    for en, cf in enumerate(self.cached_content):
      with open(tmp_root / "{}.{}".format(en+1, self.extension), 'w') as f:
        f.write(cf)
    cmd = subprocess.Popen(
      "zip -r -9 {} {}".format(self.cache_path / (self.name + ".zip"), tmp_root).split(),
      stdout = sys.stdout,
      stderr = sys.stderr
    )
    try:
      out, err = cmd.communicate()
      if err:
        raise OSError(err)
      shutil.rmtree(tmp_root)
    except Exception as e:
      raise e
    return

class fileStorage(Storage):
  def __init__(self,
               path: pathlib.Path,
               name: str,
               extension: str
               ):
    super(fileStorage, self).__init__(path, name, extension)
    self.file_counter = 0
    (self.cache_path / self.name).mkdir(exist_ok = True)

  def save(self,
           contentfile: bigQuery_database.bqFile
           ) -> None:
    if not isinstance(contentfile, bigQuery_database.bqFile):
      return
    if contentfile.content is not None:
      with open(self.cache_path / self.name / "{}{}".format(self.counter, self.extension)) as f:
        f.write(contentfile.content)
      self.file_counter += 1
    else:
      raise ValueError("Wrong format of input contentfile.")
    return

class dbStorage(Storage):
  def __init__(self,
               path: pathlib.Path,
               name: str,
               extension: str
               ):
    super(dbStorage, self).__init__(path, name, extension)
    self.db = bigQuery_database.bqDatabase("sqlite:///{}".format(self.cache_path / (self.name + ".db")))

  def save(self,
           contentfile: typing.Union[
                          bigQuery_database.bqData,
                          bigQuery_database.bqFile,
                          bigQuery_database.bqRepo
                        ]
           ) -> None:
    with self.db.Session(commit = True) as session:
      if isinstance(contentfile, bigQuery_database.bqData):
        exists = session.query(
          bigQuery_database.bqData.key
        ).filter_by(key = contentfile.key).scalar() is not None
        if exists:
          entry = session.query(
            bigQuery_database.bqData
          ).filter_by(key = contentfile.key).first()
          entry.value = contentfile.value
        else:
          session.add(contentfile)
      elif isinstance(contentfile, bigQuery_database.bqRepo):
        exists = session.query(
          bigQuery_database.bqRepo.repo_name,
          bigQuery_database.bqRepo.ref
        ).filter_by(
          repo_name = contentfile.repo_name, ref = contentfile.ref
        ).scalar() is not None
        if not exists:
          session.add(contentfile)
      else:
        session.add(contentfile)
    return

class bqStorage(Storage):
  def __init__(self,
               path: pathlib.Path,
               extension: str
               ):
    super(bqTableStorage, self).__init__(path, extension)

  def save(self,
           contentfile: bigQuery_database.bqFile
           ) -> None:
    raise NotImplementedError
