"""BigQuery Dataset structures"""
import os
import typing
import pathlib
import progressbar
import humanize
from google.cloud import bigquery

from deeplearning.clgen.proto import github_miner_pb2
from deeplearning.clgen.corpuses.github import bigQuery_database
from eupy.native import logger as l

class Storage(object):

  @classmethod
  def FromArgs(cls,
               path: pathlib.Path,
               extension: str,
               data_format: int
               ) -> Storage:
    storage = {
      github_miner_pb2.GithubMiner.DataFormat.zip   : zipStorage,
      github_miner_pb2.GithubMiner.DataFormat.folder: fileStorage,
      github_miner_pb2.GithubMiner.DataFormat.sql   : dbStorage,
      github_miner_pb2.GithubMiner.DataFormat.bq    : bqStorage,
    }[data_format](path, extension)
    return

  def __init__(self,
               path: pathlib.Path,
               extension: str):
    self.cache_path = path
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
               extension: str
               ):
    super(zipStorage, self).__init__(path, extension)
    self.cached_content = []

  def save(self, content):
    self.cached_content.append(content)
    return

class fileStorage(Storage):
  def __init__(self,
               path: pathlib.Path,
               extension: str
               ):
    super(fileStorage, self).__init__(path, extension)
    self.file_counter = 0

  def save(self, content):
    with open(self.cache_path / "{}{}".format(self.counter, self.extension)) as f:
      f.write(content)
    self.file_counter += 1
    return

class dbStorage(Storage):
  def __init__(self,
               path: pathlib.Path,
               extension: str
               ):
    super(dbStorage, self).__init__(path, extension)
    self.db = bigQuery_database.bqDatabase("sqlite:///{}".format(self.cache_path / "bq_{}.db"))

  def save(self, content):
    with self.db.Session(commit = True) as session:
      contentfile = bigQuery_database.bqFile(
        **bigQuery_database.bqFile.FromArgs(self.db.count + 1, content)
      )
      session.add(contentfile)
    return

class bqStorage(Storage):
  def __init__(self,
               path: pathlib.Path,
               extension: str
               ):
    super(bqTableStorage, self).__init__(path, extension)

  def save(self, content):
    raise NotImplementedError
