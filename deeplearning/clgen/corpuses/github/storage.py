"""BigQuery Dataset structures"""
import os
import sys
import subprocess
import typing
import json
import shutil
import pathlib
import progressbar
import humanize
import functools
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
    return {
      github_pb2.GithubMiner.DataFormat.zip    : zipStorage,
      github_pb2.GithubMiner.DataFormat.folder : fileStorage,
      github_pb2.GithubMiner.DataFormat.json   : functools.partial(JSONStorage, with_zip = False),
      github_pb2.GithubMiner.DataFormat.jsonzip: functools.partial(JSONStorage, with_zip = True),
      github_pb2.GithubMiner.DataFormat.sql    : dbStorage,
      github_pb2.GithubMiner.DataFormat.bq     : bqStorage,
    }[data_format](path, name, extension)

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

  def __exit__(self, path, name, extension):
    return

  def save(self):
    raise NotImplementedError("Abstract Class")

class zipStorage(Storage):

  @property
  def repocount(self):
    return len(self.repos)

  @property
  def filecount(self):
    return self.file_count

  def __init__(self,
               path: pathlib.Path,
               name: str,
               extension: str
               ):
    super(zipStorage, self).__init__(path, name, extension)
    self.cached_content = []
    self.flush_counter  = 20000
    self.file_count     = 0
    self.data_file      = ""
    self.repos          = set()

  def __exit__(self, path, name, extension):
    self.zipFiles()
    return

  def save(self,
           contentfile: typing.Union[
                          bigQuery_database.bqData,
                          bigQuery_database.bqFile,
                          bigQuery_database.bqRepo
                        ]
           ) -> None:
    if isinstance(contentfile, bigQuery_database.bqFile):
      if contentfile.content is not None:
        self.cached_content.append(contentfile.content)
        self.file_count += 1
        if self.file_count % self.flush_counter == 0:
          self.zipFiles()
      else:
        raise ValueError("Wrong format of input contentfile.")
    elif isinstance(contentfile, bigQuery_database.bqData):
      self.data_file = "{}\n\n{}".format(contentfile.key, contentfile.value)
    elif isinstance(contentfile, bigQuery_database.bqRepo):
      entry = "{}, {}".format(contentfile.repo_name, contentfile.ref)
      if entry not in self.repos:
        self.repos.add(entry)
    return

  def zipFiles(self) -> None:
    tmp_root = pathlib.Path("/tmp/bqZipStorageTMP/corpus")
    tmp_root.mkdir(exist_ok = True, parents = True)
    for en, cf in enumerate(self.cached_content):
      with open(tmp_root / "{}{}".format(en+1, self.extension), 'w') as f:
        f.write(cf)
    with open(tmp_root / "data.txt", 'w') as f:
      f.write(self.data_file)
    with open(tmp_root / "repos_list.json", 'w') as f:
      json.dump(
        [
          {
            'repo_name': x.split(', ')[0],
            'ref': x.split(', ')[1]
          } for x in self.repos
        ],
        f,
        sort_keys = True,
        indent = 2
      )
    p = os.getcwd()
    os.chdir(tmp_root.parent)
    cmd = subprocess.Popen(
      "zip -qr -9 {} {}".format(self.cache_path / (self.name + ".zip"), tmp_root.name).split(),
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
    finally:
      os.chdir(p)
    return

class fileStorage(Storage):

  @property
  def repocount(self):
    return len(self.repos)

  @property
  def filecount(self):
    return self.file_count

  def __init__(self,
               path: pathlib.Path,
               name: str,
               extension: str
               ):
    super(fileStorage, self).__init__(path, name, extension)
    self.file_count = 0
    self.cache_path = self.cache_path / self.name
    (self.cache_path).mkdir(exist_ok = True)
    self.repos = set()

  def __exit__(self, path, name, extension) -> None:
    with open(self.cache_path / "repos_list.json", 'w') as f:
      json.dump(
        [
          {
            'repo_name': x.split(', ')[0],
            'ref': x.split(', ')[1]
          } for x in self.repos
        ],
        f,
        sort_keys = True,
        indent = 2
      )
    return

  def save(self,
           contentfile: typing.Union[
                          bigQuery_database.bqData,
                          bigQuery_database.bqFile,
                          bigQuery_database.bqRepo
                        ]
           ) -> None:
    if isinstance(contentfile, bigQuery_database.bqFile):
      if contentfile.content is not None:
        with open(self.cache_path / "{}{}".format(self.file_count, self.extension), 'w') as f:
          f.write(contentfile.content)
        self.file_count += 1
      else:
        raise ValueError("Wrong format of input contentfile.")
    elif isinstance(contentfile, bigQuery_database.bqData):
      with open(self.cache_path / "data.txt", 'w') as f:
        f.write("{}\n\n{}".format(contentfile.key, contentfile.value))
    elif isinstance(contentfile, bigQuery_database.bqRepo):
      entry = "{}, {}".format(contentfile.repo_name, contentfile.ref)
      if entry not in self.repos:
        self.repos.add(entry)
    return

class JSONStorage(Storage):

  @property
  def repocount(self):
    return len(self.repos)

  @property
  def filecount(self):
    return self.file_count

  def __init__(self,
               path: pathlib.Path,
               name: str,
               extension: str,
               with_zip: bool,
               ):
    super(JSONStorage, self).__init__(path, name, extension)
    self.cache_path = self.cache_path / self.name
    (self.cache_path).mkdir(exist_ok = True)

    self.jsonfile_count = 0
    self.file_count = 0

    self.data  = None
    self.repos = set()
    self.files = []

    return

  def __exit__(self, path, name, extension):
  
    if len(self.files) > 0:
      self._flush_json()
  
    with open(self.cache_path / "repos_list.json", 'w') as outf:
      json.dump(
        [
          {
            'repo_name': x.split(', ')[0],
            'ref': x.split(', ')[1]
          } for x in self.repos
        ],
        f,
        sort_keys = True,
        indent = 2
      )
    self.repos = set()
  
    with open(self.cache_path / "data.txt", 'w') as outf:
      outf.write(self.data)
    self.data = None

    return

  def save(self,
           contentfile: typing.Union[
                          bigQuery_database.bqData,
                          bigQuery_database.bqFile,
                          bigQuery_database.bqRepo
                        ]
           ) -> None:
    if isinstance(contentfile, bigQuery_database.bqData):
      self.data = "{}\n\n{}".format(contentfile.key, contentfile.value)
    elif isinstance(contentfile, bigQuery_database.bqRepo):
      entry = "{}, {}".format(contentfile.repo_name, contentfile.ref)
      if entry not in self.repos:
        self.repos.add(entry)
    else:
      self.files.append(contentfile.ToDict())
      self.file_count += 1
      if self.file_count % 500000:
        self._flush_json()
    return

  def _flush_json(self) -> None:

    filename = lambda ext: "{}.{}".format(self.jsonfile_count, ext)

    with open(self.cache_path / filename("json"), 'w') as outf:
      json.dump(outf, self.files)
      if self.with_zip:
        p = os.getcwd()
        os.chdir(self.cache_path)
        cmd = subprocess.Popen(
          "zip -qr -9 {} {}".format(filename("zip"), filename("json")).split(),
          stdout = sys.stdout,
          stderr = sys.stderr
        )
        try:
          out, err = cmd.communicate()
          os.remove(filename("json"))
          if err:
            raise OSError(err)
        except Exception as e:
          raise e
        finally:
          os.chdir(p)

    self.jsonfile_count += 1
    self.files = []
    return

class dbStorage(Storage):

  @property
  def repocount(self):
    return self.db.repo_count

  @property
  def filecount(self):
    return self.db.file_count

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
        exists = session.query(
          bigQuery_database.bqFile.sha256
        ).filter_by(sha256 = contentfile.sha256).scalar() is not None
        if not exists:
          session.add(contentfile)
    return

class bqStorage(Storage):

  @property
  def repocount(self):
    return 0 # TODO

  @property
  def filecount(self):
    return 0 # TODO

  def __init__(self,
               path: pathlib.Path,
               extension: str
               ):
    super(bqTableStorage, self).__init__(path, extension)

  def save(self,
           contentfile: bigQuery_database.bqFile
           ) -> None:
    raise NotImplementedError
