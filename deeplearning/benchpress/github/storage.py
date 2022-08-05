# coding=utf-8
# Copyright 2022 Foivos Tsimpourlas and Chris Cummins.
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

from deeplearning.benchpress.proto import github_pb2
from deeplearning.benchpress.github import bigQuery_database as bqdb
from deeplearning.benchpress.util import logging as l

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

  def flush(self):
    pass

class zipStorage(Storage):

  @property
  def repocount(self):
    return len(self.repos)

  @property
  def filecount(self):
    return self.file_count

  @property
  def loadRepos(self):
    raise NotImplementedError("Open ZIP files and read repos_list.json")

  def __init__(self,
               path: pathlib.Path,
               name: str,
               extension: str
               ):
    super(zipStorage, self).__init__(path, name, extension)
    self.cached_content = []
    self.flush_counter  = 20000
    self.file_count     = 0
    self.repos          = self.loadRepos
    self.data_file      = ""
    l.logger().info("Set up ZIP storage in {}".format(self.cache_path))

  def __exit__(self, path, name, extension):
    self.zipFiles()
    return

  def save(self,
           contentfile: typing.Union[
                          bqdb.bqData,
                          bqdb.bqFile,
                          bqdb.bqRepo
                        ]
           ) -> None:
    if isinstance(contentfile, bqdb.bqFile):
      self.cached_content.append(contentfile.content)
      self.file_count += 1
      if self.file_count % self.flush_counter == 0:
        self.zipFiles()
      self.repos.add((contentfile.repo_name, contentfile.ref))
    elif isinstance(contentfile, bqdb.bqData):
      self.data_file = "{}\n\n{}".format(contentfile.key, contentfile.value)
    elif isinstance(contentfile, bqdb.bqRepo):
      self.repos.add((contentfile.repo_name, contentfile.ref))
    return

  def zipFiles(self) -> None:
    tmp_root = pathlib.Path("/tmp/bqZipStorageTMP/corpus")
    tmp_root.mkdir(exist_ok = True, parents = True)
    for cf in self.cached_content:
      with open(tmp_root / pathlib.Path(cf.path).name, 'w') as f:
        f.write(cf)
    with open(tmp_root / "data.txt", 'w') as f:
      f.write(self.data_file)
    with open(tmp_root / "repos_list.json", 'w') as f:
      json.dump(
        [
          {
            'repo_name': rn,
            'ref': rf
          } for rn, rf in self.repos
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

  @property
  def loadRepos(self):
    if (self.cache_path / "repos_list.json").exists():
      with open(self.cache_path / "repos_list.json", 'r') as f:
        repos = json.load(f)
        return [(repo['repo_name'], repo['ref']) for repo in repos]
    else:
      return set()

  def __init__(self,
               path: pathlib.Path,
               name: str,
               extension: str
               ):
    super(fileStorage, self).__init__(path, name, extension)
    self.cache_path = self.cache_path / self.name
    (self.cache_path).mkdir(exist_ok = True)
    self.repos = self.loadRepos
    l.logger().info("Set up folder storage in {}".format(self.cache_path))

  def __exit__(self, path, name, extension) -> None:
    with open(self.cache_path / "repos_list.json", 'w') as f:
      json.dump(
        [
          {
            'repo_name': rn,
            'ref': rf
          } for rf, rf in self.repos
        ],
        f,
        sort_keys = True,
        indent = 2
      )
    return

  def save(self,
           contentfile: typing.Union[
                          bqdb.bqData,
                          bqdb.bqFile,
                          bqdb.bqRepo
                        ]
           ) -> None:
    if isinstance(contentfile, bqdb.bqFile):
      with open(self.cache_path / pathlib.Path(contentfile.path).name, 'w') as f:
        f.write(contentfile.content)
      self.repos.add((contentfile.repo_name, contentfile.ref))
    elif isinstance(contentfile, bqdb.bqData):
      with open(self.cache_path / "data.txt", 'w') as f:
        f.write("{}\n\n{}".format(contentfile.key, contentfile.value))
    elif isinstance(contentfile, bqdb.bqRepo):
      self.repos.add((contentfile.repo_name, contentfile.ref))
    return

class JSONStorage(Storage):

  @property
  def repocount(self):
    return len(self.repos)

  @property
  def filecount(self):
    return self.file_count

  @property
  def loadRepos(self):
    if (self.cache_path / "repos_list.json").exists():
      with open(self.cache_path / "repos_list.json", 'r') as f:
        repos = json.load(f)
        return [(repo['repo_name'], repo['ref']) for repo in repos]
    else:
      return set()

  def __init__(self,
               path: pathlib.Path,
               name: str,
               extension: str,
               with_zip: bool,
               ):
    super(JSONStorage, self).__init__(path, name, extension)
    self.cache_path = self.cache_path / self.name
    (self.cache_path).mkdir(exist_ok = True)

    self.with_zip = with_zip
    self.jsonfile_count = 0
    self.file_count = 0

    self.files = []
    self.repos = self.loadRepos
    self.data  = ""
    l.logger().info("Set up JSON storage in {}".format(self.cache_path))

    return

  def __exit__(self, path, name, extension):
  
    if len(self.files) > 0:
      self._flush_json()
  
    with open(self.cache_path / "repos_list.json", 'w') as outf:
      json.dump(
        [
          {
            'repo_name': rn,
            'ref': rf
          } for rn, rf in self.repos
        ],
        outf,
        sort_keys = True,
        indent = 2
      )
    self.repos = set()
  
    with open(self.cache_path / "data.txt", 'w') as outf:
      outf.write(self.data)
    self.data = ""

    return

  def save(self,
           contentfile: typing.Union[
                          bqdb.bqData,
                          bqdb.bqFile,
                          bqdb.bqRepo
                        ]
           ) -> None:
    if isinstance(contentfile, bqdb.bqData):
      self.data = "{}\n\n{}".format(contentfile.key, contentfile.value)
    elif isinstance(contentfile, bqdb.bqRepo):
      self.repos.add((contentfile.repo_name, contentfile.ref))
    else:
      self.files.append(contentfile.ToJSONDict())
      self.file_count += 1
      self.repos.add((contentfile.repo_name, contentfile.ref))
      if self.file_count % 500000:
        self._flush_json()
    return

  def _flush_json(self) -> None:

    filename = lambda ext: "{}.{}".format(self.jsonfile_count, ext)

    with open(self.cache_path / filename("json"), 'w') as outf:

      json.dump(self.files, outf, indent = 2)
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
    return len(self.repos)

  @property
  def main_repocount(self):
    return self.db.main_repo_count

  @property
  def other_repocount(self):
    return self.db.other_repo_count

  @property
  def filecount(self):
    return self.maincount + self.othercount

  @property
  def maincount(self):
    return self.db.mainfile_count + len(self.main_files)

  @property
  def othercount(self):
    return self.db.otherfile_count + len(self.other_files)

  @property
  def mainfiles(self):
    return self.db.main_files

  @property
  def otherfiles(self):
    return self.db.other_files

  @property
  def loadRepos(self):
    return self.repos

  @property
  def content_data(self):
    return self.db.data

  def __init__(self,
               path: pathlib.Path,
               name: str,
               extension: str
               ):
    super(dbStorage, self).__init__(path, name, extension)
    self.db = bqdb.bqDatabase("sqlite:///{}".format(self.cache_path / (self.name + ".db")))

    self.main_ids  = self.db.main_ids
    self.other_ids = self.db.other_ids
    self.repos     = self.db.loadRepos

    self.main_files  = set()
    self.other_files = set()
    self.data  = None
    self.flush_freq = 20000

    l.logger().info("Set up SQL storage in {}".format(self.cache_path))

  def __exit__(self, path, name, extension):
    self.flush()
    return

  def save(self,
           contentfile: typing.Union[
                          bqdb.bqData,
                          bqdb.bqFile,
                          bqdb.bqRepo
                        ]
           ) -> None:
    if isinstance(contentfile, bqdb.bqData):
      self.data = contentfile
    elif isinstance(contentfile, bqdb.bqRepo):
      self.repos.add((contentfile.repo_name, contentfile.ref))
    else: # bqFile.
      if isinstance(contentfile, bqdb.bqMainFile):

        if contentfile.id not in self.main_ids:
          self.repos.add((contentfile.repo_name, contentfile.ref))
          self.main_ids.add(contentfile.id)
          self.main_files.add(contentfile)

        if len(self.main_files) > self.flush_freq:
          self.flushToDB(self.main_files)
          self.main_files = set()

      elif isinstance(contentfile, bqdb.bqOtherFile):

        if contentfile.id not in self.other_ids:
          self.repos.add((contentfile.repo_name, contentfile.ref))
          self.other_ids.add(contentfile.id)
          self.other_files.add(contentfile)

        if len(self.other_files) > self.flush_freq:
          self.flushToDB(self.other_files)
          self.other_files = set()
    return

  def flush(self):
    """Flushes all cached data to DB."""
    ## Write data
    if self.data is not None:
      with self.db.Session(commit = True) as session:
        entry = session.query(
          bqdb.bqData
        ).filter_by(key = self.data.key).first()
        if entry is not None:
          entry.value = self.data.value
        else:
          session.add(self.data)

    ## Write repos
    if self.repocount > self.db.repo_count:
      for en, (repo_name, ref) in enumerate(self.repos):
        content = bqdb.bqRepo(**bqdb.bqRepo.FromArgs(
            self.db.repo_count + en, {'repo_name': repo_name, 'ref': ref})
        )
        with self.db.Session(commit = True) as session:
          exists = session.query(
            bqdb.bqRepo
          ).filter_by(repo_name = content.repo_name, ref = content.ref).scalar() is not None
          if not exists:
            session.add(content)

    if len(self.main_files) > 0:
      self.flushToDB(self.main_files)
      self.main_files = set()

    if len(self.other_files) > 0:
      self.flushToDB(self.other_files)
      self.other_files = set()
    return

  def flushToDB(self, files: typing.Set[bqdb.bqFile]) -> None:
    with self.db.Session(commit = True) as session:
      for file in files:
        session.add(file)
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
           contentfile: bqdb.bqFile
           ) -> None:
    raise NotImplementedError
