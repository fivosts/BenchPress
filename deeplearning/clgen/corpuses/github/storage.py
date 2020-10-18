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

  @property
  def repoTuple(self) -> typing.List[typing.Tuple[str, str]]:
    # Returns repo list in type list of tuples.
    # t[0] -> repo_name, t[1] -> ref
    return [tuple(r.split(', ')) for r in self.repos]

  def __init__(self,
               path: pathlib.Path,
               name: str,
               extension: str):
    self.cache_path = path
    self.cache_path.mkdir(exist_ok = True)
    self.name       = name
    self.extension  = extension

    self.repos      = set()
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
    l.getLogger().info("Set up ZIP storage in {}".format(self.cache_path))

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
      self.cached_content.append(contentfile.content)
      self.file_count += 1
      if self.file_count % self.flush_counter == 0:
        self.zipFiles()
      self.repos.add("{}, {}".format(contentfile.repo_name, contentfile.ref))
    elif isinstance(contentfile, bigQuery_database.bqData):
      self.data_file = "{}\n\n{}".format(contentfile.key, contentfile.value)
    elif isinstance(contentfile, bigQuery_database.bqRepo):
      self.repos.add("{}, {}".format(contentfile.repo_name, contentfile.ref))
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
    l.getLogger().info("Set up folder storage in {}".format(self.cache_path))

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
      with open(self.cache_path / "{}{}".format(self.file_count, self.extension), 'w') as f:
        f.write(contentfile.content)
      self.file_count += 1
      self.repos.add("{}, {}".format(contentfile.repo_name, contentfile.ref))
    elif isinstance(contentfile, bigQuery_database.bqData):
      with open(self.cache_path / "data.txt", 'w') as f:
        f.write("{}\n\n{}".format(contentfile.key, contentfile.value))
    elif isinstance(contentfile, bigQuery_database.bqRepo):
      self.repos.add("{}, {}".format(contentfile.repo_name, contentfile.ref))
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

    self.with_zip = with_zip
    self.jsonfile_count = 0
    self.file_count = 0

    self.data  = ""
    self.files = []
    l.getLogger().info("Set up JSON storage in {}".format(self.cache_path))

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
                          bigQuery_database.bqData,
                          bigQuery_database.bqFile,
                          bigQuery_database.bqRepo
                        ]
           ) -> None:
    if isinstance(contentfile, bigQuery_database.bqData):
      self.data = "{}\n\n{}".format(contentfile.key, contentfile.value)
    elif isinstance(contentfile, bigQuery_database.bqRepo):
      self.repos.add("{}, {}".format(contentfile.repo_name, contentfile.ref))
    else:
      self.files.append(contentfile.ToJSONDict())
      self.file_count += 1
      if self.file_count % 500000:
        self._flush_json()
      self.repos.add("{}, {}".format(contentfile.repo_name, contentfile.ref))
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
  def filecount(self):
    return (self.db.file_count +
            len(self.main_files) +
            len(self.other_files) +
            len(self.header_files)
           )

  def __init__(self,
               path: pathlib.Path,
               name: str,
               extension: str
               ):
    super(dbStorage, self).__init__(path, name, extension)
    self.db = bigQuery_database.bqDatabase("sqlite:///{}".format(self.cache_path / (self.name + ".db")))

    self.data  = None
    self.repos = self.db.repo_entries
    if not self.repos:
      self.repos = set()

    self.flush_freq = 500000
    self.main_sha   = self.db.main_sha
    self.main_files = set()

    self.other_sha   = self.db.other_sha
    self.other_files = set()

    self.header_files = set()
    self.db.get_includes
    l.getLogger().info("Set up SQL storage in {}".format(self.cache_path))

  def __exit__(self, path, name, extension):
    with self.db.Session(commit = True) as session:
      ## Write data
      if self.db.data is not None:
        entry = session.query(
          bigQuery_database.bqData
        ).filter_by(key = self.data.key).first()
        entry.value = self.data.value
      else:
        session.add(self.data)

      ## Write repos
      for repo in self.repos:
        repo_name, ref = repo.split(', ')
        content = bigQuery_database.bqRepo(**bigQuery_database.bqRepo.FromArgs(
            self.db.repo_count, {'repo_name': repo_name, 'ref': ref})
        )
        exists = session.query(
          bigQuery_database.bqRepo
        ).filter_by(repo_name = content.repo_name, ref = content.ref).scalar() is not None
        if not exists:
          session.add(content)
    # Flush remaining contentfiles.
    self.flushToDB(self.main_files)
    self.flushToDB(self.other_files)
    self.flushToDB(self.header_files)
    self.main_files   = set()
    self.other_files  = set()
    self.header_files = set()
    return

  def save(self,
           contentfile: typing.Union[
                          bigQuery_database.bqData,
                          bigQuery_database.bqFile,
                          bigQuery_database.bqRepo
                        ]
           ) -> None:
    if isinstance(contentfile, bigQuery_database.bqData):
      self.data = contentfile
    else: # Do this for both bqRepo and bqFile.
      if isinstance(contentfile, bigQuery_database.bqMainFile):
        self.repos.add("{}, {}".format(contentfile.repo_name, contentfile.ref))
        if contentfile.sha256 not in self.main_sha:
          self.main_sha.add(contentfile.sha256)
          self.main_files.add(contentfile)
        if len(self.main_files) > self.flush_freq:
          self.flushToDB(self.main_files)
          self.main_files = set()
      elif isinstance(contentfile, bigQuery_database.bqOtherFile):
        self.repos.add("{}, {}".format(contentfile.repo_name, contentfile.ref))
        if contentfile.sha256 not in self.other_sha:
          self.other_sha.add(contentfile.sha256)
          self.other_files.add(contentfile)
        if len(self.other_files) > self.flush_freq:
          self.flushToDB(self.other_files)
          self.other_files = set()
      elif isinstance(contentfile, bigQuery_database.bqHeaderFile):
        self.header_files.add(contentfile)
        if len(self.header_files) > self.flush_freq:
          self.flushToDB(self.header_files)
          self.header_files = set()
    return

  def flush(self):
    """Flushes all cached data to DB."""
    self.flushToDB(self.main_files)
    self.flushToDB(self.other_files)
    self.flushToDB(self.header_files)
    self.main_files   = set()
    self.other_files  = set()
    self.header_files = set()
    return

  def flushToDB(self, files: typing.Set[bigQuery_database.bqFile]) -> None:
    with self.db.Session(commit = True) as session:
      for en, file in enumerate(files):
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
           contentfile: bigQuery_database.bqFile
           ) -> None:
    raise NotImplementedError
