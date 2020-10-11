"""BigQuery Dataset structures"""
import os
import typing
import pathlib
import progressbar
import humanize
from google.cloud import bigquery

from deeplearning.clgen.corpuses.github import bigQuery_database
from eupy.native import logger as l

languages = {
  'generic': Dataset,
  'opencl' : openclDataset,
  'c'      : cDataset,
  'cpp'    : cppDataset,
  'java'   : javaDataset,
  'python' : pythonDataset,
}

class Dataset(object):
  """Representation of dataset instance in Big Query"""
  @classmethod
  def FromArgs(cls, lang, data_format):
    if lang not in languages:
      raise NotImplementedError(lang)
    return languages[lang](data_format)
  
  def __init__(self, data_format: int, dataset_id: str = None):

    self.dataset_id = "generic_github" if dataset_id is None else dataset_id
    self.dataset_id = "{}_github".format(dataset_id or "generic")
    self.data_format = data_format

    self.query_file_id = ""
    if self.extensions is not None:
      self.query_file_id = " OR ".join(["substr(file.path, {}, {}) = '{}'".format(-len(ext), 1+len(ext), ext)
                              for ext in self.extensions
                        ])
    self.file_count = None
    return

  @property
  def filecount(self):
    """Return file count of represented query."""
    if self.file_count is None:
      return self.filecount_query()
    else:
      return self.file_count

  def filecount_query(self) -> typing.Tuple[int, int]:
    """
    Queries the file count of files intended to query.
    Returns file count in int.
    """
    count_query = """
    SELECT COUNT(*)
    FROM `bigquery-public-data.github_repos.files` as file
    {}
    """.format(not self.query_file_id or "WHERE " + self.query_file_id)
    with bigquery.Client() as client:
      job = client.query(count_query)
      for f in job:
        self.file_count = (f[0], 0)
        return self.file_count

  def repository_query(self, client) -> typing.Tuple[typing.Callable]:
    """Returns iterable of query files"""
    repo_query = """
    SELECT DISTINCT file.repo_name, file.ref
    FROM `bigquery-public-data.github_repos.files` as file
    {}
    """.format(not self.query_file_id or "WHERE " + self.query_file_id)
    return client.query(repo_query,)

  def contentfile_query(self, client) -> typing.Tuple[typing.Callable]:
    """Returns iterable of query files"""
    contentfile_query = """
    SELECT file.repo_name, file.path, file.ref, file.mode, 
           file.id, file.symlink_target, contentfile.size, 
           contentfile.content, contentfile.binary, contentfile.copies
    FROM `bigquery-public-data.github_repos.contents` as contentfile
    INNER JOIN `bigquery-public-data.github_repos.files` as file ON file.id = contentfile.id {}
    """.format(not self.query_file_id or "AND " + self.query_file_id)
    return client.query(contentfile_query,)

class openclDataset(Dataset):
  """Opencl Dataset"""
  def __init__(self, data_format: int):
    self.extensions = ['.cl']
    self.query_exception = [
        "(substr(file.path, {}, {}) = '{}' AND contentfile.content LIKE '%kernel void%')"
          .format(-len(ext), 1+len(ext), ext)
      for ext in ['.c', '.cc', '.cpp', '.cxx', '.c++']
    ]
    super(openclDataset, self).__init__(data_format, "opencl")
    return

  def filecount_query(self) -> typing.Tuple[int, int]:
    """
    Queries the file count of files intended to query.
    Returns file count in int.
    """
    super(openclDataset, self).filecount_query()
    other_count_query = """
    SELECT COUNT(*)
    FROM `bigquery-public-data.github_repos.files` as file
    {} {}
    """.format(not self.query_file_id or "WHERE " + self.query_file_id, self.query_exception)

    with bigquery.Client() as client:
      job = client.query(other_count_query)
      for f in job:
        self.file_count[1] = f[0]
        return self.file_count

  def repository_query(self,
                       client
                       ) -> typing.Tuple[typing.Callable, typing.Callable]:
    """
    Query repositories that tested positive for having CL.
    CL has its own function, because two types of files are checked:
    '.cl' files and any C/C++ file that contains the keyword 'kernel void'
    """
    cl_repo_it = super(openclDataset, self).repository_query(client)
    other_repo_query = """
    SELECT DISTINCT file.repo_name, file.ref
    FROM `bigquery-public-data.github_repos.files` as file
    {} {}
    """.format(not self.query_file_id or "WHERE " + self.query_file_id, self.query_exception)
    return (cl_repo_it, client.query(other_repo_query))

  def contentfile_query(self,
                        client,
                        ) -> typing.Tuple[typing.Callable, typing.Callable]:
    cl_file_it = super(openclDataset, self).contentfile_query(client)
    other_file_query = """
    SELECT file.repo_name, file.path, file.ref, file.mode, 
           file.id, file.symlink_target, contentfile.size, 
           contentfile.content, contentfile.binary, contentfile.copies
    FROM `bigquery-public-data.github_repos.files` as file
    {} {}
    """.format(not self.query_file_id or "WHERE " + self.query_file_id, self.query_exception)
    return (cl_file_it, client.query(other_file_query))

class cDataset(Dataset):
  """C Dataset"""
  def __init__(self, data_format: int):
    self.extensions = ['.c']
    super(cDataset, self).__init__(data_format, "c")
    return

class cppDataset(Dataset):
  """C++ Dataset"""
  def __init__(self, data_format: int):
    self.extensions = ['.cc'. 'cpp', '.cxx', '.c++']
    super(cppDataset, self).__init__(data_format, "cpp")
    return

class javaDataset(Dataset):
  """java Dataset"""
  def __init__(self, data_format: int):
    self.extensions = ['.java']
    super(javaDataset, self).__init__(data_format, "java")
    return

class pythonDataset(Dataset):
  """python Dataset"""
  def __init__(self, data_format: int):
    self.extensions = ['.py']
    super(pythonDataset, self).__init__(data_format, "python")
    return
