"""BigQuery Dataset structures"""
import os
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
    return

  def filecount_query(self, client) -> int:
    """
    Queries the file count of files intended to query.
    Returns file count in int.
    """
    count_query = """
    SELECT COUNT(*)
    FROM `bigquery-public-data.github_repos.files` as file
    {}
    """.format(not self.query_file_id or "WHERE " + self.query_file_id)
    job = client.query(count_query)
    for f in job:
      return f[0]

class openclDataset(Dataset):
  """Opencl Dataset"""
  def __init__(self, data_format: int):
    self.extensions = ['.cl']
    super(openclDataset, self).__init__(data_format, "opencl")
    return

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
