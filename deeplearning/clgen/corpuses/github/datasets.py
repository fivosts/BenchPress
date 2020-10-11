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
  def __init__(self, data_format: int, dataset_id: str = None):
    self.config = bigquery.QueryJobConfig()
    self.config.allow_large_results = True
    self.client = bigquery.Client(default_query_job_config = config)

    self.project_id = self.client.project_id
    self.dataset_id = "generic_github" if dataset_id is None else dataset_id
    self.dataset_id = "{}_github".format(dataset_id or "generic")

    self.data_format = data_format
    return

  @classmethod
  def FromArgs(cls, lang, data_format):
    if lang not in languages:
      raise NotImplementedError(lang)
    return languages[lang](data_format)

class openclDataset(Dataset):
  def __init__(self, data_format: int):
    super(openclDataset, self).__init__(data_format, "opencl")
    self.extentions = ['.cl']
    return

class cDataset(Dataset):
  def __init__(self, data_format: int):
    super(cDataset, self).__init__(data_format, "c")
    self.extentions = ['.c']
    return

class cppDataset(Dataset):
  def __init__(self, data_format: int):
    super(cppDataset, self).__init__(data_format, "cpp")
    self.extentions = ['.cc'. 'cpp', '.cxx', '.c++']
    return

class javaDataset(Dataset):
  def __init__(self, data_format: int):
    super(javaDataset, self).__init__(data_format, "java")
    self.extentions = ['.java']
    return

class pythonDataset(Dataset):
  def __init__(self, data_format: int):
    super(pythonDataset, self).__init__(data_format, "python")
    self.extentions = ['.py']
    return
