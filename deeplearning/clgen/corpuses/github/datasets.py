"""BigQuery Dataset structures"""
import os
import pathlib
import progressbar
import humanize

from google.cloud import bigquery
from absl import flags

from deeplearning.clgen.corpuses.github import bigQuery_database
from eupy.native import logger as l

FLAGS = flags.FLAGS

# flags.DEFINE_boolean(
#   "bq_only_repos",
#   False,
#   "Avoid huge queries, mine only repo entries for specified language."
# )

# flags.DEFINE_boolean(
#   "bq_only_files",
#   False,
#   "Do not explicitly mine repository entries for contentfiles."
# )

# languages = {
#   'opencl': ['.cl'],
#   'c'     : ['.c'],
#   'cpp'   : ['.cc', '.cpp', '.cxx', '.c++'],
#   'java'  : ['.java'],
#   'python': ['.py'],
# }

languages = {
  'opencl': openclDataset,
  'c'     : cDataset,
  'cpp'   : cppDataset,
  'java'  : javaDataset,
  'python': pythonDataset,
}

class Dataset(object):
  def __init__(self, dataset_id: str = None):
    self.config = bigquery.QueryJobConfig()
    self.config.allow_large_results = True
    self.client = bigquery.Client(default_query_job_config = config)

    self.project_id = self.client.project_id
    self.dataset_id = "generic_github" if dataset_id is None else dataset_id
    self.dataset_id = "{}_github".format(dataset_id or "generic")
    return

  @classmethod
  def FromArgs(cls, lang = None):
    if lang is None:
      return Dataset()
    elif lang not in languages:
      raise NotImplementedError(lang)
    return languages[lang]()

class openclDataset(Dataset):
  def __init__(self):
    super(openclDataset, self).__init__("opencl")
    self.extentions = ['.cl']
    return

class cDataset(Dataset):
  def __init__(self):
    super(cDataset, self).__init__("c")
    self.extentions = ['.c']
    return

class cppDataset(Dataset):
  def __init__(self):
    super(cppDataset, self).__init__("cpp")
    self.extentions = ['.cc'. 'cpp', '.cxx', '.c++']
    return

class javaDataset(Dataset):
  def __init__(self):
    super(javaDataset, self).__init__("java")
    self.extentions = ['.java']
    return

class pythonDataset(Dataset):
  def __init__(self):
    super(pythonDataset, self).__init__("python")
    self.extentions = ['.py']
    return
