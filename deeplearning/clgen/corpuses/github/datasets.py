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

flags.DEFINE_boolean(
  "bq_only_repos",
  False,
  "Avoid huge queries, mine only repo entries for specified language."
)

flags.DEFINE_boolean(
  "bq_only_files",
  False,
  "Do not explicitly mine repository entries for contentfiles."
)

languages = {
  'opencl': ['.cl'],
  'c'     : ['.c'],
  'cpp'   : ['.cc', '.cpp', '.cxx', '.c++'],
  'java'  : ['.java'],
  'python': ['.py'],
}

class Dataset(object):
  def __init__(self):
    return

  @classmethod
  def FromArgs(cls, lang):
    

class openclDataset(Dataset):
  def __init__(self):
    super(openclDataset, self).__init__()
    return

class cDataset(Dataset):
  def __init__(self):
    super(cDataset, self).__init__()
    return

class cppDataset(Dataset):
  def __init__(self):
    super(cppDataset, self).__init__()
    return

class javaDataset(Dataset):
  def __init__(self):
    super(javaDataset, self).__init__()
    return

class pythonDataset(Dataset):
  def __init__(self):
    super(pythonDataset, self).__init__()
    return
