"""BigQuery Dataset structures"""
import os
import typing
import pathlib
import progressbar
import humanize
import google
from google.cloud import bigquery

from deeplearning.clgen.corpuses.github import bigQuery_database
from eupy.native import logger as l

class Dataset(object):
  """Representation of dataset instance in Big Query"""
  @classmethod
  def FromArgs(cls,
               client: bigquery.Client,
               lang: int,
               ):
    """Use this classmethod to initialize a Dataset."""
    languages = {
      'generic': Dataset,
      'opencl' : openclDataset,
      'c'      : cDataset,
      'cpp'    : cppDataset,
      'java'   : javaDataset,
      'python' : pythonDataset,
    }
    if lang not in languages:
      raise NotImplementedError(lang)
    return languages[lang](client)
  
  @property
  def filecount(self) -> typing.Tuple[int, int]:
    """Return file count of represented query."""
    if self.file_count is None:
      return self.filecount_query()
    else:
      return self.file_count

  @filecount.setter
  def filecount(self, value: typing.Tuple[int, int]) -> None:
    self.file_count = value
    return

  @property
  def name(self):
    return self.dataset.dataset_id

  @property
  def extension(self):
    return self.extensions[0]
  
  def __init__(self,
               client: bigquery.Client,
               dataset_id: str = None
               ):
    """Generic Dataset class constructor. Not to be used directly."""
    self.client = client
    self.dataset, self.tables = self._setupDataset(
      "{}.clgen_{}_github".format(self.client.project, dataset_id or "generic")
    )
    self.queryConfig = lambda qt : bigquery.QueryJobConfig(
      destination = self.tables[qt],
    )

    self.query_file_id = ""
    if self.extensions is not None:
      self.query_file_id = " OR ".join(["substr(file.path, {}, {}) = '{}'".format(-len(ext), 1+len(ext), ext)
                              for ext in self.extensions
                        ])
    self.file_count = None
    return

  def _setupDataset(self, 
                    dataset_id: str
                    ) -> typing.Tuple[bigquery.Dataset, typing.Dict[str, bigquery.Table]]:
    """API request to get or set bigquery.Dataset instance and bigquery.Table."""
    dataset = bigquery.Dataset(dataset_id)
    dataset.location = "US"
    try:
      dataset = self.client.get_dataset(dataset_id)
    except google.api_core.exceptions.NotFound:
      dataset = self.client.create_dataset(dataset, timeout = 30)
    except Exception as e:
      raise e

    return dataset, self._setupTables(dataset_id)

  def _setupTables(self, dataset_id: str) -> typing.Dict[str, bigquery.Table]:
    """API request that gets or sets bigquery.Table instances."""
    table_reg = {
      'bq_contentfiles': bigQuery_database.bqFile.bqSchema,
      'bq_repofiles'   : bigQuery_database.bqRepo.bqSchema,
      'bq_data'        : bigQuery_database.bqData.bqSchema,
    }
    for reg, get_sc in table_reg.items():
      table_id = "{}.{}".format(dataset_id, reg)
      table = bigquery.Table(table_id, schema = get_sc())
      try:
        table_reg[reg] = self.client.get_table(table_id)
      except google.api_core.exceptions.NotFound:
        table_reg[reg] = self.client.create_table(table)
      except Exception as e:
        raise e
    return table_reg

  def filecount_query(self) -> typing.Tuple[int, int]:
    """
    Queries the file count of files intended to query.
    Returns file count in int.
    """
    l.getLogger().info("Running file count query...")
    query = """
    SELECT COUNT(*)
    FROM `bigquery-public-data.github_repos.files` as file
    {}
    """.format(not self.query_file_id or "WHERE " + self.query_file_id)

    try:
      job = self.client.query(query)
      for f in job:
        self.file_count = (f[0], 0)
        return self.file_count
    except google.api_core.exceptions.Forbidden as e:
      l.getLogger().error(e)
      exit()

  def repository_query(self) -> typing.Tuple[typing.Callable]:
    """Returns iterable of query files"""
    l.getLogger().info("Retrieving repository list of specs...")
    query = """
    SELECT DISTINCT file.repo_name, file.ref
    FROM `bigquery-public-data.github_repos.files` as file
    {}
    """.format(not self.query_file_id or "WHERE " + self.query_file_id)
    try:
      rows = self.client.query(query, job_config = self.queryConfig('bq_repofiles')).result()
    except google.api_core.exceptions.Forbidden as e:
      l.getLogger().error(e)
      exit()
    return (rows, None)

  def contentfile_query(self) -> typing.Tuple[typing.Callable]:
    """Returns iterable of query files"""
    l.getLogger().info("Retrieving contentfiles...")
    if not self.query_file_id:
      l.getLogger().warning(
        "contentfile_query for generic language is not allowed, unless one wants to query the whole universe."
      )
      return None, None
    query = """
    SELECT file.repo_name, file.path, file.ref, file.mode, 
           file.id, file.symlink_target, contentfile.size, 
           contentfile.content, contentfile.binary, contentfile.copies
    FROM `bigquery-public-data.github_repos.contents` as contentfile
    INNER JOIN `bigquery-public-data.github_repos.files` as file ON file.id = contentfile.id {}
    """.format(not self.query_file_id or "AND (" + self.query_file_id + ")")
    try:
      rows = self.client.query(query, job_config = self.queryConfig('bq_contentfiles')).result()
    except google.api_core.exceptions.Forbidden as e:
      l.getLogger().error(e)
      exit()
    return (rows, None)

class openclDataset(Dataset):
  """Opencl Dataset"""
  def __init__(self,
               client: bigquery.Client,
               ):

    self.extensions = ['.cl']
    self.query_exception = ' AND (' + ' OR '.join([
        "(substr(file.path, {}, {}) = '{}' AND contentfile.content LIKE '%kernel void%')"
          .format(-len(ext), 1+len(ext), ext)
      for ext in ['.c', '.cc', '.cpp', '.cxx', '.c++', '.h', '.hpp']
    ]) + ')'
    super(openclDataset, self).__init__(client, "opencl")
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
    INNER JOIN `bigquery-public-data.github_repos.contents` as contentfile
    ON file.id = contentfile.id
    {}
    """.format(self.query_exception or "")

    try:
      job = self.client.query(other_count_query)
      for f in job:
        self.file_count[1] = f[0]
        return self.file_count
    except google.api_core.exceptions.Forbidden as e:
      l.getLogger().error(e)
      exit()

  def repository_query(self) -> typing.Tuple[typing.Callable, typing.Callable]:
    """
    Query repositories that tested positive for having CL.
    CL has its own function, because two types of files are checked:
    '.cl' files and any C/C++ file that contains the keyword 'kernel void'
    """
    cl_repo_it, _ = super(openclDataset, self).repository_query()
    query = """
    SELECT DISTINCT file.repo_name, file.ref
    FROM `bigquery-public-data.github_repos.files` as file
    INNER JOIN `bigquery-public-data.github_repos.contents` as contentfile
    ON file.id = contentfile.id
    {}
    """.format(self.query_exception or "")
    try:
      rows = self.client.query(query, job_config = self.queryConfig('bq_repofiles')).result()
    except google.api_core.exceptions.Forbidden as e:
      l.getLogger().error(e)
      exit()
    return (cl_repo_it, rows)

  def contentfile_query(self) -> typing.Tuple[typing.Callable, typing.Callable]:
    """
    Query contentfiles that tested positive for being CL.
    CL has its own function, because two types of files are checked:
    '.cl' files and any C/C++ file that contains the keyword 'kernel void'
    """
    cl_file_it, _ = super(openclDataset, self).contentfile_query()
    query = """
    SELECT file.repo_name, file.path, file.ref, file.mode, 
           file.id, file.symlink_target, contentfile.size, 
           contentfile.content, contentfile.binary, contentfile.copies
    FROM `bigquery-public-data.github_repos.files` as file
    INNER JOIN `bigquery-public-data.github_repos.contents` as contentfile
    ON file.id = contentfile.id
    {}
    """.format(self.query_exception or "")
    try:
      rows = self.client.query(query, job_config = self.queryConfig('bq_contentfiles')).result()
    except google.api_core.exceptions.Forbidden as e:
      l.getLogger().error(e)
      exit()
    return (cl_file_it, rows)

class cDataset(Dataset):
  """C Dataset"""
  def __init__(self,
               client: bigquery.Client,
               ):
    self.extensions = ['.c', '.h']
    super(cDataset, self).__init__(client, "c")
    return

class cppDataset(Dataset):
  """C++ Dataset"""
  def __init__(self,
               client: bigquery.Client,
               ):
    self.extensions = ['.cpp', 'cc', '.cxx', '.c++', '.h', '.hpp']
    super(cppDataset, self).__init__(client, "cpp")
    return

class javaDataset(Dataset):
  """java Dataset"""
  def __init__(self,
               client: bigquery.Client,
               ):
    self.extensions = ['.java']
    super(javaDataset, self).__init__(client, "java")
    return

class pythonDataset(Dataset):
  """python Dataset"""
  def __init__(self,
               client: bigquery.Client,
               ):
    self.extensions = ['.py']
    super(pythonDataset, self).__init__(client, "python")
    return
