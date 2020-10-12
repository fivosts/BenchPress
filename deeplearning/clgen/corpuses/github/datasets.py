"""BigQuery Dataset structures"""
import os
import typing
import pathlib
import progressbar
import humanize
from google.cloud import bigquery

from deeplearning.clgen.corpuses.github import bigQuery_database
from eupy.native import logger as l

# Currently available dataset subclasses.
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
  def FromArgs(cls,
               client: bigquery.Client,
               lang: int,
               ) -> Dataset:
    """Use this classmethod to initialize a Dataset."""
    if lang not in languages:
      raise NotImplementedError(lang)
    return languages[lang](client)
  
  @property
  def filecount(self):
    """Return file count of represented query."""
    if self.file_count is None:
      return self.filecount_query()
    else:
      return self.file_count

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
    self.client  = client    
    self.dataset, self.tables = self._setupDataset(
      "{}.clgen_{}_github".format(self.client.project, dataset_id or "generic")
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
      return dataset
    except Exception as e:
      raise e
      dataset = client.create_dataset(dataset, timeout = 30)

    return dataset, self._setupTables(self, dataset_id)

  def _setupTable(self, dataset_id: str) -> typing.Dict[str, bigquery.Table]:
    """API request that gets or sets bigquery.Table instances."""
    table_reg = {
      'bq_contentfiles': bigQuery_database.bqFile.bqSchema
      'bq_repofiles'   : bigQuery_database.bqRepo.bqSchema,
      'bq_data'        : bigQuery_database.bqData.bqSchema,
    }
    for reg, sc in table_reg.items():
      table_id = "{}.{}".format(dataset_id, reg)
      table = bigquery.Table(table_id, schema = sc)
      try:
        table_reg[reg] = client.get_table(table_id)
      except Exception as e:
        raise e
        table_reg[reg] = client.create_table(table)
    return table_reg

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

    job = self.client.query(count_query)
    for f in job:
      self.file_count = (f[0], 0)
      return self.file_count

  def repository_query(self) -> typing.Tuple[typing.Callable]:
    """Returns iterable of query files"""
    repo_query = """
    SELECT DISTINCT file.repo_name, file.ref
    FROM `bigquery-public-data.github_repos.files` as file
    {}
    """.format(not self.query_file_id or "WHERE " + self.query_file_id)
    return (self.client.query(repo_query), None)

  def contentfile_query(self) -> typing.Tuple[typing.Callable]:
    """Returns iterable of query files"""
    contentfile_query = """
    SELECT file.repo_name, file.path, file.ref, file.mode, 
           file.id, file.symlink_target, contentfile.size, 
           contentfile.content, contentfile.binary, contentfile.copies
    FROM `bigquery-public-data.github_repos.contents` as contentfile
    INNER JOIN `bigquery-public-data.github_repos.files` as file ON file.id = contentfile.id {}
    """.format(not self.query_file_id or "AND " + self.query_file_id)
    return (self.client.query(contentfile_query), None)

class openclDataset(Dataset):
  """Opencl Dataset"""
  def __init__(self,
               client: bigquery.Client,
               ):

    self.extensions = ['.cl']
    self.query_exception = [
        "(substr(file.path, {}, {}) = '{}' AND contentfile.content LIKE '%kernel void%')"
          .format(-len(ext), 1+len(ext), ext)
      for ext in ['.c', '.cc', '.cpp', '.cxx', '.c++']
    ]
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
    {} {}
    """.format(not self.query_file_id or "WHERE " + self.query_file_id, self.query_exception)

    job = self.client.query(other_count_query)
    for f in job:
      self.file_count[1] = f[0]
      return self.file_count

  def repository_query(self) -> typing.Tuple[typing.Callable, typing.Callable]:
    """
    Query repositories that tested positive for having CL.
    CL has its own function, because two types of files are checked:
    '.cl' files and any C/C++ file that contains the keyword 'kernel void'
    """
    cl_repo_it, _ = super(openclDataset, self).repository_query()
    other_repo_query = """
    SELECT DISTINCT file.repo_name, file.ref
    FROM `bigquery-public-data.github_repos.files` as file
    {} {}
    """.format(not self.query_file_id or "WHERE " + self.query_file_id, self.query_exception)
    return (cl_repo_it, self.client.query(other_repo_query))

  def contentfile_query(self) -> typing.Tuple[typing.Callable, typing.Callable]:
    """
    Query contentfiles that tested positive for being CL.
    CL has its own function, because two types of files are checked:
    '.cl' files and any C/C++ file that contains the keyword 'kernel void'
    """
    cl_file_it, _ = super(openclDataset, self).contentfile_query()
    other_file_query = """
    SELECT file.repo_name, file.path, file.ref, file.mode, 
           file.id, file.symlink_target, contentfile.size, 
           contentfile.content, contentfile.binary, contentfile.copies
    FROM `bigquery-public-data.github_repos.files` as file
    {} {}
    """.format(not self.query_file_id or "WHERE " + self.query_file_id, self.query_exception)
    return (cl_file_it, self.client.query(other_file_query))

class cDataset(Dataset):
  """C Dataset"""
  def __init__(self,
               client: bigquery.Client,
               ):
    self.extensions = ['.c']
    super(cDataset, self).__init__(client, "c")
    return

class cppDataset(Dataset):
  """C++ Dataset"""
  def __init__(self,
               client: bigquery.Client,
               ):
    self.extensions = ['.cpp'. 'cc', '.cxx', '.c++']
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
