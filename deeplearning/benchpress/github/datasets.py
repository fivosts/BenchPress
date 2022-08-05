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
import typing
import pathlib
import progressbar
import humanize
import google
from google.cloud import bigquery
from absl import flags

from deeplearning.benchpress.github import bigQuery_database
from deeplearning.benchpress.util import logging as l

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
  "bq_wait_permission",
  True,
  "Ask for permission every time a query is about to happen."
)

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
  def language(self):
    return "generic"

  @property
  def extension(self):
    if self.extensions:
      return self.extensions[0]
    else:
      return None

  def __init__(self,
               client: bigquery.Client,
               dataset_id: str = None,
               extensions: typing.List[str] = None,
               ):
    """Generic Dataset class constructor. Not to be used directly."""
    self.client = client
    self.dataset, self.tables = self._setupDataset(
      "{}.clgen_{}_github".format(self.client.project, dataset_id or "generic")
    )
    self.queryConfig = lambda qt, qr = [], dr = False : bigquery.QueryJobConfig(
      destination = self.tables[qt],
      write_disposition = 'WRITE_TRUNCATE',
      query_parameters = qr,
      dry_run = dr,
    )

    self.extensions = extensions
    self.query_file_id = ""
    if self.extensions is not None:
      self.query_file_id = " OR ".join(["substr(file.path, {}, {}) = '{}'".format(-len(ext), 1+len(ext), ext)
                              for ext in self.extensions
                        ])
    self.file_count = None
    l.logger().info("{} dataset initialized.".format(self.language))
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
      'main_files'   : bigQuery_database.bqFile.bqSchema,
      'other_files'  : bigQuery_database.bqFile.bqSchema,
      'repositories' : bigQuery_database.bqRepo.bqSchema,
      'data'         : bigQuery_database.bqData.bqSchema,
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
    query = """
    SELECT COUNT(*)
    FROM `bigquery-public-data.github_repos.files` as file
    {}
    """.format("" if not self.query_file_id else "WHERE " + self.query_file_id)

    dry_run_job = self.client.query(query, job_config = self.queryConfig('main_files', dr = True))
    l.logger().warn("This query is going to consume {}".format(
        humanize.naturalsize(dry_run_job.total_bytes_processed)
      )
    )
    l.logger().info(query)
    if FLAGS.bq_wait_permission:
      l.logger().warn("Hit any button to continue...")
      try:
        input()
      except KeyboardInterrupt:
        return (0, 0)

    l.logger().info("Running file count query...")

    try:
      job = self.client.query(query)
      for f in job:
        self.file_count = (f[0], 0)
        return self.file_count
    except google.api_core.exceptions.Forbidden as e:
      l.logger().error(e)
      exit()

  def repository_query(self) -> typing.Tuple[bigquery.table.RowIterator]:
    """
    Queries the repositories' name/branch that contain files with requested
    specifications (e.g. OpenCL files).

    Returns iterable of query.
    """
    query = """
    SELECT DISTINCT file.repo_name, file.ref
    FROM `bigquery-public-data.github_repos.files` as file
    {}
    """.format("" if not self.query_file_id else "WHERE " + self.query_file_id)

    dry_run_job = self.client.query(query, job_config = self.queryConfig('repositories', dr = True))
    l.logger().warn("This query is going to consume {}".format(
        humanize.naturalsize(dry_run_job.total_bytes_processed)
      )
    )
    l.logger().info(query)
    if FLAGS.bq_wait_permission:
      l.logger().warn("Hit any button to continue...")
      try:
        input()
      except KeyboardInterrupt:
        return (None, None)

    l.logger().info("Retrieving repository list of specs...")

    try:
      rows = self.client.query(query, job_config = self.queryConfig('repositories')).result()
    except google.api_core.exceptions.Forbidden as e:
      l.logger().error(e)
      exit()
    return (rows, None)

  def contentfile_query(self) -> typing.Tuple[bigquery.table.RowIterator]:
    """
    Queries all contentfiles with requested specifications (e.g. specific file extensions).

    Returns iterable of query files
    """
    query = """
    SELECT MIN(file.repo_name) as repo_name,
           MIN(file.path) as path,
           MIN(file.ref) as ref,
           file.id,
           MIN(contentfile.size) as size,
           MIN(contentfile.content) as content
    FROM (`bigquery-public-data.github_repos.contents` as contentfile
    INNER JOIN `bigquery-public-data.github_repos.files` as file ON file.id = contentfile.id {})
    GROUP BY file.id
    """.format("" if not self.query_file_id else "AND (" + self.query_file_id + ")")

    # query = """
    # SELECT file.id
    # FROM `bigquery-public-data.github_repos.files` as file
    # WHERE {}
    # GROUP BY file.id
    # """.format("" if not self.query_file_id else "(" + self.query_file_id + ")")

    dry_run_job = self.client.query(query, job_config = self.queryConfig('main_files', dr = True))
    l.logger().warn("This query is going to consume {}".format(
        humanize.naturalsize(dry_run_job.total_bytes_processed)
      )
    )
    l.logger().info(query)
    if FLAGS.bq_wait_permission:
      l.logger().warn("Hit any button to continue...")
      try:
        input()
      except KeyboardInterrupt:
        return (None, None)

    l.logger().info("Retrieving {} contentfiles...".format(self.dataset.dataset_id))

    try:
      rows = self.client.query(query, job_config = self.queryConfig('main_files')).result()
    except google.api_core.exceptions.Forbidden as e:
      l.logger().error(e)
      exit()
    return (rows, None)

  def header_file_query(self) -> None:
    """
    From the repositories that contentfiles were scraped from, also get their header files
    for header inlining reasons.

    Override this method IF you want header files fetched with the language's contentfiles.
    """
    return None

class openclDataset(Dataset):
  """Opencl Dataset"""

  @property
  def language(self):
    return "openCL"

  def __init__(self,
               client: bigquery.Client,
               ):

    extensions = ['.cl']
    super(openclDataset, self).__init__(client, "opencl", extensions)

    self.other_extensions = ['.c', '.cc', '.cpp', '.cxx', '.c++', '.h', '.hpp']
    self.query_exception = ' AND (' + ' OR '.join([
        "(substr(file.path, {}, {}) = '{}' AND contentfile.content LIKE '%kernel void%')"
          .format(-len(ext), 1+len(ext), ext)
      for ext in self.other_extensions
    ]) + ')'
    return

  def filecount_query(self) -> typing.Tuple[int, int]:
    """
    Queries the file count of files intended to query.
    Returns file count in int.
    """
    super(openclDataset, self).filecount_query()
    query = """
    SELECT COUNT(*)
    FROM `bigquery-public-data.github_repos.files` as file
    INNER JOIN `bigquery-public-data.github_repos.contents` as contentfile
    ON file.id = contentfile.id
    {}
    """.format(self.query_exception or "")

    dry_run_job = self.client.query(query, job_config = self.queryConfig('repositories', dr = True))
    l.logger().warn("This query is going to consume {}".format(
        humanize.naturalsize(dry_run_job.total_bytes_processed)
      )
    )
    l.logger().info(query)
    if FLAGS.bq_wait_permission:
      l.logger().warn("Hit any button to continue...")
      try:
        input()
      except KeyboardInterrupt:
        return (0, 0)

    try:
      job = self.client.query(query)
      for f in job:
        self.file_count[1] = f[0]
        return self.file_count
    except google.api_core.exceptions.Forbidden as e:
      l.logger().error(e)
      exit()

  def repository_query(self) -> typing.Tuple[bigquery.table.RowIterator, bigquery.table.RowIterator]:
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

    dry_run_job = self.client.query(query, job_config = self.queryConfig('repositories', dr = True))
    l.logger().warn("This query is going to consume {}".format(
        humanize.naturalsize(dry_run_job.total_bytes_processed)
      )
    )
    l.logger().info(query)
    if FLAGS.bq_wait_permission:
      l.logger().warn("Hit any button to continue...")
      try:
        input()
      except KeyboardInterrupt:
        return (cl_repo_it, None)

    l.logger().info("Retrieving etc. repo list...")

    try:
      rows = self.client.query(query, job_config = self.queryConfig('repositories')).result()
    except google.api_core.exceptions.Forbidden as e:
      l.logger().error(e)
      exit()
    return (cl_repo_it, rows)

  def contentfile_query(self) -> typing.Tuple[bigquery.table.RowIterator, bigquery.table.RowIterator]:
    """
    Query contentfiles that tested positive for being CL.
    CL has its own function, because two types of files are checked:
    '.cl' files and any C/C++ file that contains the keyword 'kernel void'
    """
    cl_file_it, _ = super(openclDataset, self).contentfile_query()
    query = """
    SELECT file.repo_name, file.path, file.ref, file.id,
           contentfile.size, contentfile.content
    FROM `bigquery-public-data.github_repos.files` as file
    INNER JOIN `bigquery-public-data.github_repos.contents` as contentfile
    ON file.id = contentfile.id
    {}
    """.format(self.query_exception or "")

    dry_run_job = self.client.query(query, job_config = self.queryConfig('other_files', dr = True))
    l.logger().warn("This query is going to consume {}".format(
        humanize.naturalsize(dry_run_job.total_bytes_processed)
      )
    )
    l.logger().info(query)
    if FLAGS.bq_wait_permission:
      l.logger().warn("Hit any button to continue...")
      try:
        input()
      except KeyboardInterrupt:
        return (cl_file_it, None)

    l.logger().info("Retrieving etc. contentfiles...")

    try:
      rows = self.client.query(query, job_config = self.queryConfig('other_files')).result()
    except google.api_core.exceptions.Forbidden as e:
      l.logger().error(e)
      exit()
    return (cl_file_it, rows)

class cDataset(Dataset):
  """C Dataset"""
  @property
  def language(self):
    return "C"

  def __init__(self,
               client: bigquery.Client,
               ):
    extensions = ['.c']
    super(cDataset, self).__init__(client, "c", extensions)
    return

class cppDataset(Dataset):
  """C++ Dataset"""
  @property
  def language(self):
    return "C++"

  def __init__(self,
               client: bigquery.Client,
               ):
    extensions = ['.cpp', 'cc', '.cxx', '.c++', '.hpp']
    super(cppDataset, self).__init__(client, "cpp", extensions)
    return

class javaDataset(Dataset):
  """java Dataset"""
  @property
  def language(self):
    return "Java"

  def __init__(self,
               client: bigquery.Client,
               ):
    extensions = ['.java']
    super(javaDataset, self).__init__(client, "java", extensions)
    return

class pythonDataset(Dataset):
  """python Dataset"""
  @property
  def language(self):
    return "Python"

  def __init__(self,
               client: bigquery.Client,
               ):
    extensions = ['.py']
    super(pythonDataset, self).__init__(client, "python", extensions)
    return
