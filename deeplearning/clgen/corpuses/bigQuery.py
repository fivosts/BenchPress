"""BigQuery API to fetch github data."""
import os
import pathlib
import progressbar
import humanize

from google.cloud import bigquery
from eupy.native import logger as l

from deeplearning.clgen import bigQuery_database

languages = {
  'opencl': ['.cl'],
  'c'     : ['.c'],
  'cpp'   : ['.cc', '.cpp', '.cxx', '.c++'],
  'java'  : ['.java'],
  'python': ['.py'],
}

def fetch(path, lang: str = "opencl"):
  # Construct a BigQuery client object.
  config = bigquery.QueryJobConfig(allowLargeResults = True)
  config.allow_large_results = True
  l.getLogger().warn(config.allow_large_results)
  client = bigquery.Client(default_query_job_config = config)

  substr_command = ""
  for en, ext in enumerate(languages[lang]):
    if en == 0:
      substr_command = "substr(file.path, {}, {}) = '{}'".format(-len(ext), 1 + len(ext), ext)
    else:
      substr_command += " OR substr(file.path, {}, {}) = '{}'".format(-len(ext), 1 + len(ext), ext)

  count_query = """
  SELECT COUNT(*)
  FROM `bigquery-public-data.github_repos.files` as file
  WHERE {}
  """.format(substr_command)

  db_query = """
  SELECT file.repo_name, file.path, file.ref, file.mode, 
         file.id, file.symlink_target, contentfile.size, 
         contentfile.content, contentfile.binary, contentfile.copies
  FROM `bigquery-public-data.github_repos.contents` as contentfile
  INNER JOIN `bigquery-public-data.github_repos.files` as file ON file.id = contentfile.id AND {}
  """.format(substr_command)

  repo_query = """
  SELECT DISTINCT file.repo_name, file.ref
  FROM `bigquery-public-data.github_repos.files` as file
  WHERE {}
  """.format(substr_command)

  # TODO(developer): Set table_id to the ID of the table to create.

  # dataset_id = "{}.your_dataset".format(client.project)

  # Construct a full Dataset object to send to the API.
  # dataset = bigquery.Dataset(dataset_id)

  # TODO(developer): Specify the geographic location where the dataset should reside.
  # dataset.location = "US"

  # Send the dataset to the API for creation, with an explicit timeout.
  # Raises google.api_core.exceptions.Conflict if the Dataset already
  # exists within the project.
  # dataset = client.create_dataset(dataset, timeout=30)  # Make an API request.
  # print("Created dataset {}.{}".format(client.project, dataset.dataset_id))

  # table_id = "your-project.your_dataset.your_table_name"

  # schema = [
  #   bigquery.SchemaField("full_name", "STRING", mode="REQUIRED"),
  #   bigquery.SchemaField("age", "INTEGER", mode="REQUIRED"),
  # ]

  # table = bigquery.Table(table_id, schema=schema)
  # table = client.create_table(table)  # Make an API request.
  # print(
  #   "Created table {}.{}.{}".format(table.project, table.dataset_id, table.table_id)
  # )

  count_job = client.query(count_query)
  file_job  = client.query(db_query)
  repo_job  = client.query(db_query)

  for f, r in zip(count_job, repo_job):
    file_count = f[0]
    repo_count = r[0]

  l.getLogger().info("Fetching {} {} files from {} repos".format(
      humanize.intcomma(file_count), lang, humanize.intcomma(repo_count)
    )
  )

  url = "sqlite:///{}{}".format(path, "bqcorpus_{}.db".format(lang))
  db = bigQuery_database.bqDatabase(url)
  with db.Session(commit = True) as session, progressbar.ProgressBar(max_value = file_count) as bar:
    with progressbar.ProgressBar(max_value = file_count) as bar:
      for en,row in enumerate(file_job):
        contentfile = bigQuery_database.bqFile(
          **bigQuery_database.bqFile.FromArgs(en, row)
        )
        session.add(contentfile)
        bar.update(en)
    with progressbar.ProgressBar(max_value = repo_count) as bar:
      for en, row in enumerate(repo_job):
        repofile = bigQuery_database.bqRepo(
          **bigQuery_database.bqRepo.FromArgs(en, row)
        )
        session.add(repofile)
        bar.update(en)