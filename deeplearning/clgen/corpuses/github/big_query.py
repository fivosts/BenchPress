"""BigQuery API to fetch github data."""
import os
import pathlib
import progressbar
import humanize

from google.cloud import bigquery
from absl import flags

from deeplearning.clgen.corpuses.github import bigQuery_database
from deeplearning.clgen.corpuses.github import miner
from eupy.native import logger as l

FLAGS = flags.FLAGS

# languages = {
#   'opencl': ['.cl'],
#   'c'     : ['.c'],
#   'cpp'   : ['.cc', '.cpp', '.cxx', '.c++'],
#   'java'  : ['.java'],
#   'python': ['.py'],
# }

class BigQuery(miner.GithubMiner):
  def __init__(self, config):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(pathlib.Path(config.credentials, must_exist = True))
    self.cache_path = pathlib.Path(config.path, must_exist = False, parents = True)
    self.dataset    = datasets.FromArgs(self.language, config.data_format)
    return

def fetch(path, lang: str = None):
  # Construct a BigQuery client object.
  config = bigquery.QueryJobConfig(allowLargeResults = True)
  config.allow_large_results = True
  l.getLogger().warn(config.allow_large_results)
  client = bigquery.Client(default_query_job_config = config)

  substr_command = ""
  if lang is not None:
    for en, ext in enumerate(languages[lang]):
      if en == 0:
        substr_command = "substr(file.path, {}, {}) = '{}'".format(-len(ext), 1 + len(ext), ext)
      else:
        substr_command += " OR substr(file.path, {}, {}) = '{}'".format(-len(ext), 1 + len(ext), ext)

  count_query = """
  SELECT COUNT(*)
  FROM `bigquery-public-data.github_repos.files` as file
  {}
  """.format("WHERE " + substr_command if lang is not None else "")

  db_query = """
  SELECT file.repo_name, file.path, file.ref, file.mode, 
         file.id, file.symlink_target, contentfile.size, 
         contentfile.content, contentfile.binary, contentfile.copies
  FROM `bigquery-public-data.github_repos.contents` as contentfile
  INNER JOIN `bigquery-public-data.github_repos.files` as file ON file.id = contentfile.id {}
  """.format("AND " + substr_command if lang is not None else "")

  repo_query = """
  SELECT DISTINCT file.repo_name, file.ref
  FROM `bigquery-public-data.github_repos.files` as file
  {}
  """.format("WHERE " + substr_command if lang is not None else "")

  if lang is None:
    lang = "All"
  # TODO(developer): Set table_id to the ID of the table to create.
  # Construct a full Dataset object to send to the API.
  # TODO(developer): Specify the geographic location where the dataset should reside.

  dataset_id = "{}.clgen_{}".format(client.project, lang)
  dataset = bigquery.Dataset(dataset_id)
  dataset.location = "US"


  # # Send the dataset to the API for creation, with an explicit timeout.
  # # Raises google.api_core.exceptions.Conflict if the Dataset already
  # # exists within the project.
  dataset = client.get_dataset(dataset_id)
  print(dataset)
  exit()
  # dataset = client.create_dataset(dataset, timeout=30)  # Make an API request.
  print("Created dataset {}.{}".format(client.project, dataset.dataset_id))
  exit()
  table_id = "{}.clgen_{}.bq_contentfiles".format(client.project, lang)

  schema = [
    bigquery.SchemaField("full_name", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("age", "INTEGER", mode="REQUIRED"),
  ]

  table = bigquery.Table(table_id, schema=schema)
  table = client.create_table(table)  # Make an API request.
  print(
    "Created table {}.{}.{}".format(table.project, table.dataset_id, table.table_id)
  )

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
    r = [
      "total_contentfiles: {}".format(db.file_count),
      "total_repositories: {}".format(db.repo_count),
    ]
    exists = session.query(
        bigQuery_database.bqData.key
      ).filter_by(key = lang).scalar() is not None
    if exists:
      entry = session.query(
          bigQuery_database.bqData
        ).filter_by(key = lang).first()
      entry.value = "\n".join(r)
    else:
      session.add(
        bigQuery_database.bqData(key = lang, value = "\n".join(r))
      )