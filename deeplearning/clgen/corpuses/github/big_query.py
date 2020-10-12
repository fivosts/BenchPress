"""BigQuery API to fetch github data."""
import os
import pathlib
import progressbar
import humanize
from google.cloud import bigquery

from deeplearning.clgen.proto import github_miner_pb2
from deeplearning.clgen.corpuses.github import bigQuery_database
from deeplearning.clgen.corpuses.github import miner
from eupy.native import logger as l

class BigQuery(miner.GithubMiner):
  def __init__(self,
               config: github_miner_pb2.GithubMiner
               ):

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(pathlib.Path(config.credentials, must_exist = True))
    self.cache_path = pathlib.Path(config.path, must_exist = False, parents = True)

    self.config  = bigquery.QueryJobConfig(allowLargeResults = True)
    self.config.allow_large_results = True

    self.client  = bigquery.Client(default_query_job_config = config)
    self.dataset = datasets.FromArgs(self.client, self.language, config.data_format)
    return

  def fetch(self):
    # Construct a BigQuery client object.

    url = "sqlite:///{}{}".format(self.cache_path, "bqcorpus_{}.db".format(lang))
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