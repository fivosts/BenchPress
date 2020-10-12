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

    self.dataset = datasets.Dataset.FromArgs(self.client, self.language)
    self.storage = storage.Storage.FromArgs(self.cache_path, self.dataset.name, self.dataset.extension, config.data_format)
    return

  def fetch(self):

    # Filecount of requested file specifications.
    mainfile_count, otherfile_count = self.dataset.filecount

    # Get repository list of requested file specifications.
    mainrep_it, otherrep_it = self.dataset.repository_query()
    if mainrep_it is not None:
      main_repo_count = 0
      for en, mr in enumerate(mainrep_it):
        self.storage.save(bigQuery_database.bqRepo(
            **bigQuery_database.bqRepo.FromArgs(en, mr)
          )
        )
        main_repo_count = en

    if otherrep_it is not None:
      other_repo_count = 0
      for en, orep in enumerate(otherrep_it):
        self.storage.save(bigQuery_database.bqRepo(
            **bigQuery_database.bqRepo.FromArgs(en, orep)
          )
        )
        other_repo_count = en

    # Get contentfiles.
    mainf_it, otherf_it = self.dataset.contentfile_query()
    if mainf_it is not None:
      for en, mf in enumerate(mainf_it):
        self.storage.save(bigQuery_database.bqFile(
            **bigQuery_database.bqFile.FromArgs(en, mf)
          )
        )
    if otherf_it is not None:
      for en, of in enumerate(otherf_it):
        self.storage.save(bigQuery_database.bqFile(
            **bigQuery_database.bqFile.FromArgs(en, of)
          )
        )

    query_data = [
      "main_contentfiles : {}".format(mainfile_count),
      "other_contentfiles: {}".format(otherfile_count),
      "total_contentfiles: {}".format(mainfile_count + otherfile_count),
      "main_repositories : {}".format(0 if not mainrep_it else main_repo_count),
      "other_repositories: {}".format(0 if not otherrep_it else other_repo_count),
      "total_repositories: {}".format(0 if not (mainrep_it or otherrep_it) else main_repo_count + other_repo_count),
    ]
    self.storage.save(bigQuery_database.bqData(key = self.dataset.name, value = '\n'.join(r)))
    return

    # url = "sqlite:///{}{}".format(self.cache_path, "bqcorpus_{}.db".format(lang))
    # db = bigQuery_database.bqDatabase(url)
    # with db.Session(commit = True) as session, progressbar.ProgressBar(max_value = file_count) as bar:
    #   with progressbar.ProgressBar(max_value = file_count) as bar:

    #   r = [
    #     "total_contentfiles: {}".format(db.file_count),
    #     "total_repositories: {}".format(db.repo_count),
    #   ]
    #   exists = session.query(
    #       bigQuery_database.bqData.key
    #     ).filter_by(key = lang).scalar() is not None
    #   if exists:
    #     entry = session.query(
    #         bigQuery_database.bqData
    #       ).filter_by(key = lang).first()
    #     entry.value = "\n".join(r)
    #   else:
    #     session.add(
    #       bigQuery_database.bqData(key = lang, value = "\n".join(r))
    #     )
