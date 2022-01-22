"""Github mining configuration"""
import json
import os
import io
import re
import time
import requests
import functools
import sys
import typing
import pathlib
import github
import progressbar
import copy
import numpy as np
from absl import flags

from base64 import b64decode
from google.cloud import bigquery

from deeplearning.clgen.util import pbutil
from deeplearning.clgen.util import crypto
from deeplearning.clgen.proto import github_pb2
from deeplearning.clgen.github import datasets
from deeplearning.clgen.github import storage
from deeplearning.clgen.github import bigQuery_database

from deeplearning.clgen.util import logging as l

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
  "bq_force_update",
  False,
  "Select to force querying data in a seemingly updated satorage."
)

flags.DEFINE_string(
  "exclude_repos_from_db",
  None,
  "Specify repo-db to bypass repositories in recursive fetcher."
)

flags.DEFINE_string(
  "enhance_from_db",
  None,
  "Specify bq DB to enhance corpus with contentfiles"
)

flags.DEFINE_boolean(
  "remove_identical_files",
  False,
  "Select to load all files, calculate hashes and remove duplicates."
)

flags.DEFINE_boolean(
  "export_db",
  False,
  "Dumps bigquery database to folder of files."
)

class GithubMiner(object):
  """Base abstract class of a github miner"""

  @classmethod
  def FromConfig(cls, config: github_pb2.GithubMiner):
    """Constructs github miner from protobuf configuration."""
    try:
      pbutil.AssertFieldIsSet(config, "path")
      pbutil.AssertFieldIsSet(config, "data_format")
      pbutil.AssertFieldIsSet(config, "miner")

      if config.HasField("big_query"):
        pbutil.AssertFieldIsSet(config.big_query, "credentials")
        pbutil.AssertFieldConstraint(
          config.big_query,
          "language",
          lambda x: x in {'generic', 'opencl', 'c', 'cpp', 'java', 'python'},
          "language must be one of opencl, c, cpp, java, python. 'generic' for language agnostic queries.",
        )
        if config.big_query.HasField("export_corpus"):
          pbutil.AssertFieldIsSet(config.big_query.export_corpus, "data_format")
          pbutil.AssertFieldIsSet(config.big_query.export_corpus, "access_token")
        return BigQuery(config)
      elif config.HasField("recursive"):
        pbutil.AssertFieldIsSet(config.recursive, "access_token")
        pbutil.AssertFieldConstraint(
          config.recursive,
          "flush_limit_K",
          lambda x: x>0,
          "flush limit cannot be non-positive."
          )
        pbutil.AssertFieldConstraint(
          config.recursive,
          "corpus_size_K",
          lambda x: x >= -1,
          "corpus size must either be -1 or non-negative."
          )
        if config.data_format != github_pb2.GithubMiner.DataFormat.folder:
          raise NotImplementedError("RecursiveFetcher only stores files in local folder.")
        return RecursiveFetcher(config)
      else:
        raise SystemError("{} miner not recognized".format(config))
    except Exception as e:
      raise e

  def __init__(self):
    return

  def fetch(self) -> None:
    raise NotImplementedError("Abstract class")

class BigQuery(GithubMiner):
  def __init__(self,
               config: github_pb2.GithubMiner
               ):
    super(BigQuery, self).__init__()
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(pathlib.Path(config.big_query.credentials, must_exist = True))
    self.cache_path = pathlib.Path(config.path, must_exist = False).expanduser().resolve()
    self.cache_path.mkdir(exist_ok = True, parents = True)
    self.config = config

    l.logger().info("Initializing BigQuery miner in {}".format(self.cache_path))
    job_config = bigquery.QueryJobConfig(allowLargeResults = True)
    job_config.allow_large_results = True
    self.client = bigquery.Client(default_query_job_config = job_config)

    self.dataset = datasets.Dataset.FromArgs(self.client, self.config.big_query.language)
    self.storage = storage.Storage.FromArgs(self.cache_path, self.dataset.name, self.dataset.extension, self.config.data_format)
    return

  def fetch(self):
    if FLAGS.export_db:
      folder = self.cache_path / "export_files"
      folder.mkdir(exist_ok = True, parents = True)
      with progressbar.ProgressBar(max_value = self.storage.maincount, prefix = "Export") as bar:
        for mf in bar(self.storage.db.main_files):
          with open(folder / mf.id, 'w') as outf:
            outf.write(mf.content)
      return
    self._query_github()
    if self.config.big_query.export_corpus.inline_headers:
      self._export_corpus()
    return

  def _query_github(self) -> None:
    """Apply bigQuery requests to get all contentfiles"""
    with self.storage as st:

      if st.content_data is not None and not FLAGS.bq_force_update:
        l.logger().info("Query storage has been updated. Skipping...")
        return

      mainf_it, otherf_it = self.dataset.contentfile_query()

      if mainf_it:
        with progressbar.ProgressBar(max_value = mainf_it.total_rows, prefix = "Main Files") as bar:
          try:
            for mf in bar(mainf_it):
              st.save(
                bigQuery_database.bqMainFile(**bigQuery_database.bqMainFile.FromArgs(mf))
              )
          except KeyboardInterrupt:
            pass
        st.flush()

      if otherf_it:
        with progressbar.ProgressBar(max_value = otherf_it.total_rows, prefix = "Other Files") as bar:
          try:
            for of in bar(otherf_it):
              st.save(
                bigQuery_database.bqOtherFile(**bigQuery_database.bqOtherFile.FromArgs(of))
              )
          except KeyboardInterrupt:
            pass
        st.flush()

      # Get repository list of requested file specifications.
      # If contentfile_query has taken place, use cached results instead of re-querying.
      if mainf_it or otherf_it:
        mainrep_it, otherrep_it = None, None
      else:
        mainrep_it, otherrep_it = self.dataset.repository_query()

      main_repo_count = None
      if mainrep_it:
        with progressbar.ProgressBar(max_value = mainrep_it.total_rows, prefix = "Main Repos") as bar:
          for mr in bar(mainrep_it):
            st.save(
              bigQuery_database.bqRepo(**bigQuery_database.bqRepo.FromArgs(st.repocount, mr))
            )
        main_repo_count = st.repocount
        st.flush()

      other_repo_count = None
      if otherrep_it:
        with progressbar.ProgressBar(max_value = otherrep_it.total_rows, prefix = "Other Repos") as bar:
          for orep in bar(otherrep_it):
            st.save(
              bigQuery_database.bqRepo(**bigQuery_database.bqRepo.FromArgs(st.repocount, orep))
            )
        other_repo_count = st.repocount - main_repo_count
        st.flush()

      # Filecount of requested file specifications.
      # Use cached results if contentfile has taken place.
      if mainf_it or otherf_it:
        self.dataset.filecount = (mainf_it.total_rows if mainf_it else 0, otherf_it.total_rows if otherf_it else 0)
      mainfile_count, otherfile_count = self.dataset.filecount

      if main_repo_count is None:
        main_repo_count = st.main_repocount
      if other_repo_count is None:
        other_repo_count = st.other_repocount

      query_data = [
        "main_contentfiles : {}".format(mainfile_count),
        "other_contentfiles: {}".format(otherfile_count),
        "total_contentfiles: {}".format(mainfile_count + otherfile_count),
        "",
        "main_repositories : {}".format(main_repo_count),
        "other_repositories: {}".format(other_repo_count),
        "total_repositories: {}".format(st.repocount),
      ]
      st.save(bigQuery_database.bqData(key = self.dataset.name, value = '\n'.join(query_data)))
    return

  def _export_corpus(self) -> None:
    """
    Get all raw files requested from BQ and export them to CLGEN corpus.

    The most important aspect is inlining includes into the source files.

    In case the selected storage type is SQL DB, all needed header files
    will be found in bq_header_contentfiles table and will be drawn from there.
    The original storage DB can be diminished in size, by deleting the header
    files that were not eventually used.
    """
    export_storage = storage.Storage.FromArgs(
      self.cache_path,
      "export_{}".format(self.dataset.name),
      self.dataset.extension,
      self.config.data_format
    )

    g = github.Github(self.config.big_query.export_corpus.access_token)
    iterated_history = export_storage.db.mainfile_entries
    iterated_history.update(export_storage.db.otherfile_entries)

    with export_storage as st:
      with progressbar.ProgressBar(max_value = self.storage.maincount) as bar:
        for cf in bar(self.storage.mainfiles):
          if (cf.repo_name, cf.path) in iterated_history:
            continue
          try:
            rem = g.get_rate_limit().rate.remaining
            while rem < 100:
              time.sleep(1)
              print('\r\033[KWaiting on rate limit: {}'.format(rem), sep='', end='')
              sys.stdout.flush()
              rem = g.get_rate_limit().rate.remaining
            repo = g.get_repo(cf.repo_name)
            cf = self._inline_headers(repo, cf.ref, cf)
            st.save(
              bigQuery_database.bqMainFile(**bigQuery_database.bqMainFile.FromArgs(cf.ToJSONDict()))
            )
          except Exception as e:
            st.flush()
            if "404" in str(e):
              l.logger().error("Not found: {}-{}".format(cf.repo_name, cf.path))
              st.save(
                bigQuery_database.bqMainFile(**bigQuery_database.bqMainFile.FromArgs(cf.ToJSONDict()))
              )
            else:
              raise e

      with progressbar.ProgressBar(max_value = self.storage.othercount) as bar:
        try:
          for cf in bar(self.storage.otherfiles):
            if (cf.repo_name, cf.path) in iterated_history:
              continue
            ### Rate limit
            rem = g.get_rate_limit().rate.remaining
            while rem < 100:
              time.sleep(1)
              print("Waiting on rate limit: {}".format(rem), sep='', end='')
              sys.stdout.flush()
              rem = g.get_rate_limit().rate.remaining
            ### Save file if repo not found
            try:
              repo = g.get_repo(cf.repo_name)
            except github.GithubException as e:
              if "Not Found" in str(e):
                st.save(
                  bigQuery_database.bqMainFile(**bigQuery_database.bqOtherFile.FromArgs(cf.ToJSONDict()))
                )
                continue
              else:
                raise e

            cf = self._inline_headers(repo, cf.ref, cf)
            st.save(
              bigQuery_database.bqMainFile(**bigQuery_database.bqOtherFile.FromArgs(cf.ToJSONDict()))
            )
        except Exception as e:
          st.flush()
          raise e
    return

  def _inline_headers(self,
                      repo   : github.Repository.Repository,
                      ref    : str,
                      content: bigQuery_database.bqFile,
                      ) -> str:
    ## Do the same as inlineHeaders
    #  1. Parse file for #include
    #  2. Resolve include path
    #  3. Ping DB to get it
    #  4. Do BFS on included


    def get_included_file(file_path      : pathlib.Path,
                          incl_path      : pathlib.Path,
                          ) -> github.ContentFile.ContentFile:

      parent_folder = file_path.parent
      if parent_folder == file_path:
        return None
      folder_files = repo.get_contents(str(parent_folder), ref = ref)
      while folder_files:
        file = folder_files.pop(0)
        if file.path == file_path:
          continue
        elif file.type == "dir":
          folder_files.extend(repo.get_contents(file.path, ref = ref))
        elif file.path.endswith(str(incl_path)):
          return file
      return get_included_file(parent_folder, incl_path)

    inlined_cf    = []
    inlined_paths = set()
    inlined_paths.add(content.path)
    include_exist = True
    while include_exist:
      include_exist = False

      for line in content.content.split('\n'):

        match = re.match(re.compile('\w*#include ["<](.*)[">]'), line)
        if match:
          include_exist = True
          include_path  = match.group(1)

          # Try and resolve relative paths
          include_path = pathlib.Path(include_path.replace('../', ''))            
          incl_file = get_included_file(pathlib.Path(content.path), include_path)

          if incl_file and incl_file.path not in inlined_paths:
            inlined_paths.add(incl_file.path)
            inlined_cf.append("// [FETCH] included: {}\n".format(line))
            if incl_file.size < 1*1000*1000:
              inlined_cf.append(incl_file.content)
            else:
              response = json.loads(requests.get(
                incl_file.git_url, headers={'Authorization': 'token {}'.format(self.config.big_query.export_corpus.access_token)}
              ).content.decode('utf-8'))
              incl_cf = b64decode(response['content']).decode('utf-8')
              inlined_cf.append(incl_cf)
            inlined_cf.append('// [FETCH] eof({})'.format(line))
          else:
            if not incl_file:
              inlined_cf.append('// [FETCH] didnt find: {}'.format(line))
            else:
              inlined_cf.append('// [FETCH] skipped: {}'.format(line))

        else:
          inlined_cf.append(line)
      content.content = '\n'.join(inlined_cf)
      inlined_cf = []

    return content

class RecursiveFetcher(GithubMiner):
  """GitHub API wrapper to pull from github a fresh corpus of OpenCL kernels"""

  class GithubRepoHandler():
    """Repo manager for recursive fetcher"""

    class GithubRepo():
      """Class representation of a single github Repo."""
      def __init__(self, **kwargs):
        # url of a repo is immutable.
        self.url = kwargs.get('url')
        if kwargs:
          self.update(**kwargs)
        return

      def update(self,
                 url          : str,
                 owner        : str,
                 name         : str,
                 fork         : int,
                 stars        : str,
                 contributors : int,
                 forks        : str,
                 created_at   : str,
                 updated_at   : str):

        if url != self.url:
          raise ValueError("Updated url of already existent repo does not match.")
        self.owner        = owner
        self.name         = name
        self.fork         = fork
        self.stars        = stars
        self.contributors = contributors
        self.forks        = forks
        self.created_at   = created_at
        self.updated_at   = updated_at
        return

    class GithubFile():
      """Class representation of a single github file."""
      def __init__(self, **kwargs):
        # url of a file is immutable
        self.url  = kwargs.get('url')
        self.size = 0
        if kwargs:
          self.update(**kwargs)

      def update(self,
                 url      : str,
                 contents : str,
                 path     : str,
                 repo_url : str,
                 sha      : str,
                 size     : int):

        if url != self.url:
          raise ValueError("Updated url of already existent file does not match.")

        self.contents   = contents
        self.path       = path
        self.repo_url   = repo_url
        self.sha        = sha
        if self.size != 0:
          current_size  = size - self.size
        else:
          current_size  = size
        self.size       = size
        return current_size

    def __init__(self, 
                 corpus_path: str,
                 corpus_size: int,
                 flush_limit: int,
                 ):

      ## Use this to read a json file with all current sha files
      ## And of course to append the json file every time you flush
      ## ..and to flush
      self.cache_path              = corpus_path
      self.stored_file_idx         = "record.json"

      self.updated_length          = 0

      self._scraped_repos          = {}
      self._stored_repos           = {}
      self._scraped_files          = {}

      self.repos_new_counter       = 0
      self.repos_modified_counter  = 0
      self.repos_unchanged_counter = 0
      self.repos_stored_counter    = 0

      self.files_new_counter       = 0
      self.files_modified_counter  = 0
      self.files_unchanged_counter = 0
      self.file_size_counter       = 0
      self.file_size_limit         = flush_limit

      self.collectHistory()
      self.is_finished             = False if (corpus_size // 1000) == -1 else (self.updated_length >= corpus_size)
      return

    def collectHistory(self) -> None:
      storage_file = os.path.join(self.cache_path, self.stored_file_idx)
      if os.path.isfile(storage_file):
        with open(storage_file, 'r') as f:
          try:
            data                = json.load(f)
            assert len(data)    == 2, "Wrong format of kernel history provided"
            self._stored_repos  = data[0]
            self.updated_length = data[1]['total_files']
          except json.JSONDecodeError:
            l.logger().warn("Problem encountered with reading kernel file record.")
      return

    def appendHistory(self) -> None:
      storage_file = os.path.join(self.cache_path, self.stored_file_idx)
      with open(storage_file, 'w') as f:
        json.dump(
          [self._stored_repos, 
           {'total_files': self.updated_length + copy.deepcopy(len(self._scraped_files))}],
          f, 
          indent = 2)
      return

    def is_repo_updated(self, url, updated_at) -> bool:
      if url in self._scraped_repos and self._scraped_repos[url].updated_at == updated_at:
        self.repos_unchanged_counter += 1
        return True
      elif url in self._stored_repos:# and self._stored_repos[url] == updated_at:
        self.repos_stored_counter    += 1
        return True
      return False
   
    def is_file_updated(self, url, sha) -> bool:
      if url in self._scraped_files and self._scraped_files[url].sha == sha:
        self.files_unchanged_counter += 1
        return True
      return False

    def update_file(self, **kwargs) -> bool:

      url = kwargs.get('url')
      if url in self._scraped_files:
        self.file_size_counter      += self._scraped_files[url].update(**kwargs)
        self.files_modified_counter += 1
      else:
        self._scraped_files[url]    =  RecursiveFetcher.GithubRepoHandler.GithubFile(**kwargs)
        self.files_new_counter      += 1
        self.file_size_counter      += kwargs.get('size')

      if self.file_size_counter >= self.file_size_limit:
        l.logger().warn("time to flush!")
        self.Flush()
        self.collectHistory()
        self.file_size_counter = 0

      return True

    def update_repo(self, **kwargs) -> bool:

      url = kwargs.get('url')
      l.logger().info("Add: {}".format(url))
      if url in self._scraped_repos:
        self._scraped_repos[url].update(**kwargs)
        self.repos_modified_counter += 1
      else:
        self._scraped_repos[url]    =  RecursiveFetcher.GithubRepoHandler.GithubRepo(**kwargs)
        self.repos_new_counter      += 1
      return True

    def Flush(self) -> None:
      for idx, file in enumerate(self._scraped_files):
        if self._scraped_files[file].repo_url in self._scraped_repos:
          with open(os.path.join(self.cache_path, "{}.cl".format(idx + self.updated_length)), 'w') as f:
            f.write(self._scraped_files[file].contents)
      for repo in self._scraped_repos:
        self._stored_repos[repo] = self._scraped_repos[repo].updated_at
      self.appendHistory()
      self._scraped_repos.clear()
      self._scraped_files.clear()
      self.file_size_counter  = 0
      return

    def print_counters(self) -> None:
      """
      Print analytics counters.
      """
      print('\r\033[Kfiles: new: ',  self.files_new_counter,
          ', modified: ',            self.files_modified_counter,
          ', mem_size: ',            self.file_size_counter, 'B',
          sep='', end='')


  def __init__(self,
               config: github_pb2.GithubMiner
               ):
    self.cache_path = pathlib.Path(config.path, must_exist = False).expanduser().resolve()
    self.cache_path.mkdir(exist_ok = True, parents = True)
    l.logger().info("Github fetcher initialized: {}".format(self.cache_path))

    self.token           = config.recursive.access_token
    self.repo_handler    = RecursiveFetcher.GithubRepoHandler(
      self.cache_path, 
      config.recursive.corpus_size_K * 1000,
      config.recursive.flush_limit_K * 1000,
    )

    self.current_status  = ""
    self.errors_counter  = 0
    return

  def print_counters(self) -> None:
    self.repo_handler.print_counters()
    print('. errors: ', self.errors_counter,
          '. ',        self.current_status[0:80],
        sep='', end='')
    sys.stdout.flush()

  def fetch(self) -> None:
    """
    Download all of the OpenCL on GitHub (!)

    Shortcomings of this appraoch:
      * Only includes exclusively OpenCL files, no inline strings.
      * Occasionally (< 1%) can't find headers to include.

    """

    # ### Dummy code to compare similarities of recursive corpus and bq CL corpus.
    # db = bigQuery_database.bqDatabase("sqlite:////home/fivosts/PhD/Code/clgen/bq_corpus/exported_clgen_opencl_github.db")
    # self.bq_repos = set()
    # with db.Session() as s:
    #   for r in s.query(bigQuery_database.bqMainFile.repo_name):
    #     self.bq_repos.add(r[0])
    #   for r in s.query(bigQuery_database.bqOtherFile.repo_name):
    #     self.bq_repos.add(r[0])

    # with open("/home/fivosts/Downloads/record.json", 'r') as f:
    #   chris = json.load(f)

    # chris_repos = set(x.replace('https://api.github.com/repos/', '') for x, v in chris[0].items())

    # common_repos = set()
    # for r in chris_repos:
    #   if r in self.bq_repos:
    #     common_repos.add(r)

    # l.logger().warn(len(common_repos))

    # file_count = 0
    # with db.Session() as s:
    #   for r in s.query(bigQuery_database.bqMainFile).all():
    #     if r.repo_name in common_repos:
    #       file_count += 1
    #   for r in s.query(bigQuery_database.bqOtherFile).all():
    #     if r.repo_name in common_repos:
    #       file_count += 1

    # l.logger().info(file_count)
    # exit()

    if FLAGS.remove_identical_files:
      if FLAGS.enhance_from_db:
        self.enhance_from_db(pathlib.Path(FLAGS.enhance_from_db).resolve())
      self.remove_identical_files()
      return

    if FLAGS.exclude_repos_from_db:
      db = bigQuery_database.bqDatabase("sqlite:///{}".format(pathlib.Path(FLAGS.exclude_repos_from_db).resolve()))
      self.db_excluded_repos = set()
      with db.Session() as s:
        for r in s.query(bigQuery_database.bqRepo.repo_name):
          self.db_excluded_repos.add(r[0])

    g = github.Github(self.token)
    handle_repo = functools.partial(self.process_repo, g)

    # fetch the repositories to iterate over. Since opencl isn't
    # treated as a first-class language by GitHub, we can't use the
    # 'language=' keyword for queries, so instead we through a much
    # wider net and filter the results afterwards.
    query_terms = [
      'opencl',
      'cl',
      'khronos',
      'gpu',
      'gpgpu',
      'cuda',
      'amd',
      'nvidia',
      'heterogeneous',
      'language:C',
      'language:C++',
      'language:LLVM',
    ]
    try:
      for query in query_terms:
        # forks are okay - we use checksums to ensure uniqueness in
        # final dataset
        repos = g.search_repositories(query + ' fork:true sort:stars')

        for repo in repos:
          self.cached_includes = {}
          if self.repo_handler.is_finished:
            self.print_counters()
            self.repo_handler.Flush()
            l.logger().info("Finished gathering Github kernels.")
            return

          repo_modified = handle_repo(repo)

          # do nothing unless the repo is new or modified
          if not repo_modified:
            continue

          handle_file = functools.partial(self.process_file, g, repo)

          # iterate over the entire git tree of the repo's default
          # branch (usually 'master'). If a file ends with the .cl
          # extension, check to see if we already have it, else download
          # it
          try:
            branch = repo.default_branch
            tree_iterator = repo.get_git_tree(branch, recursive=True).tree
            for f in tree_iterator:
              try:
                handle_file(f)
              except UnicodeError:
                self.errors_counter += 1
                pass
              except Exception as e:
                raise e
            try:
              contributors = len([x for x in repo.get_contributors()])
            except github.GithubException:
              contributors = -1
            self.repo_handler.update_repo(
              url          = repo.url,
              owner        = repo.owner.email,
              name         = repo.name,
              fork         = 1 if repo.fork else 0,
              stars        = repo.stargazers_count,
              contributors = contributors,
              forks        = repo.forks,
              created_at   = repo.created_at,
              updated_at   = str(repo.updated_at)
            )
          except github.GithubException:
            # do nothing in case of error (such as an empty repo)
            pass
    except KeyboardInterrupt:
      # Don't gather any more files
      pass
    except Exception as e:
      self.errors_counter += 1
      self.repo_handler.Flush()
      raise e

    self.print_counters()
    self.repo_handler.Flush()
    l.logger().info("Finished gathering Github kernels.")
    return

  def process_repo(self, g, repo) -> bool:
    """
    GitHub repository handler.

    Determines if a repository needs to be scraped. There are two cases for
    this:
      * The repository has not already been visited.
      * The repository has been modified since it was last visited.

    Parameters
    ----------
    g
      GitHub connection.
    repo
      Repository.
    Returns
    -------
    bool
      True if repository should be scraped, else False.
    """
    self.rate_limit(g)
    self.current_status = repo.name
    self.print_counters()

    if FLAGS.exclude_repos_from_db and repo.full_name in self.db_excluded_repos:
      return False

    if self.repo_handler.is_repo_updated(repo.url, str(repo.updated_at)):
      # Timestamp of already scraped repo matches, so nothing to do.
      return False
    return True

  def process_file(self, g, repo, file) -> bool:
    """
    GitHub file handler.

    Parameters
    ----------
    g
      GitHub connection.
    repo
      Repository.
    file
      File.

    Returns
    -------
    bool
      True on success, else False.
    """
    # We're only interested in OpenCL files.
    if not (file.path.endswith('.cl') or file.path.endswith('.ocl')):
      return

    url = file.url
    sha = file.sha
    path = file.path
    self.current_status = repo.name + '/' + path
    self.print_counters()

    if self.repo_handler.is_file_updated(url, sha):
      # Do nothing unless checksums don't match
      return False

    repo_url = repo.url
    contents, _ = self.download_file(g, repo, url, [])
    size     = file.size or 0

    self.repo_handler.update_file(
      url = url, contents = contents, path = path,
      sha = sha, repo_url = repo_url, size = size,
    )
    return True

  def download_file(self, g, repo, url: str, stack: typing.List[str]) -> typing.Tuple[str, typing.List[str]]:
    """
    Fetch file from GitHub.

    Recursively downloads and inlines headers.

    Parameters
    ----------
    repo
      Repository.
    url : str
      Path.
    stack : List[str]
      URL stack.

    Returns
    -------
    str
      File contents.
    """
    # Recursion stack
    stack.append(url)
    exc_idx = 0
    while True:
      self.rate_limit(g)
      try:
        response = json.loads(requests.get(
          url,
          headers={
            'Authorization': 'token ' + str(self.token)
          }
        ).content.decode('utf-8'))
        if 'content' in response:
          src = b64decode(response['content']).decode('utf-8')
        else:
          src = ""
        break
      except requests.exceptions.RequestException as e:
        if exc_idx == 0:
          l.logger().error(e)
        exc_idx += 1
        time.sleep(10)

    outlines = []
    for line in src.split('\n'):
      match = re.match(re.compile('\w*#include ["<](.*)[">]'), line)
      if match:
        include_name = match.group(1)

        # Try and resolve relative paths
        include_name = include_name.replace('../', '')

        branch = repo.default_branch
        tree_iterator = repo.get_git_tree(branch, recursive=True).tree
        include_url = ''
        for f in tree_iterator:
          if f.path.endswith(include_name):
            include_url = f.url
            break

        if include_url and include_url not in stack:
          if include_url not in self.cached_includes:
            self.cached_includes[include_url], stack = self.download_file(g, repo, include_url, stack)

          outlines.append("// [FETCH] included: {}\n".format(line))
          outlines.append(self.cached_includes[include_url])
          outlines.append('// [FETCH] eof({})'.format(line))
        else:
          if not include_url:
            outlines.append('// [FETCH] didnt find: {}'.format(line))
          else:
            outlines.append('// [FETCH] skipped: {}'.format(line))
      else:
        outlines.append(line)
    return '\n'.join(outlines), stack

  def rate_limit(self, g) -> None:
    """
    Block on GitHub rate limit.

    Parameters
    ----------
    g
      GitHub connection.
    """
    remaining = g.get_rate_limit().rate.remaining
    while remaining < 100:
      time.sleep(1)
      self.current_status = 'WAITING ON RATE LIMIT: {}'.format(remaining)
      self.print_counters()
      remaining = g.get_rate_limit().rate.remaining

  def remove_identical_files(self) -> None:

    l.logger().info("Removing duplicate files from mined corpus...")
    if os.path.isfile(str(self.cache_path / "record.json")):
      with open(self.cache_path / "record.json", 'r') as f:
        data = json.load(f)
        repos  = data[0]
        length = data[1]['total_files']

    cache_map = {}
    for i in range(length):
      with open(self.cache_path / "{}.cl".format(i), 'r') as f:
        cf = f.read()
        cf_hash = crypto.sha256_str(cf)
        if cf_hash not in cache_map:
          cache_map[cf_hash] = cf

    new_path = self.cache_path / "distinct_corpus"
    new_path.mkdir(exist_ok = True, parents = True)
    for k, v in cache_map.items():
      with open(new_path / "{}.cl".format(k), 'w') as f:
        f.write(v)

    with open(new_path / "record.json", 'w') as f:
      data[1]['total_files'] = len(cache_map)
      json.dump(data, f, indent = 2)
    return

  def enhance_from_db(self, db_path: pathlib.Path) -> None:
    l.logger().info("Enhancing dataset with {}".format(db_path.name))
    if not db_path.exists():
      l.logger().warn("{} db not found. Returning...".format(db_path))
    db = bigQuery_database.bqDatabase("sqlite:///{}".format(db_path))
    contentfiles  = [cf.content for cf in db.main_files ]
    contentfiles += [cf.content for cf in db.other_files]

    if os.path.isfile(str(self.cache_path / "record.json")):
      with open(self.cache_path / "record.json", 'r') as f:
        data = json.load(f)
        length = data[1]['total_files']
    else:
      l.logger().warn("record.json not found. Returning...")
      return

    for cf in contentfiles:
      with open(self.cache_path / "{}.cl".format(length), 'w') as f:
        f.write(cf)
      length += 1
    with open(self.cache_path / "record.json", 'w') as f:
      data[1]['total_files'] = length
      json.dump(data, f, indent = 2)
    return
