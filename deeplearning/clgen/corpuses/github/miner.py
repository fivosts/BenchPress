"""Github mining configuration"""
import json
import os
import io
import re
import time
import requests
import sys
import typing
import pathlib
import github
import copy

from base64 import b64decode
from functools import partial
from google.cloud import bigquery

from deeplearning.clgen.util import pbutil
from deeplearning.clgen.proto import github_pb2
from deeplearning.clgen.corpuses.github import datasets
from deeplearning.clgen.corpuses.github import storage

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
        return BigQuery(config)
      elif config.HasField("recursive"):
        pbutil.AssertFieldConstraint(
          config.recursive,
          "flush_limit_K",
          lambda x: x>0,
          "flush limit cannot be non-positive."
          )
        pbutil.AssertFieldConstraint(
          config.recursive,
          "corpus_size_K",
          lambda x: x>0,
          "corpus size cannot be non-positive."
          )
        if config.data_format != config.GithubMiner.DataFormat.folder:
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
    self.cache_path = pathlib.Path(config.path, must_exist = False, parents = True).absolute()

    job_config = bigquery.QueryJobConfig(allowLargeResults = True)
    job_config.allow_large_results = True
    self.client = bigquery.Client(default_query_job_config = job_config)

    self.dataset = datasets.Dataset.FromArgs(self.client, config.big_query.language)
    self.storage = storage.Storage.FromArgs(self.cache_path, self.dataset.name, self.dataset.extension, config.data_format)
    return

  def fetch(self):

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

    # Get repository list of requested file specifications.
    # If contentfile_query has taken place, use cached results instead of re-querying.
    if not mainf_it and not otherf_it:
      mainrep_it, otherrep_it = self.dataset.repository_query()
    else:
      mainrep_it, otherrep_it = mainf_it, otherf_it

    if mainrep_it is not None:
      for en, mr in enumerate(mainrep_it):
        self.storage.save(bigQuery_database.bqRepo(
            **bigQuery_database.bqRepo.FromArgs(en, mr)
          )
        )
      main_repo_count = self.storage.repo_count

    if otherrep_it is not None:
      for en, orep in enumerate(otherrep_it):
        self.storage.save(bigQuery_database.bqRepo(
            **bigQuery_database.bqRepo.FromArgs(en, orep)
          )
        )
      other_repo_count = self.storage.repo_count - (main_repo_count or 0)

    # Filecount of requested file specifications.
    # Use cached results if contentfile has taken place.
    if not mainf_it and not otherf_it:
      self.dataset.filecount = (mainf_it.page.num_items, otherf_it.page.num_items or 0)
    mainfile_count, otherfile_count = self.dataset.filecount

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
      self.corpus_path              = corpus_path
      self.stored_file_idx          = "record.json"

      self.updated_length           = 0

      self._scraped_repos           = {}
      self._stored_repos            = {}
      self._scraped_files           = {}

      self.repos_new_counter        = 0
      self.repos_modified_counter   = 0
      self.repos_unchanged_counter  = 0
      self.repos_stored_counter     = 0

      self.files_new_counter        = 0
      self.files_modified_counter   = 0
      self.files_unchanged_counter  = 0
      self.file_size_counter        = 0
      self.file_size_limit          = flush_limit

      self.collectHistory()
      self.is_finished              = False if corpus_size is None else (self.updated_length >= corpus_size)
      return

    def collectHistory(self):
      storage_file = os.path.join(self.corpus_path, self.stored_file_idx)
      if os.path.isfile(storage_file):
        with open(storage_file, 'r') as f:
          try:
            data                = json.load(f)
            assert len(data)    == 2, "Wrong format of kernel history provided"
            self._stored_repos  = data[0]
            self.updated_length = data[1]['total_files']
          except json.JSONDecodeError:
            l.getLogger().warn("Problem encountered with reading kernel file record.")
      return

    def appendHistory(self):
      storage_file = os.path.join(self.corpus_path, self.stored_file_idx)
      with open(storage_file, 'w') as f:
        json.dump(
          [self._stored_repos, 
           {'total_files': self.updated_length + copy.deepcopy(len(self._scraped_files))}],
          f, 
          indent = 2)
      return

    def is_repo_updated(self, url, updated_at):
      if url in self._scraped_repos and self._scraped_repos[url].updated_at == updated_at:
        self.repos_unchanged_counter += 1
        return True
      elif url in self._stored_repos:# and self._stored_repos[url] == updated_at:
        self.repos_stored_counter    += 1
        return True
      return False
   
    def is_file_updated(self, url, sha):
      if url in self._scraped_files and self._scraped_files[url].sha == sha:
        self.files_unchanged_counter += 1
        return True
      return False

    def update_file(self, **kwargs):

      url = kwargs.get('url')
      if url in self._scraped_files:
        self.file_size_counter      += self._scraped_files[url].update(**kwargs)
        self.files_modified_counter += 1
      else:
        self._scraped_files[url]    =  GithubFile(**kwargs)
        self.files_new_counter      += 1
        self.file_size_counter      += kwargs.get('size')

      if self.file_size_counter >= self.file_size_limit:
        l.getLogger().warn("time to flush!")
        self.Flush()
        self.collectHistory()
        self.file_size_counter = 0

      return True

    def update_repo(self, **kwargs):

      url = kwargs.get('url')
      if url in self._scraped_repos:
        self._scraped_repos[url].update(**kwargs)
        self.repos_modified_counter += 1
      else:
        self._scraped_repos[url]    =  GithubRepo(**kwargs)
        self.repos_new_counter      += 1
      return True

    def Flush(self):
      for idx, file in enumerate(self._scraped_files):
        with open(os.path.join(self.corpus_path, "{}.cl".format(idx + self.updated_length)), 'w') as f:
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
    self.corpus_path     = config.path
    git_credentials = {
      'GITHUB_USERNAME'  : None,
      'GITHUB_PW'        : None,
      'GITHUB_TOKEN'     : None,
    }
    l.getLogger().info("Github fetcher initialized: {}".format(corpus_path))

    if not all(k in os.environ for k in git_credentials.keys()):
      l.getLogger().warn("Export github credentials as environment variables to speed up the process")

    for key in git_credentials:
      if key in os.environ:
        git_credentials[key] = os.environ[key]
      else:
        git_credentials[key] = input("{}: ".format(key))
        os.environ[key]      = git_credentials[key]

    self.username        = git_credentials['GITHUB_USERNAME']
    self.password        = git_credentials['GITHUB_PW']
    self.token           = git_credentials['GITHUB_TOKEN']
    self.repo_handler    = GithubRepoHandler(
      self.corpus_path, 
      config.recursive.corpus_size_K * 1000,
      config.recursive.flush_limit_K * 1000,
    )

    self.current_status  = ""
    self.errors_counter  = 0
    return

  def print_counters(self):
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
    g = github.Github(self.username, self.password)
    handle_repo = partial(self.process_repo, g)

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
      'heterogeneous'
    ]
    try:
      for query in query_terms:
        # forks are okay - we use checksums to ensure uniqueness in
        # final dataset
        repos = g.search_repositories(query + ' fork:true sort:stars')

        for repo in repos:
          if self.repo_handler.is_finished:
            self.print_counters()
            self.repo_handler.Flush()
            l.getLogger().info("Finished gathering Github kernels.")
            return

          repo_modified = handle_repo(repo)

          # do nothing unless the repo is new or modified
          if not repo_modified:
            continue

          handle_file = partial(self.process_file, g, repo)

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
    l.getLogger().info("Finished gathering Github kernels.")
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

    url                   = repo.url
    name                  = repo.name
    updated_at            = str(repo.updated_at)
    self.current_status   = name
    self.print_counters()

    if self.repo_handler.is_repo_updated(url, updated_at):
      # Timestamp of already scraped repo matches, so nothing to do.
      return False

    owner  = repo.owner.email
    fork   = 1 if repo.fork else 0
    stars  = repo.stargazers_count
    try:
      contributors = len([x for x in repo.get_contributors()])
    except github.GithubException:
      contributors = -1

    forks      = repo.forks
    created_at = repo.created_at

    self.repo_handler.update_repo(url          = url,       owner        = owner,
                                  name         = name,      fork         = fork,
                                  stars        = stars,     contributors = contributors,
                                  forks        = forks,     created_at   = created_at,
                                  updated_at   = updated_at )

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
    contents = self.download_file(repo, url)
    size     = file.size

    self.repo_handler.update_file(
      url = url, contents = contents, path = path,
      sha = sha, repo_url = repo_url, size = size
    )
    return True

  def download_file(self, repo, url: str, stack = []) -> str:
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

    response = json.loads(requests.get(
      url,
      headers={
        'Authorization': 'token ' + str(self.token)
      }
    ).content.decode('utf-8'))
    src = b64decode(response['content']).decode('utf-8')

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
          include_src = self.download_file(repo, include_url, stack)
          outlines.append(include_src)
        else:
          if not include_url:
            outlines.append('// [FETCH] didnt find: \n' + line)
          else:
            outlines.append('// [FETCH] skipped: ' + line)
      else:
        outlines.append(line)

    return '\n'.join(outlines)

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

  def inline_fs_headers(self, path: str, stack: typing.List[str]) -> str:
    """
    Recursively inline headers in file.

    Parameters
    ----------
    path : str
      File.
    stack : typing.List[str]
      File stack.

    Returns
    -------
    str
      Inlined file.
    """
    stack.append(path)

    with io.open(path) as infile:
      src = infile.read()

    outlines = []
    for line in src.split('\n'):
      match = re.match(re.compile('\w*#include ["<](.*)[">]'), line)
      if match:
        include_name = match.group(1)

        # try and resolve relative paths
        include_name = include_name.replace('../', '')

        include_path = os.path.join(os.path.dirname(path), include_name)

        if os.path.exists(include_path) and include_path not in stack:
          include_src = inline_fs_headers(include_path, stack)
          outlines.append('// [FETCH] include: ' + include_path)
          outlines.append(include_src)
          outlines.append('// [FETCH] eof(' + include_path + ')')
        else:
          if include_path in stack:
            outlines.append('// [FETCH] ignored recursive include: ' +
                    include_path)
          else:
            outlines.append('// [FETCH] 404 not found: ' +
                    include_path)
      else:
        outlines.append(line)

    return '\n'.join(outlines)


  def process_cl_file(self, db_path: str, path: str) -> None:
    """
    Process OpenCL file.

    Parameters
    ----------
    db_path : str
      Path to output database.
    path : str
      Path to input file.

    Raises
    ------
    IOError
      In case of IO error.
    """
    db = dbutil.connect(db_path)
    c = db.cursor()

    l.getLogger().info("fetch {path}".format(path=os.path.abspath(path)))
    try:
      contents = inline_fs_headers(path, [])
    except IOError:
      raise IOError(
        "cannot read file '{path}'".format(path=os.path.abspath(path)))
    c.execute('INSERT OR IGNORE INTO ContentFiles VALUES(?,?)',
          (path, contents))

    db.commit()
    c.close()


  def fetch_files(self, db_path: str, paths: typing.List[str]=[]) -> None:
    """
    Fetch from a list of files.

    Parameters
    ----------
    db_path : str
      Output dataset.
    paths : typing.List[str]
      typing.List of file paths.
    """
    paths = fs.files_from_list(*paths)  # expand directories

    db = dbutil.connect(db_path)
    c = db.cursor()

    for path in paths:
      l.getLogger().info("fetch", path)
      try:
        contents = inline_fs_headers(path, [])
      except IOError:
        db.commit()
        raise IOError(
          "cannot read file '{path}'".format(path=os.path.abspath(path)))
      c.execute('INSERT OR IGNORE INTO ContentFiles VALUES(?,?)',
            (path, contents))

    db.commit()
