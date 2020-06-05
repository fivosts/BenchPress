#
# Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of CLgen.
#
# CLgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CLgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CLgen.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Fetch OpenCL files
"""
import json
import os
import io
import re
import time
import requests
import sys
import typing
import github

from base64 import b64decode
from functools import partial
from labm8.py import fs
from eupy.native import logger as l
# from deeplearning.clgen import dbutil

# Counters
repos_new_counter = 0
repos_modified_counter = 0
repos_unchanged_counter = 0
files_new_counter = 0
files_modified_counter = 0
files_unchanged_counter = 0
errors_counter = 0
status_string = ''


class GithubFetcher():
  """GitHub API wrapper to pull from github a fresh corpus of OpenCL kernels"""
  def __init__(self,
               corpus_path: str
               ):

    l.getLogger().info("Github fetcher initialized: {}".format(corpus_path))

    self.corpus_path     = corpus_path
    git_credentials = {
      'GITHUB_USERNAME'  : None,
      'GITHUB_PW'        : None,
      'GITHUB_TOKEN'     : None,
    }

    if not all(k in os.environ for k in git_credentials.keys()):
      l.getLogger().warn("Export github credentials as environment variables to speed up the process")

    for key in git_credentials:
      if key in os.environ:
        git_credentials[key] = os.environ[key]
      else:
        git_credentials[key] = input("{}: ".format(key))
        os.environ[key]      = git_credentials[key]

    self.username = git_credentials['GITHUB_USERNAME']
    self.password = git_credentials['GITHUB_PW']
    self.token    = git_credentials['GITHUB_TOKEN']
    return

  def _print_counters(self) -> None:
    """
    Print analytics counters.
    """
    print('\r\033[Kfiles: new ', files_new_counter,
        ', modified ', files_modified_counter,
        '. errors ', errors_counter,
        '. ', status_string[0:25],
        sep='', end='')
    sys.stdout.flush()


  def _rate_limit(self, g) -> None:
    """
    Block on GitHub rate limit.

    Parameters
    ----------
    g
      GitHub connection.
    """
    global status_string
    remaining = g.get_rate_limit().rate.remaining
    while remaining < 100:
      time.sleep(1)
      status_string = 'WAITING ON RATE LIMIT'
      self._print_counters()
      remaining = g.get_rate_limit().rate.remaining


  def _process_repo(self, g, db, repo) -> bool:
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
    db : sqlite3.Connection
      Dataset.
    repo
      Repository.

    Returns
    -------
    bool
      True if repository should be scraped, else False.
    """
    global repos_new_counter
    global repos_modified_counter
    global repos_unchanged_counter
    global status_string

    _rate_limit(g)
    url = repo.url
    updated_at = str(repo.updated_at)
    name = repo.name
    status_string = name
    self._print_counters()

    c = db.cursor()
    c.execute("SELECT updated_at FROM Repositories WHERE url=?", (url,))
    cached_updated_at = c.fetchone()

    # Do nothing unless updated timestamps don't match
    if cached_updated_at and cached_updated_at[0] == updated_at:
      repos_unchanged_counter += 1
      return False

    owner = repo.owner.email
    fork = 1 if repo.fork else 0
    stars = repo.stargazers_count
    try:
      contributors = len([x for x in repo.get_contributors()])
    except github.GithubException:
      contributors = -1
    forks = repo.forks
    created_at = repo.created_at
    updated_at = repo.updated_at

    c.execute("DELETE FROM Repositories WHERE url=?", (url,))
    c.execute("INSERT INTO Repositories VALUES(?,?,?,?,?,?,?,?,?)",
          (url, owner, name, fork, stars, contributors, forks, created_at,
           updated_at))

    if cached_updated_at:
      repos_modified_counter += 1
    else:
      repos_new_counter += 1
    db.commit()
    return True

  def _download_file(self, github_token: str, repo, url: str, stack: typing.List[str]) -> str:
    """
    Fetch file from GitHub.

    Recursively downloads and inlines headers.

    Parameters
    ----------
    github_token : str
      Authorization.
    repo
      Repository.
    url : str
      Path.
    stack : typing.List[str]
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
        'Authorization': 'token ' + str(github_token)
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
          include_src = _download_file(github_token, repo, include_url)
          outlines.append(include_src)
        else:
          if not include_url:
            outlines.append('// [FETCH] didnt find: ' + line)
          else:
            outlines.append('// [FETCH] skipped: ' + line)
      else:
        outlines.append(line)

    return '\n'.join(outlines)


  def _process_file(self, g, github_token: str, db, repo, file) -> bool:
    """
    GitHub file handler.

    Parameters
    ----------
    g
      GitHub connection.
    github_token : str
      Authorization.
    db : sqlite3.Connection
      Dataset.
    repo
      Repository.
    file
      File.

    Returns
    -------
    bool
      True on success, else False.
    """
    global files_new_counter
    global files_modified_counter
    global files_unchanged_counter
    global status_string

    # We're only interested in OpenCL files.
    if not (file.path.endswith('.cl') or path.endswith('.ocl')):
      return

    url = file.url
    sha = file.sha
    path = file.path
    status_string = repo.name + '/' + path
    self._print_counters()

    c = db.cursor()
    c.execute("SELECT sha FROM ContentMeta WHERE id=?", (url,))
    cached_sha = c.fetchone()

    # Do nothing unless checksums don't match
    if cached_sha and cached_sha[0] == sha:
      files_unchanged_counter += 1
      return False

    repo_url = repo.url
    contents = _download_file(github_token, repo, file.url, [])
    size = file.size

    c.execute("DELETE FROM ContentFiles WHERE id=?", (url,))
    c.execute("DELETE FROM ContentMeta WHERE id=?", (url,))
    c.execute("INSERT INTO ContentFiles VALUES(?,?)",
          (url, contents))
    c.execute("INSERT INTO ContentMeta VALUES(?,?,?,?,?)",
          (url, path, repo_url, sha, size))

    if cached_sha:
      files_modified_counter += 1
    else:
      files_new_counter += 1

    db.commit()
    return True


  def fetch(self, db_path: str, github_username: str, github_pw: str,
           github_token: str) -> None:
    """
    Download all of the OpenCL on GitHub (!)

    Shortcomings of this appraoch:
      * Only includes exclusively OpenCL files, no inline strings.
      * Occasionally (< 1%) can't find headers to include.

    Parameters
    ----------
    db_path : str
      Dataset path.
    github_username : str
      Authorization.
    github_pw : str
      Authorization.
    github_token : str
      Authorization.
    """
    global errors_counter

    g = github.Github(github_username, github_pw)
    db = dbutil.connect(db_path)

    if not dbutil.is_github:
      raise ValueError("Not a github database")

    handle_repo = partial(self._process_repo, g, db)

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
    for query in query_terms:
      # forks are okay - we use checksums to ensure uniqueness in
      # final dataset
      repos = g.search_repositories(query + ' fork:true sort:stars')

      for repo in repos:
        repo_modified = handle_repo(repo)

        # do nothing unless the repo is new or modified
        if not repo_modified:
          continue

        handle_file = partial(_process_file, g, github_token, db, repo)

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
            except Exception:
              errors_counter += 1
        except github.GithubException:
          # do nothing in case of error (such as an empty repo)
          pass

    self._print_counters()
    print("\n\ndone.")
    db.close()


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

    l.getLogger().info("fetch {path}".format(path=fs.abspath(path)))
    try:
      contents = inline_fs_headers(path, [])
    except IOError:
      raise IOError(
        "cannot read file '{path}'".format(path=fs.abspath(path)))
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
          "cannot read file '{path}'".format(path=fs.abspath(path)))
      c.execute('INSERT OR IGNORE INTO ContentFiles VALUES(?,?)',
            (path, contents))

    db.commit()
