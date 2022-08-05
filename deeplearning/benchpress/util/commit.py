# coding=utf-8
# Copyright 2022 Foivos Tsimpourlas.
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
import git
import pathlib
import typing

def saveCommit(path: pathlib.Path) -> None:
  curr_commits = loadCommit(path)
  repo = git.Repo(search_parent_directories = True)
  cm = repo.head.object.hexsha
  if cm not in curr_commits:
    with open(path / "commit", 'a') as cf:
      cf.write(repo.head.object.hexsha + "\n")
  return

def loadCommit(path: pathlib.Path) -> typing.List[str]:
  if (path / "commit").exists():
    with open(path / "commit", 'r') as cf:
      return [hx.replace('\n', '') for hx in cf.readlines()]
  else:
    return []