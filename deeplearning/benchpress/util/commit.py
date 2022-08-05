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