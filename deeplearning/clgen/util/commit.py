import git
import pathlib

repo = git.Repo(search_parent_directories = True)

def saveCommit(path: pathlib.Path):
  with open(path / "commit", 'a') as cf:
    cf.write(repo.head.object.hexsha + "\n")
  return

def loadCommit(path: pathlib.Path):
  with open(path / "commit", 'r') as cf:
    return [hx.replace('\n', '') for hx in cf.readlines()]