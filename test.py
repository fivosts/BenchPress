import sys
import pathlib
from absl import app, flags

from deeplearning.clgen import validation_database

def str_with_color(text, color = None, background_color = None, font_family = None, font_size = None):
  return(
    "<p style=\"color:{};background-color:{};font-family:{};font-size:{}\">{}</p>"
    .format(color, background_color, font_family, font_size,
      text)
    )

def rundb():
  db = validation_database.ValidationDatabase("sqlite:////home/fivosts/PhD/Code/clgen/validation_samples.db")
  print("<br>")
  print(db, end = "<br>")
  print(db.count)
  print(str_with_color("malakia", color = "white", font_family = "Monospace", font_size = 18))
  print(sys.argv)

def get_workspaces():
  base_path = pathlib.Path("./workspace")
  for file in base_path.iterdir():
    if file.is_dir():
      print(file.stem, end = "<br>")

def main(*args, **kwargs):
  if sys.argv[1] == "rundb":
    rundb()
  elif sys.argv[1] == "get_workspaces":
    get_workspaces()

app.run(main)

