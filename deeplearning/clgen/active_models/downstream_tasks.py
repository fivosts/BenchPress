"""
This module specifies the range of available
downstream tasks that the committee can be trained on.

The input and output features per downstream task are defined.
"""
import pathlib

from deeplearning.clgen.active_models import data_generator
from deeplearning.clgen.experiments import cldrive

class DownstreamTask(object):
  """
  Downstream Task generic class.
  """
  @classmethod
  def FromTask(cls, task: str, corpus: pathlib.Path) -> "DownstreamTask":
    return TASKS[task](corpus)

  def __init__(self, name) -> None:
    self.name = name
    return

class GrewePredictive(DownstreamTask):
  """
  Specification class for Grewe et al. CGO 2013 predictive model.
  """
  def __init__(self, corpus: pathlib.Path) -> None:
    super(GrewePredictive, self).__init__("GrewePredictive")
    self.inputs         = ["comp", "mem", "localmem", "coalesced", "atomic"]
    self.input_size     = 10
    self.output_labels  = ["CPU", "GPU"]
    self.output_size    = 2
    self.corpus_db      = cldrive.CLDriveExecutions(url = "sqlite:///{}".format(str(self.corpus)), must_exist = True)
    self.data_generator = data_generator.GrewePredictiveLoader()
    return

TASKS = {
  "GrewePredictive": GrewePredictive,
}
