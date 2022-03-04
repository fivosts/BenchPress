"""
This module specifies the range of available
downstream tasks that the committee can be trained on.

The input and output features per downstream task are defined.
"""
import pathlib
import tqdm
import multiprocessing

from deeplearning.clgen.active_models import data_generator
from deeplearning.clgen.experiments import cldrive
from deeplearning.clgen.features import extractors

def ExtractorWorker(src: str, fspace: str, cldrive_entry: cldrive.CLDriveSample):
  """
  Worker that extracts features and buffers cldrive entry, to maintain consistency
  among multiprocessed data.
  """
  features = extractors.ExtractFeatures(src, [fspace])
  if fspace in features and features[fspace]:
    return features, cldrive_entry
  return None

class DownstreamTask(object):
  """
  Downstream Task generic class.
  """
  @classmethod
  def FromTask(cls, task: str, corpus_path: pathlib.Path) -> "DownstreamTask":
    return TASKS[task](corpus_path)

  def __init__(self, name) -> None:
    self.name = name
    return

class GrewePredictive(DownstreamTask):
  """
  Specification class for Grewe et al. CGO 2013 predictive model.
  """
  def __init__(self, corpus_path: pathlib.Path) -> None:
    super(GrewePredictive, self).__init__("GrewePredictive")
    self.inputs         = ["comp", "mem", "localmem", "coalesced", "atomic"]
    self.input_size     = 10
    self.output_labels  = ["CPU", "GPU"]
    self.output_size    = 2
    self.setup_dataset()
    # self.data_generator = data_generator.GrewePredictiveLoader(self.corpus_db.get_the_right_data)
    return

  def setup_dataset(self) -> None:
    """
    Fetch data and preprocess into corpus for Grewe's predictive model.
    """
    self.dataset = []

    self.corpus_db = cldrive.CLDriveExecutions(url = "sqlite:///{}".format(str(self.corpus_path)), must_exist = True)
    data    = [x for x in self.corpus_db.get_valid_data()]
    sources = [(x, "") for x.source in data]

    pool = multiprocessing.Pool()
    it = zip(pool.imap_unordered(workers.data), sources)
    for dp in tqdm.tqdm(self.corpus_db.get_valid_data(), total = self.corpus_db.count, desc = "Grewe corpus setup", leave = False):
      feats = 


  def TargettoLabels(id: int) -> str:
    """
    Integer ID to label of predictive model.
    """
    return {
      0: "CPU",
      1: "GPU",
    }[id]

  def TargettoID(label: str) -> int:
    """
    Predictive label to ID.
    """
    return {
      "CPU": 0,
      "GPU": 1,
    }

  def TargettoEncodedVector(label: str) -> typing.List[int]:
    """
    Label to target vector.
    """
    return {
      "CPU": [1, 0],
      "GPU": [0, 1],
    }

TASKS = {
  "GrewePredictive": GrewePredictive,
}
