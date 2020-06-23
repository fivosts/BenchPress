"""CLgen: a deep learning program generator.

The core operations of CLgen are:

  1. Preprocess and encode a corpus of handwritten example programs.
  2. Define and train a machine learning model on the corpus.
  3. Sample the trained model to generate new programs.

This program automates the execution of all three stages of the pipeline.
The pipeline can be interrupted and resumed at any time. Results are cached
across runs. Please note that many of the steps in the pipeline are extremely
compute intensive and highly parallelized. If configured with CUDA support,
any NVIDIA GPUs will be used to improve performance where possible.
"""
import contextlib
import cProfile
import os
import pathlib
import shutil
import sys
import typing

from absl import app, flags

from deeplearning.clgen import sample_observers as sample_observers_lib
from deeplearning.clgen import samplers
from deeplearning.clgen import pbutil
from deeplearning.clgen import tf
from deeplearning.clgen.dashboard import dashboard
from deeplearning.clgen.models import models
from deeplearning.clgen.models import pretrained
from deeplearning.clgen.proto import clgen_pb2
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import sampler_pb2

from eupy.native import logger as l

FLAGS = flags.FLAGS

flags.DEFINE_integer(
  "level",
  20,
  "Define logging level of logger"
)

flags.DEFINE_boolean(
  "color", True, "Colorize or not, logging messages"
)

flags.DEFINE_boolean(
  "step", False, "Enable step execution on debug logs (debug level must be selected)"
)

flags.DEFINE_string(
  "config", "/clgen/config.pbtxt", "Path to a clgen.Instance proto file."
)
flags.DEFINE_string(
  "workspace_dir",
  "/tmp/clgen",
  "Root path of the working space directory. Corpus, dataset, model and all meta files"
  "will be stored here. Default value is /tmp folder.",
)
flags.DEFINE_integer(
  "min_samples",
  0,
  "The minimum number of samples to make. If <= 0, sampling continues "
  "indefinitely and never terminates.",
)
flags.DEFINE_boolean(
  "print_samples", True, "If set, print the generated samples."
)
flags.DEFINE_boolean(
  "store_samples_db", True, "If set, store generated samples to database."
)
flags.DEFINE_boolean(
  "cache_samples", False, "If set, cache the generated sample protobufs."
)
flags.DEFINE_string(
  "sample_text_dir", None, "A directory to write plain text samples to."
)
flags.DEFINE_string(
  "stop_after",
  None,
  'Stop CLgen early. Valid options are: "corpus", or "train".',
)
flags.DEFINE_string(
  "print_cache_path",
  None,
  'Print the directory of a cache and exit. Valid options are: "corpus", '
  '"model", or "sampler".',
)
flags.DEFINE_string(
  "export_model",
  None,
  "Path to export a trained TensorFlow model to. This exports all of the "
  "files required for sampling to specified directory. The directory can "
  "then be used as the pretrained_model field of an Instance proto config.",
)
flags.DEFINE_boolean(
  "clgen_debug",
  False,
  "Enable a debugging mode of CLgen python runtime. When enabled, errors "
  "which may otherwise be caught lead to program crashes and stack traces.",
)
flags.DEFINE_boolean(
  "clgen_profiling",
  False,
  "Enable CLgen self profiling. Profiling results be logged.",
)
flags.DEFINE_boolean(
  "clgen_dashboard_only", False, "If true, launch dashboard only."
)

class Instance(object):
  """A CLgen instance encapsulates a model, sampler, and working directory."""

  def __init__(self, config: clgen_pb2.Instance, dashboard_opts={}):
    """Instantiate an instance.

    Args:
      config: An Instance proto.

    Raises:
      UserError: If the instance proto contains invalid values, is missing
        a model or sampler fields.
    """
    try:
      pbutil.AssertFieldIsSet(config, "model_specification")
      pbutil.AssertFieldIsSet(config, "sampler")
    except pbutil.ProtoValueError as e:
      raise ValueError(e)

    self.config = config
    self.working_dir = None
    if config.HasField("working_dir"):
      self.working_dir: pathlib.Path = pathlib.Path(
        os.path.join(FLAGS.workspace_dir, config.working_dir)
      ).expanduser().absolute()
    # Enter a session so that the cache paths are set relative to any requested
    # working directory.
    with self.Session():
      if config.HasField("model"):
        self.model: models.Model = models.Model(config.model)
      else:
        self.model: pretrained.PreTrainedModel = pretrained.PreTrainedModel(
          pathlib.Path(config.pretrained_model)
        )
      self.sampler: samplers.Sampler = samplers.Sampler(config.sampler)

    self.dashboard = dashboard.Launch(**dashboard_opts)

  @contextlib.contextmanager
  def Session(self) -> "Instance":
    """Scoped $CLGEN_CACHE value."""
    old_working_dir = os.environ.get("CLGEN_CACHE", "")
    if self.working_dir:
      os.environ["CLGEN_CACHE"] = str(self.working_dir)
    yield self
    if self.working_dir:
      os.environ["CLGEN_CACHE"] = old_working_dir

  def Create(self) -> None:
    with self.Session():
      self.model.Create()

  def Train(self, *args, **kwargs) -> None:
    with self.Session():
      test_sampler_config = sampler_pb2.Sampler()
      test_sampler_config.CopyFrom(self.sampler.config)
      # Make all test samples the same sequence_length length.
      del test_sampler_config.termination_criteria[:]
      test_sampler_config.termination_criteria.extend(
        [
          sampler_pb2.SampleTerminationCriterion(
            maxlen=sampler_pb2.MaxTokenLength(maximum_tokens_in_sample=self.sampler.sequence_length)
          ),
        ]
      )
      test_sampler = samplers.Sampler(test_sampler_config, sample_db_name = "epoch_samples.db")

      # We inject the `test_sampler` argument so that we can create samples
      # during training.
      self.model.Train(*args, test_sampler = test_sampler, **kwargs)

  def Sample(self, *args, **kwargs) -> typing.List[model_pb2.Sample]:
    self.Train()
    with self.Session():
      return self.model.Sample(self.sampler, *args, **kwargs)

  def ExportPretrainedModel(self, export_dir: pathlib.Path) -> None:
    """Export a trained model."""
    if isinstance(self.model, pretrained.PreTrainedModel):
      shutil.copytree(self.config.pretrained_model, export_dir / "model")
    else:
      self.Train()
      for path in self.model.InferenceManifest():
        relpath = pathlib.Path(os.path.relpath(path, self.model.cache.path))
        (export_dir / relpath.parent).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, export_dir / relpath)

  def ToProto(self) -> clgen_pb2.Instance:
    """Get the proto config for the instance."""
    config = clgen_pb2.Instance()
    config.working_dir = str(self.working_dir)
    config.model.CopyFrom(self.model.config)
    config.sampler.CopyFrom(self.sampler.config)
    return config

  @classmethod
  def FromFile(cls, path: pathlib.Path) -> "Instance":
    return cls(pbutil.FromFile(path, clgen_pb2.Instance()))

def ConfigFromFlags() -> clgen_pb2.Instance:

  config_path = pathlib.Path(FLAGS.config)
  if not config_path.is_file():
    raise FileNotFoundError (f"CLgen --config file not found: '{config_path}'")
  config = pbutil.FromFile(config_path, clgen_pb2.Instance())
  os.environ["PWD"] = str(config_path.parent)
  return config

def SampleObserversFromFlags(instance: Instance) -> typing.List[
  sample_observers_lib.SampleObserver
]:
  """Instantiate sample observers from flag values."""
  sample_observers = []
  if FLAGS.min_samples <= 0:
    l.getLogger().warning(
      "Entering an infinite sample loop, this process will never end!"
    )
  else:
    sample_observers.append(
      sample_observers_lib.MaxSampleCountObserver(FLAGS.min_samples)
    )
  if FLAGS.print_samples:
    sample_observers.append(sample_observers_lib.PrintSampleObserver())
  if FLAGS.store_samples_db:
    (instance.model.cache.path / "samples" / instance.sampler.hash).mkdir(exist_ok = True)
    sample_observers.append(sample_observers_lib.SamplesDatabaseObserver(
      "sqlite:///{}".format(instance.model.cache.path / "samples" / instance.sampler.hash / instance.sampler.sample_db_name)
      )
    )
    instance.sampler.symlinkModelDB(
      instance.model.cache.path / "samples" / instance.sampler.hash,
      instance.model.hash
    )
  if FLAGS.cache_samples:
    sample_observers.append(sample_observers_lib.LegacySampleCacheObserver())
  if FLAGS.sample_text_dir:
    sample_observers.append(
      sample_observers_lib.SaveSampleTextObserver(
        pathlib.Path(FLAGS.sample_text_dir)
      )
    )
  return sample_observers

def DoFlagsAction(
  instance: Instance,
  sample_observers: typing.List[sample_observers_lib.SampleObserver],
) -> None:
  """Do the action requested by the command line flags.

  By default, this method trains and samples the instance using the given
  sample observers. Flags which affect this behaviour are:

    --print_cache_path={corpus,model,sampler}: Prints the path and returns.
    --stop_after={corpus,train}: Stops after corpus creation or training,
        respectively
    --export_model=<path>: Train the model and export it to the requested path.

  Args:
    config: The CLgen instance to act on.
    sample_observer: A list of sample observers. Unused if no sampling occurs.
  """

  with instance.Session():
    if FLAGS.clgen_dashboard_only:
      instance.Create()
      return
    if FLAGS.print_cache_path == "corpus":
      print(instance.model.corpus.cache.path)
      return
    elif FLAGS.print_cache_path == "model":
      print(instance.model.cache.path)
      return
    elif FLAGS.print_cache_path == "sampler":
      print(instance.model.SamplerCache(instance.sampler))
      return
    elif FLAGS.print_cache_path:
      raise ValueError(f"Invalid --print_cache_path argument: '{FLAGS.print_cache_path}'")

    # The default action is to sample the model.
    if FLAGS.stop_after == "corpus":
      instance.model.corpus.Create()
    elif FLAGS.stop_after == "train":
      instance.Train()
      l.getLogger().info("Model: {}".format(instance.model.cache.path))
    elif FLAGS.stop_after:
      raise ValueError(
        f"Invalid --stop_after argument: '{FLAGS.stop_after}'"
      )
    elif FLAGS.export_model:
      instance.ExportPretrainedModel(pathlib.Path(FLAGS.export_model))
    else:
      instance.Sample(sample_observers)

def main():
  """Main entry point."""
  instance = Instance(
    ConfigFromFlags(), dashboard_opts={"debug": FLAGS.clgen_dashboard_only,}
  )
  sample_observers = SampleObserversFromFlags(instance)
  DoFlagsAction(instance, sample_observers)
  return

def initMain(*args, **kwargs):
  """
  Pre-initialization for the main function of the program

  Args:
    *args: Arguments to be passed to the function.
    **kwargs: Arguments to be passed to the function.
  """
  l.initLogger(name = "clgen", lvl = FLAGS.level, colorize = FLAGS.color, step = FLAGS.step)
  tf.initTensorflow()
  if FLAGS.clgen_debug:
    # Enable verbose stack traces. See: https://pymotw.com/2/cgitb/
    import cgitb
    cgitb.enable(format="text")
    main()
    sys.exit(0)
  try:
    if FLAGS.clgen_profiling:
      cProfile.runctx("main()", None, None, sort="tottime")
    else:
      main()
  except KeyboardInterrupt:
    sys.exit(1)
  except Exception as e:
    l.getLogger().error(e)
    raise
    sys.exit(1)
  sys.exit(0)

if __name__ == "__main__":
  app.run(initMain)
