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
import datetime

from absl import app, flags

from deeplearning.clgen.samplers import sample_observers as sample_observers_lib
from deeplearning.clgen.samplers import samplers
from deeplearning.clgen.util import pbutil
from deeplearning.clgen.util import memory
from deeplearning.clgen.util import environment
from deeplearning.clgen.util import logging as l
from deeplearning.clgen.dashboard import dashboard
from deeplearning.clgen.models import models
from deeplearning.clgen.proto import clgen_pb2
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import sampler_pb2
from deeplearning.clgen.github import miner

from eupy.hermes import client

FLAGS = flags.FLAGS

flags.DEFINE_string(
  "notify_me",
  None,
  "Set receiver mail address to notify for program failures or termination."
)

flags.DEFINE_integer(
  "notify_me_level",
  5,
  "Define logging level of mail client"
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
flags.DEFINE_boolean(
  "only_sample",
  False,
  "Select to deploy sampling without training."
)
flags.DEFINE_string(
  "print_cache_path",
  None,
  'Print the directory of a cache and exit. Valid options are: "pre_train_corpus", "corpus", '
  '"model", or "sampler".',
)
flags.DEFINE_boolean(
  "debug",
  False,
  "Enable a debugging mode of CLgen python runtime. When enabled, errors "
  "which may otherwise be caught lead to program crashes and stack traces.",
)
flags.DEFINE_boolean(
  "profiling",
  False,
  "Enable CLgen self profiling. Profiling results be logged.",
)
flags.DEFINE_boolean(
  "monitor_mem_usage",
  False,
  "Plot application RAM and GPU memory usage."
)
flags.DEFINE_boolean(
  "dashboard_only", False, "If true, launch dashboard only."
)

class Instance(object):
  """A CLgen instance encapsulates a github_miner, model, sampler, and working directory."""

  def __init__(self, config: clgen_pb2.Instance):
    """Instantiate an instance.

    Args:
      config: An Instance proto.
    """
    self.working_dir = None
    self.github      = None
    self.model       = None
    self.sampler     = None

    self.config = config

    if config.HasField("github_miner"):
      self.github = miner.GithubMiner.FromConfig(config.github_miner)

    if config.HasField("working_dir"):
      self.working_dir: pathlib.Path = pathlib.Path(
        os.path.join(FLAGS.workspace_dir, config.working_dir)
      ).expanduser().resolve()
    # Enter a session so that the cache paths are set relative to any requested
    # working directory.
    with self.Session():
      if config.HasField("model"):
        self.model: models.Model = models.Model(config.model)
      if config.HasField("sampler"):
        self.sampler: samplers.Sampler = samplers.Sampler(config.sampler)

    self.dashboard = dashboard.Launch()

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

  def PreTrain(self, *args, **kwargs) -> None:
    if self.config.model.HasField("pre_train_corpus"):
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
        test_sampler = samplers.Sampler(test_sampler_config, sample_db_name = "pre_epoch_samples.db")

        # We inject the `test_sampler` argument so that we can create samples
        # during training.
        self.model.PreTrain(*args, test_sampler = test_sampler, **kwargs)

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
    self.PreTrain()
    self.Train()
    with self.Session():
      self.model.Sample(self.sampler, *args, **kwargs)

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
  if instance.sampler is None:
    return []

  sample_observers = []
  if FLAGS.min_samples <= 0:
    l.logger().warning(
      "Entering an infinite sample loop, this process will never end!"
    )
  else:
    sample_observers.append(
      sample_observers_lib.MaxSampleCountObserver(FLAGS.min_samples * instance.sampler.batch_size)
    )
  if FLAGS.print_samples:
    sample_observers.append(sample_observers_lib.PrintSampleObserver())
  if FLAGS.store_samples_db:
    (instance.model.cache.path / "samples" / instance.sampler.hash).mkdir(exist_ok = True)
    sample_observers.append(sample_observers_lib.SamplesDatabaseObserver(
        instance.model.cache.path / "samples" / instance.sampler.hash / instance.sampler.sample_db_name,
        plot_sample_status = True
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
    instance: The CLgen instance to act on.
    sample_observer: A list of sample observers. Unused if no sampling occurs.
  """

  if instance.github:
    instance.github.fetch()

  if instance.model:
    with instance.Session():
      if FLAGS.print_cache_path == "pre_train_corpus":
        print(instance.model.pre_train_corpus.cache.path)
        return
      elif FLAGS.print_cache_path == "corpus":
        print(instance.model.corpus.cache.path)
        return
      elif FLAGS.print_cache_path == "model":
        print(instance.model.cache.path)
        return
      elif FLAGS.print_cache_path == "sampler":
        if instance.sampler:
          print(instance.model.SamplerCache(instance.sampler))
        else:
          raise ValueError("Sampler config has not been specified.")
        return
      elif FLAGS.print_cache_path:
        raise ValueError(f"Invalid --print_cache_path argument: '{FLAGS.print_cache_path}'")

    # The default action is to sample the model.
      if FLAGS.stop_after == "corpus":
        instance.model.corpus.Create()
        if instance.model.pre_train_corpus:
          instance.model.pre_train_corpus.Create(tokenizer = instance.model.corpus.tokenizer)
      elif FLAGS.stop_after == "pre_train":
        instance.PreTrain()
        l.logger().info("Model: {}".format(instance.model.cache.path))
      elif FLAGS.stop_after == "train":
        instance.Train()
        l.logger().info("Model: {}".format(instance.model.cache.path))
      elif FLAGS.stop_after:
        raise ValueError(
          f"Invalid --stop_after argument: '{FLAGS.stop_after}'"
        )
      else:
        if instance.sampler:
          instance.Sample(sample_observers)
          instance.sampler.symlinkModelDB(
            instance.model.cache.path / "samples" / instance.sampler.hash,
            instance.model.hash
          )
        else:
          l.logger().warn("Sampler has not been provided. Use --stop_after to create corpus or train.")
  else:
    if FLAGS.stop_after in {"corpus", "train"}:
      l.logger().warn("FLAGS.stop_after {} will be ignored without model config.".format(FLAGS.stop_after))
    if FLAGS.print_cache_path in {"pre_train_corpus", "corpus", "model", "sampler"}:
      raise ValueError("{} config has not been specified.".format(FLAGS.print_cache_path))
    elif FLAGS.print_cache_path:
      raise ValueError(f"Invalid --print_cache_path argument: '{FLAGS.print_cache_path}'")
  return

def main():
  """Main entry point."""
  if FLAGS.dashboard_only:
    dash = dashboard.Launch(debug = {"debug": True})
  else:
    instance = Instance(ConfigFromFlags())
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
  mail = None
  if FLAGS.notify_me:
    mail = client.initClient(FLAGS.notify_me)
  l.initLogger(name = "clgen", mail = mail, rank = environment.WORLD_RANK)
  if FLAGS.local_filesystem:
    pathlib.Path(FLAGS.local_filesystem).resolve().mkdir(exist_ok = True, parents = True)
  if FLAGS.monitor_mem_usage:
    mem_monitor_threads = memory.init_mem_monitors(
      pathlib.Path(FLAGS.workspace_dir).resolve()
    )
  if FLAGS.debug:
    # Enable verbose stack traces. See: https://pymotw.com/2/cgitb/
    import cgitb
    cgitb.enable(format="text")
    main()
    return
  try:
    if FLAGS.profiling:
      cProfile.runctx("main()", None, None, sort="tottime")
    else:
      main()
  except KeyboardInterrupt:
    return
  except Exception as e:
    l.logger().error(e)
    if mail:
      if FLAGS.config is not None:
        job = pathlib.Path(FLAGS.config)
      else:
        job = ""
      mail.send_message("clgen:{}".format(str(job.stem)), e)
    raise

  if mail:
    if FLAGS.config is not None:
      job = pathlib.Path(FLAGS.config)
    else:
      job = ""
    mail.send_message("clgen: {}".format(str(job.stem)), "Program terminated successfully at {}.".format(datetime.datetime.utcnow().strftime("%m/%d/%Y, %H:%M:%S")))
  return

if __name__ == "__main__":
  app.run(initMain)
  sys.exit(0)
