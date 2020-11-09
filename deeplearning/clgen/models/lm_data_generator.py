"""Data generator specifically used for Mask LM models (namely BERT)."""
import sys
import time
import random
import progressbar
import copy
import glob
import humanize
import typing
import multiprocessing
import functools
import pathlib
import pickle
import numpy as np

from deeplearning.clgen.util import cache
from deeplearning.clgen.util import pbutil
from deeplearning.clgen.util import distributions
from deeplearning.clgen.util import monitors
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.models import sequence_masking
from absl import flags
from eupy.native import logger as l

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
  "write_text_dataset", 
  False, 
  "Set True for MaskLM data generator to write dataset in text format, along with the dataset record."
)

flags.DEFINE_integer(
  "memory_limit",
  4,
  "Set maximum amount of available memory used for masking sequences in Gb. [Default]: 4",
)
flags.DEFINE_boolean(
  "force_remake_dataset",
  False,
  "Force data generator to re-mask encoded dataset and store dataset record."
)

def AssertConfigIsValid(config: model_pb2.DataGenerator,
                        is_sampling: bool = False
                        ) -> model_pb2.DataGenerator:
  """
  Parse data generator protobuf message.
  Raise Exception if format is wrong.
  """
  if not is_sampling:
    # The attributes below are not allowed to be changed in sampling.
    pbutil.AssertFieldConstraint(
      config,
      "datapoint_type",
      lambda x: x == "kernel" or x == "statement",
      "Valid options for datapoint_type are 'kernel' and 'statement'",
    )
    pbutil.AssertFieldIsSet(
      config,
      "use_start_end",
    )
    pbutil.AssertFieldIsSet(
      config,
      "steps_per_epoch",
    )
    pbutil.AssertFieldIsSet(
      config,
      "validation_split",
    )
    if len(config.validation_set) > 0:
      for val_opt in config.validation_set:
        if val_opt.HasField("mask"):
          pbutil.AssertFieldIsSet(
            val_opt.mask,
            "random_placed_mask",
          )
        elif val_opt.HasField("hole"):
          pbutil.AssertFieldConstraint(
            val_opt.hole,
            "hole_length",
            lambda x : x > 0,
            "hole_length is the upper bound range of a hole's length. Therefore should be > 0."
          )
          if val_opt.hole.HasField("normal_distribution"):
            pbutil.AssertFieldIsSet(
              val_opt.hole.normal_distribution,
              "mean",
            )
            pbutil.AssertFieldIsSet(
              val_opt.hole.normal_distribution,
              "variance",
            )
          elif not val_opt.hole.HasField("uniform_distribution"):
            raise ValueError("Hole length distribution has not been set.")
  # Parse masking technique for bert's data generator
  pbutil.AssertFieldIsSet(config, "mask_technique")
  if config.HasField("mask"):
    pbutil.AssertFieldIsSet(
      config.mask,
      "random_placed_mask",
    )
  elif config.HasField("hole"):
    pbutil.AssertFieldConstraint(
      config.hole,
      "hole_length",
      lambda x : x > 0,
      "hole_length is the upper bound range of a hole's length. Therefore should be > 0."
    )
    if config.hole.HasField("normal_distribution"):
      pbutil.AssertFieldIsSet(
        config.hole.normal_distribution,
        "mean",
      )
      pbutil.AssertFieldIsSet(
        config.hole.normal_distribution,
        "variance",
      )
    elif not config.hole.HasField("uniform_distribution"):
      raise ValueError("Hole length distribution has not been set.")
    if not is_sampling:
      pbutil.AssertFieldIsSet(
        config.hole,
        "stage_training",
      )
  return config

class MaskLMDataGenerator(object):
  """Abstract class, shared among TORCH and TF BERT data generators."""
  @property
  def is_torch(self):
    if self.file_extension == "pt_record":
      return True
    return False
  
  def __init__(self, file_extension: str):

    self.file_extension = file_extension
    self.mask_func      = sequence_masking.MaskSequence
    self.hole_func      = sequence_masking.HoleSequence

    self.dataset                 = None
    self.corpus                  = None
    self.atomizer                = None
    self.config                  = None
    self.cache                   = None

    self.training_opts           = None
    self.steps_per_epoch         = None
    self.max_position_embeddings = None
    self.num_epochs              = None
    self.steps_per_epoch         = None

    self.sampler                 = None
    self.rngen                   = None
    return

  def TrainMaskLMBatchGenerator(self,
                                corpus: "corpuses.Corpus",
                                training_opts: model_pb2.TrainingOptions,
                                cache_path: pathlib.Path,
                                ) -> "data_generator.MaskLMDataGenerator":
    """Initializes data generator for training."""
    self.cache         = cache.mkcache(cache_path, "dataset")
    self.cache.path.mkdir(exist_ok = True, parents = True)

    self.dataset       = {}
    self.corpus        = corpus
    self.atomizer      = corpus.atomizer
    self.config        = training_opts.data_generator
    self.training_opts = training_opts
    self.rngen         = random.Random(training_opts.random_seed)

    shaped_corpus = self.createCorpus(self.cache.path)
    self.configDataset(shaped_corpus)
    return self

  def SampleMaskLMBatchGenerator(self,
                                 model_opts,
                                 sampler,
                                 atomizer,
                                 seed: int,
                                 max_position_embeddings: int,
                                 cache_path: pathlib.Path,
                                 ) -> "data_generator.MaskLMBatchGenerator":
    """Initializes data generator for inference."""
    self.cache                   = cache.mkcache(cache_path, "dataset")
    self.cache.path.mkdir(exist_ok = True, parents = True)

    self.dataset                 = {}
    self.sampler                 = sampler
    self.atomizer                = atomizer
    self.rngen                   = random.Random(seed)
    self.max_position_embeddings = max_position_embeddings

    self.training_opts                 = model_opts
    self.training_opts.sequence_length = sampler.sequence_length
    self.training_opts.batch_size      = sampler.batch_size
    return self

  def configDataset(self,
                    shaped_corpus: np.array,
                    ) -> None:
    """
      Configs all necessary training and validation sets described in the model protobuf.
      First constructs training set and optionally splits it into validation set, if selected in config.
      Then configValidationSets is called which constructs any additional validation_set elements
      provided in the model's config.
    """
    assert self.config.validation_split >= 0 and self.config.validation_split <= 100

    if FLAGS.force_remake_dataset:
      l.getLogger().warn("Force remaking datasets can cause lots of problems on an already trained model. Are you sure you want to proceed ? [y/n]")
      a = input()
      if a.lower() != "yes" and a.lower() != "y":
        l.getLogger().warn("Overwriting dataset process was aborted. Good call.")
        exit()

    if len(glob.glob(str(self.cache.path / "train_dataset_*.{}".format(self.file_extension)))) == 0 or FLAGS.force_remake_dataset:
      if self.config.validation_split == 0:
        self._maskCorpus(
          shaped_corpus, set_name = "train_dataset", path = self.cache.path, train_set = True
        )
      else:
        split_index  = (len(shaped_corpus) // 100) * self.config.validation_split
        self._maskCorpus(
          shaped_corpus[split_index:], set_name = "train_dataset", path = self.cache.path, train_set = True
        )
        self._maskCorpus(
          shaped_corpus[:split_index], set_name = "validation_dataset", path = self.cache.path, train_set = False
        )
    else:
      self.dataset["train_dataset"] = {
        "file": sorted(glob.glob(str(self.cache.path / "train_dataset_*.{}".format(self.file_extension)))),
        "txt" : sorted(glob.glob(str(self.cache.path / "train_dataset_*.txt"))),
      }
      if len(glob.glob(str(self.cache.path / "validation_dataset_*.{}".format(self.file_extension)))) != 0:
        self.dataset["validation_dataset"] = {
          "file": sorted(glob.glob(str(self.cache.path / "validation_dataset_*.{}".format(self.file_extension)))),
          "txt" : sorted(glob.glob(str(self.cache.path / "validation_dataset_*.txt"))),
        }

    self.configValidationSets(self.config.validation_set, shaped_corpus, self.cache.path)
    return

  def configValidationSets(self,
                           valset_list: typing.List,
                           shaped_corpus: np.array,
                           path: pathlib.Path,
                           ) -> None:
    """
      Mask and store any extra validation datasets defined into
      model protobuf.
      Example:
        validation_set {
          max_predictions_per_seq: 10
          hole {
            hole_length: 15
            uniform_distribution: true
          }
        }

      Arguments:
        valset_list: list of validation_set items
      Returns:
        None
    """
    for valset in valset_list:
      set_name = "pred_{}_{}".format(
        valset.max_predictions_per_seq,
        "mask" if valset.HasField("mask") else "hole_{}".format(valset.hole.hole_length)
      )
      if set_name in self.dataset or len(glob.glob(str(path / "{}_*.{}".format(set_name, self.file_extension)))) > 0:
        continue
      self._maskCorpus(
        shaped_corpus, train_set = False, set_name = set_name, path = path, config = valset
      )
    return

  def configSampleSets(self) -> typing.List[pathlib.Path]:
    """
    Parses the types of datasets asked from sampler.

    These can be training, validation or a custom sample set
    (defined by type of target and hole/mask specs). 
    
    If the set does not exist, it is constructed.

    Returns:
      A list of paths for the requested datasets.
    Raises:
      FileNotFoundError: 
        In case sampler asks for validation set, 
        but this had not been constructed during training.
    """
    if self.sampler.config.HasField("train_set"):
      path = self.cache.path
      sampledDataset = "train_dataset"
    elif self.sampler.config.HasField("validation_set"):
      path = self.cache.path
      sampledDataset = "validation_dataset"
    elif self.sampler.config.HasField("sample_set"):
      path = self.cache.path
      sampledDataset = "pred_{}_{}".format(
        self.sampler.config.sample_set.max_predictions_per_seq,
        "mask" if self.sampler.config.sample_set.HasField("mask") 
               else "hole_{}".format(self.sampler.config.sample_set.hole.hole_length)
      )
    elif self.sampler.config.HasField("sample_corpus"):
      path = self.sampler.corpus_directory
      sampledDataset = "sample_corpus"
    else:
      raise ValueError("Unknown dataset type")

    return self.getDatasetPath(sampledDataset, path)

  def getDatasetPath(self,
                     set_name: str,
                     path: pathlib.Path,
                     ) -> typing.List[pathlib.Path]:
    """
    Based on a set name, search cache path for its existence.
    If not existing, get original pickled corpus, do the masking
    and save dataset in pt/tf_record format.

    Returns list of created datasets.
    """
    path_search = lambda: sorted(
      glob.glob(
        str(path / "{}_*.{}".format(set_name, self.file_extension))
      )
    )
    path_list = path_search()
    if len(path_list) == 0:

      if set_name == "validation_dataset":
        raise FileNotFoundError("Corpus had not been split in train-val, therefore validation dataset is not found.")
      elif set_name == "train_dataset":
        raise FileNotFoundError("Trying to sample training dataset, but it doesn't exist!")

      shaped_corpus = self.createCorpus(path)

      if self.sampler.config.HasField("sample_set"):
        config_list = [self.sampler.config.sample_set]
      elif self.sampler.config.HasField("sample_corpus"):
        config_list = [self.sampler.config.sample_corpus.config]
      else:
        raise ValueError("sampler sets can either be sample_set or sample_corpus")
      self.configValidationSets(config_list, shaped_corpus, path)
    return path_search()

  def createCorpus(self, path: pathlib.Path) -> np.array:
    """
    Constructs training corpus in text format, stores it in
    shaped_corpus

    Each corpus datapoint is either a single kernel or a random
    sequence of size sequence_length (legacy).

    If corpus has been previously pickled and stored, then it is loaded.
    """
    start_time = time.time()

    # Set corpus dimension parameters
    sequence_length = self.training_opts.sequence_length
    batch_size      = self.training_opts.batch_size
    dupe_factor     = self.training_opts.dupe_factor
    shuffle         = self.training_opts.shuffle_corpus_contentfiles_between_epochs
    pad             = [self.atomizer.padToken   ]
    start           = [self.atomizer.startToken ]
    end             = [self.atomizer.endToken   ]
    shaped_corpus   = None

    if (path / "corpus.pkl").exists():
      with open(path / "corpus.pkl", 'rb') as infile:
        shaped_corpus = pickle.load(infile)
        self.num_epochs      = self.training_opts.num_train_steps // self.config.steps_per_epoch
        self.steps_per_epoch = self.config.steps_per_epoch
        l.getLogger().info(
          "Loaded from file corpus of {} examples in {} ms.".format(
                    humanize.intcomma(len(shaped_corpus)),
                    humanize.intcomma(int((time.time() - start_time) * 1000)),
                )
        )
      return shaped_corpus

    # generate a kernel corpus
    if (path / "text_corpus.pkl").exists():
      with open(path / "text_corpus.pkl", 'rb') as infile:
        encoded_corpus = [self.atomizer.AtomizeString(x) for x in pickle.load(infile)]
    else:
      encoded_corpus  = self.corpus.GetTrainingData()

    if self.config.datapoint_type == "kernel":

      # Reject larger than sequence length
      initial_length       = copy.deepcopy(len(encoded_corpus))
      encoded_corpus       = [list(x) for x in encoded_corpus if 
                             len(x) <= sequence_length - (2 if self.config.use_start_end else 0)] # Account for start and end token
      reduced_length       = copy.deepcopy(len(encoded_corpus))
      # Add start/end tokens
      if self.config.use_start_end:
        encoded_corpus     = [self._addStartEndToken(kf) for kf in encoded_corpus]
      # pad sequences to sequence length
      encoded_corpus       = np.array([x + pad * (sequence_length - len(x)) for x in encoded_corpus])
      # Clone datapoints dupe_factor times
      # shaped_corpus   = np.repeat(encoded_corpus, dupe_factor, axis = 0)
      shaped_corpus     = encoded_corpus
      # Shuffle
      if shuffle:
        self.rngen.shuffle(shaped_corpus)
      assert len(shaped_corpus) != 0, "Not enought data. All kernels have been rejected."

      # Set corpus epoch parameters
      self.num_epochs      = self.training_opts.num_train_steps // self.config.steps_per_epoch
      self.steps_per_epoch = self.config.steps_per_epoch

      assert shaped_corpus.ndim     == 2, "corpus dim: {}".format(shaped_corpus.shape)
      assert shaped_corpus.shape[1] == sequence_length, "Dim 1 shape mismatch: {}, target: {}".format(encoded_corpus.shape[1], sequence_length)

      l.getLogger().info("{} kernels were rejected (larger than sequence_length)".format(initial_length - reduced_length))
      l.getLogger().info(
        "Loaded corpus of shape {} multiplied by dupe factor: {} in {} ms.".format(
                  shaped_corpus.shape,
                  dupe_factor,
                  humanize.intcomma(int((time.time() - start_time) * 1000)),
              )
      )
    elif self.config.datapoint_type == "statement":
    ## This branch is legacy data processing

      if shuffle:
        self.rngen.shuffle(encoded_corpus)
      encoded_corpus = np.concatenate(encoded_corpus)
      # encoded_corpus = np.tile(encoded_corpus, dupe_factor)

      # Set corpus dimension parameters
      self.steps_per_epoch        = len(encoded_corpus) // (batch_size * sequence_length * dupe_factor)
      assert self.steps_per_epoch != 0, "Not enought data. Use smaller sequence_length and/or batch_size"
      self.num_epochs             = self.training_opts.num_train_steps // self.steps_per_epoch

      # clipped_corpus_length       = dupe_factor * self.steps_per_epoch * batch_size * sequence_length
      clipped_corpus_length       = self.steps_per_epoch * batch_size * sequence_length
      clipped_corpus              = encoded_corpus[:clipped_corpus_length]

      # shaped_corpus = np.split(clipped_corpus, batch_size * self.steps_per_epoch * dupe_factor, 0)
      shaped_corpus = np.split(clipped_corpus, batch_size * self.steps_per_epoch, 0)

      np_corpus = np.asarray(shaped_corpus)
      assert np_corpus.ndim == 2, "Wrong dimensions for shaped_corpus: {}".format(np_corpus.shape)
      assert np_corpus.shape[1] == sequence_length, "Second dimension is not equal to sequence length: {}".format(np_corpus.shape[1])

      l.getLogger().info(
        "Loaded corpus of {} tokens (clipped last {} tokens) in {} ms.".format(
                  humanize.intcomma(clipped_corpus_length),
                  humanize.intcomma(len(encoded_corpus) - clipped_corpus_length),
                  humanize.intcomma(int((time.time() - start_time) * 1000)),
              )
      )

    else:
      raise ValueError("Unrecognized datapoint_type: {}".format(self.config.datapoint_type))

    with open(path / "corpus.pkl", 'wb') as outf:
      pickle.dump(shaped_corpus, outf)
    return shaped_corpus

  def _maskCorpus(self,
                  corpus: np.array,
                  train_set: bool,
                  set_name: str,
                  path: pathlib.Path,
                  config   = None,
                  )-> None:
    """
    Entrypoint function that inserts masks or holes to the corpus.

    Arguments:
      corpus: [num_datapoints, sequence_length], 
              where num_datapoints = num_batches * dupe_factor * batch_size
    Returns:
      The masked corpus
    """

    # Set-up self.dataset entry
    self.dataset[set_name] = {
      'file': [],
      'txt' : [],
    }

    # Set up max predictions
    if config is None:
      config = self.config
      max_predictions = self.training_opts.max_predictions_per_seq
    else:
      max_predictions = config.max_predictions_per_seq

    # Apply dupe factor in stages to avoid stressing RAM.
    # Limit has been set to 4GB.
    single_item_bytes = self.estimatedSize(
      1, self.training_opts.sequence_length, self.training_opts.max_predictions_per_seq
    )
    corpus_bytes = single_item_bytes * len(corpus) + sys.getsizeof(corpus)
    # max_dupe is how many times (dupes) the corpus can fit into a dataset record file.
    max_dupe     = min((FLAGS.memory_limit * (1024**3)) // corpus_bytes, self.training_opts.dupe_factor)
    assert max_dupe != 0, "Increase RAM limit to fit corpus."

    iterations   = self.training_opts.dupe_factor // max_dupe
    remaining    = self.training_opts.dupe_factor % max_dupe

    extended_corpus  = np.repeat(corpus, max_dupe, axis = 0)
    remaining_corpus = np.repeat(corpus, remaining, axis = 0)

    l.getLogger().info("Estimated element size: {}. Dupe factor {} split into {} iterations of {} (plus {} remaining)".format(
        humanize.naturalsize(single_item_bytes), self.training_opts.dupe_factor, iterations, max_dupe, remaining
      )
    )
    pool = multiprocessing.Pool()
    distribution = None
    # Specify the desired masking routine
    if config.HasField("hole"):
      distribution = distributions.Distribution.FromHoleConfig(config.hole, path, set_name)
      maskedSeq    = lambda c: pool.imap_unordered(
        functools.partial(self.hole_func,
                          train_set            = train_set,
                          max_predictions      = max_predictions,
                          pickled_distribution = pickle.dumps(distribution),
                          pickled_atomizer     = pickle.dumps(self.atomizer),
                          training_opts        = self.training_opts,
                          is_torch             = self.is_torch,
                          ),
        c
      )
    elif config.HasField("mask"):
      maskedSeq    = lambda c: pool.imap_unordered(
        functools.partial(self.mask_func._maskSequence,
                          train_set          = train_set,
                          max_predictions    = max_predictions,
                          config             = config,
                          pickled_atomizer   = pickle.dumps(self.atomizer),
                          training_opts      = self.training_opts,
                          rngen              = self.rngen,
                          is_torch           = self.is_torch,
                          ),
        c
      )
    else:
      raise AttributeError("target predictions can only be mask or hole {}".format(self.config))

    # Masking generation data monitors
    start_idx_monitor = monitors.FrequencyMonitor(path, "target_mask_idx")
    idx_monitor       = monitors.FrequencyMonitor(path, "mask_idx")
    direction_monitor = monitors.FrequencyMonitor(path, "masking_direction")

    ## Core loop of masking.
    masked_corpus = []
    with progressbar.ProgressBar(max_value = len(corpus) * self.training_opts.dupe_factor) as bar:
      kernel_idx = 0
      try:
        for iteration in range(iterations + 1):
          masked_corpus = []
          # Select between normal iterations or dupe factor residual and shuffle
          if iteration != iterations:
            multiproc_corpus = maskedSeq(extended_corpus)
            if self.training_opts.shuffle_corpus_contentfiles_between_epochs:
              self.rngen.shuffle(extended_corpus)
          elif remaining != 0:
            multiproc_corpus = maskedSeq(remaining_corpus)
            if self.training_opts.shuffle_corpus_contentfiles_between_epochs:
              self.rngen.shuffle(remaining_corpus)
          else:
            continue

          # Do parallel masking over corpus
          for kernel, hole_lengths, masked_idxs in multiproc_corpus:
            if distribution:
              distribution.register(hole_lengths)

            try:
              if self.is_torch:
                actual_length = np.where(kernel['original_input'] == self.atomizer.padToken)[0][0]
              else:
                actual_length = np.where(kernel.original_input == self.atomizer.padToken)[0][0]
            except IndexError:
              actual_length = len(kernel['original_input'])
            for hole in masked_idxs:
              hole_idx = hole.pos_index
              selected_idx = hole.pos_index
              if hole.extend_left:
                selected_idx += hole.hole_length - 1 if hole.hole_length != 0 else 0

              start_idx_monitor.register(int(2 * round(100.0 * (selected_idx / actual_length) / 2.0)))
              idx_monitor.register([int(2 * round(100.0 * ((hole_idx + i) / actual_length) / 2.0)) for i in range(hole.hole_length)])
              direction_monitor.register(1 if hole.extend_left else 0)

            masked_corpus.append(kernel)
            bar.update(kernel_idx)
            kernel_idx += 1
            if kernel_idx == 1:
              self.LogBatchTelemetry(
                self.training_opts.batch_size, self.training_opts.sequence_length,
                max_predictions, self.steps_per_epoch, self.num_epochs
                )

          # write masked_corpus before flushing the list
          self.dataset[set_name]['file'].append(
            path / "{}_{}.{}".format(set_name, iteration, self.file_extension)
            )
          self.dataset[set_name]['txt'].append(
            path / "{}_{}.txt".format(set_name, iteration)
            )
          self._saveCorpusRecord({
              'corpus': masked_corpus,
              'file'  : path / "{}_{}.{}".format(set_name, iteration, self.file_extension),
              'txt'   : path / "{}_{}.txt".format(set_name, iteration)
            })
        pool.close()
      except KeyboardInterrupt as e:
        pool.terminate()
        raise e
      except Exception as e:
        pool.terminate()
        raise e

    if distribution:
      distribution.plot()
    start_idx_monitor.plot()
    idx_monitor.plot()
    direction_monitor.plot()
    return

  def estimatedSize(self, batch_size, sequence_length, max_predictions_per_seq):
    """
    Calculate estimated size of single training example as a dictionary.
    """
    return (
      2 * np.zeros([batch_size, 1], dtype = np.int64).nbytes + 
      5 * np.zeros([batch_size, sequence_length], dtype = np.int64).nbytes +
      2 * np.zeros([batch_size, max_predictions_per_seq], dtype = np.int64).nbytes
      )

  def LogBatchTelemetry(self,
                        batch_size: int,
                        sequence_length: int,
                        max_predictions_per_seq: int,
                        steps_per_epoch: int,
                        num_epochs: int,
                        ) -> None:
    """Log analytics about the batch."""
    if steps_per_epoch is not None and num_epochs is not None:
      l.getLogger().info(
        "Memory: {} per batch, {} per epoch, {} total.".format(
                humanize.naturalsize(self.estimatedSize(1, sequence_length, max_predictions_per_seq) * batch_size, binary = True),
                humanize.naturalsize(self.estimatedSize(1, sequence_length, max_predictions_per_seq) * batch_size * steps_per_epoch, binary = True),
                humanize.naturalsize(self.estimatedSize(1, sequence_length, max_predictions_per_seq) * batch_size * steps_per_epoch * num_epochs, binary = True),
            )
      )
    else:
      l.getLogger().info(
        "Memory: {} per batch.".format(
                humanize.naturalsize(self.estimatedSize(1, sequence_length, max_predictions_per_seq) * batch_size, binary = True),
            )
      )

  def _padToMaxPosition(self, input_sample):
    """
    Pads a given sequence to the maximum allowed sequence length, which is max_position_embeddings
    
    Arguments:
      input_sample: np.array or list that represents a sequence

    Returns:
      padded sequence in np.array format
    """
    return np.concatenate([input_sample, 
                          np.array([self.atomizer.padToken] * 
                              (self.max_position_embeddings - len(input_sample)), dtype = np.int64)
                          ])

  def _addStartEndToken(self, inp: list) -> list:
    """
    Inserts [START] and [END] token at the beginnning and end of a sequence
    
    Arguments:
      inp: input_sequence

    Returns:
      [START] + input_sequence + [END]
    """
    assert len(inp) != 0, "Empty list provided."
    assert self.atomizer.padToken not in inp, "Use this function before padding a sequence!"

    start = [self.atomizer.startToken] if inp[0]  != self.atomizer.startToken else []
    end   = [self.atomizer.endToken  ] if inp[-1] != self.atomizer.endToken   else []
    if isinstance(inp, list):
      return start + inp + end
    elif isinstance(inp, np.ndarray):
      raise NotImplementedError
