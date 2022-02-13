"""Data generator specifically used for Mask LM models (namely BERT)."""
import sys
import json
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
from deeplearning.clgen.util import environment
from deeplearning.clgen.util import distrib
from deeplearning.clgen.features import extractor
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.models import sequence_masking
from deeplearning.clgen.models import lm_database
from absl import flags
from deeplearning.clgen.util import logging as l

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

flags.DEFINE_boolean(
  "store_datasets_to_DB",
  False,
  "Set True to store masked datasets to SQL Database for observation."
)

def AssertConfigIsValid(config: model_pb2.DataGenerator,
                        ) -> model_pb2.DataGenerator:
  """
  Parse data generator protobuf message.
  Raise Exception if format is wrong.
  """
  pbutil.AssertFieldConstraint(
    config,
    "datapoint_type",
    lambda x: x == "kernel" or x == "statement",
    "Valid options for datapoint_type are 'kernel' and 'statement'",
  )
  pbutil.AssertFieldConstraint(
    config,
    "datapoint_time",
    lambda x: x == "online" or x == "pre",
    "Valid options for datapoint_time are 'online' and 'pre'",
  )
  pbutil.AssertFieldIsSet(
    config,
    "use_start_end",
  )
  pbutil.AssertFieldIsSet(
    config,
    "steps_per_epoch",
  )
  pbutil.AssertFieldConstraint(
    config,
    "validation_split",
    lambda x : 0 <= x <= 100,
    "Validation split is expressed in [0-100]%."
  )
  if config.datapoint_type == "kernel":
    pbutil.AssertFieldIsSet(
      config,
      "truncate_large_kernels",
    )
  if len(config.validation_set) > 0:
    for val_opt in config.validation_set:
      if val_opt.HasField("mask"):
        pbutil.AssertFieldIsSet(
          val_opt.mask,
          "random_placed_mask",
        )
      elif val_opt.HasField("hole"):
        if val_opt.HasField("absolute_length"):
          pbutil.AssertFieldConstraint(
            val_opt.hole,
            "absolute_length",
            lambda x : x > 0,
            "absolute length is the upper bound range of a hole's length. Therefore should be > 0."
          )
        else:
          pbutil.AssertFieldConstraint(
            val_opt.hole,
            "relative_length",
            lambda x : 0.0 < x <= 1.0,
            "relative length must be between 0 and 100% of a kernel's actual length."
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
      elif val_opt.HasField("mask_seq"):
        if val_opt.HasField("absolute_length"):
          pbutil.AssertFieldConstraint(
            val_opt.mask_seq,
            "absolute_length",
            lambda x : x > 0,
            "absolute length is the upper bound range of a mask_seq's length. Therefore should be > 0."
          )
        else:
          pbutil.AssertFieldConstraint(
            val_opt.mask_seq,
            "relative_length",
            lambda x : 0.0 < x <= 1.0,
            "relative length must be between 0 and 100% of a kernel's actual length."
          )
        if val_opt.mask_seq.HasField("normal_distribution"):
          pbutil.AssertFieldIsSet(
            val_opt.mask_seq.normal_distribution,
            "mean",
          )
          pbutil.AssertFieldIsSet(
            val_opt.mask_seq.normal_distribution,
            "variance",
          )
        elif not val_opt.mask_seq.HasField("uniform_distribution"):
          raise ValueError("Hole length distribution has not been set.")
  # Parse masking technique for bert's data generator
  pbutil.AssertFieldIsSet(config, "mask_technique")
  if config.HasField("mask"):
    pbutil.AssertFieldIsSet(
      config.mask,
      "random_placed_mask",
    )
  elif config.HasField("hole"):
    if config.hole.HasField("absolute_length"):
      pbutil.AssertFieldConstraint(
        config.hole,
        "absolute_length",
        lambda x : x > 0,
        "absolute length is the upper bound range of a hole's length. Therefore should be > 0."
      )
    else:
      pbutil.AssertFieldConstraint(
        config.hole,
        "relative_length",
        lambda x : 0.0 < x <= 1.0,
        "relative length must be between 0 and 100% of a kernel's actual length."
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
    pbutil.AssertFieldIsSet(
      config.hole,
      "stage_training",
    )
  elif config.HasField("mask_seq"):
    if config.mask_seq.HasField("absolute_length"):
      pbutil.AssertFieldConstraint(
        config.mask_seq,
        "absolute_length",
        lambda x : x > 0,
        "absolute length is the upper bound range of a mask_seq's length. Therefore should be > 0."
      )
    else:
      pbutil.AssertFieldConstraint(
        config.mask_seq,
        "relative_length",
        lambda x : 0.0 < x <= 1.0,
        "relative length must be between 0 and 100% of a kernel's actual length."
      )
    if config.mask_seq.HasField("normal_distribution"):
      pbutil.AssertFieldIsSet(
        config.mask_seq.normal_distribution,
        "mean",
      )
      pbutil.AssertFieldIsSet(
        config.mask_seq.normal_distribution,
        "variance",
      )
    elif not config.mask_seq.HasField("uniform_distribution"):
      raise ValueError("Hole length distribution has not been set.")
    pbutil.AssertFieldIsSet(
      config.mask_seq,
      "stage_training",
    )
  return config

def _addStartEndPadToken(inp: list, tokenizer, trunc: int = None, seq_len: int = None) -> typing.Tuple[int, np.array]:
  """
  Inserts [START] and [END] token at the beginnning and end of a sequence
  
  Arguments:
    inp: input_sequence

  Returns:
    [START] + input_sequence + [END]
  """
  tokenizer = pickle.loads(tokenizer)
  try:
    if trunc is not None:
      inp = inp[:trunc]

    assert len(inp) != 0, "Empty list provided."
    assert tokenizer.padToken not in inp, "Use this function before padding a sequence!"

    start = [tokenizer.startToken] if inp[0]  != tokenizer.startToken else []
    end   = [tokenizer.endToken  ] if inp[-1] != tokenizer.endToken   else []

    if isinstance(inp, list):
      ret = start + inp + end
      rlen = len(ret)
      if seq_len is not None:
        ret += [tokenizer.padToken] * (seq_len - len(ret))
      return rlen, np.array(ret)
    elif isinstance(inp, np.ndarray):
      raise NotImplementedError
  except AssertionError:
    return None

class MaskLMDataGenerator(object):
  """Abstract class, shared among TORCH and TF BERT data generators."""
  @property
  def is_torch(self):
    if self.file_extension == "pt_record":
      return True
    return False
  
  def __init__(self, file_extension: str):

    self.file_extension = file_extension
    self.mask_func      = sequence_masking.MPMaskSequence
    self.hole_func      = sequence_masking.MPHoleSequence
    self.mask_seq_func  = sequence_masking.HoleSequenceSeqMasks

    self.dataset                 = None
    self.corpus                  = None
    self.tokenizer               = None
    self.config                  = None
    self.cache                   = None

    self.training_opts           = None
    self.pre_train               = None
    self.num_train_steps         = None
    self.steps_per_epoch         = None
    self.sample_batch_size       = None
    self.max_position_embeddings = None
    self.num_epochs              = None

    self.sampler                 = None
    self.rngen                   = None
    return

  def TrainMaskLMBatchGenerator(self,
                                corpus: "corpuses.Corpus",
                                training_opts: model_pb2.TrainingOptions,
                                cache_path: pathlib.Path,
                                num_train_steps: int = None,
                                pre_train: bool = False,
                                ) -> "data_generator.MaskLMDataGenerator":
    """Initializes data generator for training."""
    self.cache         = cache.mkcache(cache_path, "dataset")
    self.cache.path.mkdir(exist_ok = True, parents = True)

    self.dataset       = {}
    self.corpus        = corpus
    self.tokenizer     = corpus.tokenizer
    self.config        = training_opts.data_generator
    self.training_opts = training_opts
    self.rngen         = np.random # random.Random(training_opts.random_seed)
    self.pre_train     = pre_train
    if num_train_steps:
      self.num_train_steps = num_train_steps
    else:
      self.num_train_steps = self.training_opts.num_train_steps
    shaped_corpus = self.createCorpus(self.cache.path)
    if self.config.datapoint_time == "pre":
      # 'pre' pre-processes/masks training/validation/sampling corpus for the model to use.
      # 'online' stores the raw data and masks them on the fly.
      self.configDataset(shaped_corpus)
    return self

  def SampleMaskLMBatchGenerator(self,
                                 model_opts,
                                 sampler,
                                 tokenizer,
                                 seed: int,
                                 sample_batch_size: int,
                                 max_position_embeddings: int,
                                 cache_path: pathlib.Path,
                                 ) -> "data_generator.MaskLMBatchGenerator":
    """Initializes data generator for inference."""
    self.cache                   = cache.mkcache(cache_path, "dataset")
    self.cache.path.mkdir(exist_ok = True, parents = True)

    self.dataset                 = {}
    self.sampler                 = sampler
    self.corpus                  = sampler.sample_corpus
    self.tokenizer               = tokenizer
    self.config                  = model_opts.data_generator
    self.rngen                   = np.random
    self.sample_batch_size       = sample_batch_size
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
    if FLAGS.force_remake_dataset:
      l.logger().warn("Force remaking datasets can cause lots of problems on an already trained model. Are you sure you want to proceed ? [y/n]")
      a = input()
      if a.lower() != "yes" and a.lower() != "y":
        l.logger().warn("Overwriting dataset process was aborted. Good call.")
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
        "mask" if valset.HasField("mask") else "hole_{}".format(valset.hole.absolute_length
                                                                if valset.hole.HasField("absolute_length")
                                                                else valset.hole.relative_length
                                                                )
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
               else "hole_{}".format(self.sampler.config.sample_set.hole.absolute_length
                                     if self.sampler.config.sample_set.hole.HasField("absolute_length")
                                     else self.sampler.config.sample_set.hole.relative_length)
      )
    elif self.sampler.config.HasField("sample_corpus"):
      path = self.sampler.corpus_directory
      sampledDataset = "pred_{}_{}".format(
        self.sampler.config.sample_corpus.corpus_config.max_predictions_per_seq,
        "mask" if self.sampler.config.sample_corpus.corpus_config.HasField("mask")
               else "hole_{}".format(self.sampler.config.sample_corpus.corpus_config.hole.absolute_length
                                     if self.sampler.config.sample_corpus.hole.HasField("absolute_length")
                                     else self.sampler.config.sample_corpus.hole.relative_length)
      )
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
        config_list = [self.sampler.config.sample_corpus.corpus_config]
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
    sequence_length   = self.training_opts.sequence_length
    effect_seq_length = sequence_length - (2 if self.config.use_start_end else 0)
    batch_size        = self.training_opts.batch_size
    dupe_factor       = self.training_opts.dupe_factor
    shuffle           = self.training_opts.shuffle_corpus_contentfiles_between_epochs
    pad               = [self.tokenizer.padToken   ]
    start             = [self.tokenizer.startToken ]
    end               = [self.tokenizer.endToken   ]
    shaped_corpus     = None

    corpus_file = "{}corpus.pkl".format("pre_" if self.pre_train else "")
    # Monitor counts actual length distribution of kernel instances.
    kernel_length_monitor = monitors.FrequencyMonitor(path, "{}kernel_length".format("pre_" if self.pre_train else ""))
    # Token frequency distribution monitor.
    if not self.pre_train:
      feature_monitors = {
        ftype: monitors.CategoricalDistribMonitor(
                          path,
                          "{}{}_distribution".format("pre_" if self.pre_train else "", ftype)
                        )
        for ftype in extractor.extractors.keys()
      }

    if (path / corpus_file).exists():
      with open(path / corpus_file, 'rb') as infile:
        shaped_corpus = pickle.load(infile)
        if self.num_train_steps:
          self.num_epochs      = self.num_train_steps // self.config.steps_per_epoch
          self.steps_per_epoch = self.config.steps_per_epoch
        l.logger().info(
          "Loaded from file corpus of {} examples in {} ms.".format(
                    humanize.intcomma(len(shaped_corpus)),
                    humanize.intcomma(int((time.time() - start_time) * 1000)),
                )
        )
      return shaped_corpus

    # generate a kernel corpus
    if (path / "text_corpus.pkl").exists():
      # Only sampler writes a text_corpus.pkl, to do online or active sampling.
      # The reason is, corpus is saved in text format, to be picked up with the
      # right tokenizer. And that is the model's tokenizer.
      with open(path / "text_corpus.pkl", 'rb') as infile:
        encoded_corpus = [self.tokenizer.TokenizeString(x) for x in pickle.load(infile)]
    else:
      if self.pre_train:
        if self.num_train_steps:
          self.num_epochs      = self.num_train_steps // self.config.steps_per_epoch
        self.steps_per_epoch = self.config.steps_per_epoch
        if environment.WORLD_RANK == 0:
          if len(glob.glob(str(path / "pre_corpus_*.pkl"))) > 0:
            return []
          encoded_corpus = []
          cache_lengths  = {}
          chunk_size = 250000
          i, ch_idx = 0, 0
          bar  = tqdm.tqdm(total = self.corpus.encoded.size, desc = "Chunk pre-train corpus")
          pool = multiprocessing.Pool()
          l.logger().info("Processing pre-train corpus chunks...")
          for dp in pool.imap_unordered(
                              functools.partial(
                                _addStartEndPadToken,
                                tokenizer = pickle.dumps(self.tokenizer),
                                trunc     = effect_seq_length,
                                seq_len   = sequence_length),
                              self.corpus.GetTrainingDataGenerator()):
            if dp:
              rlen, enc_kernel = dp
              kernel_length_monitor.register(rlen)
              encoded_corpus.append(enc_kernel)
            i += 1
            if i % chunk_size == 0:
              encoded_corpus = np.array(encoded_corpus)
              corpus_file = "pre_corpus_{}.pkl".format(ch_idx)
              cache_lengths[corpus_file] = len(encoded_corpus)
              l.logger().info("Storing chunk {}, len: {}".format(ch_idx, encoded_corpus.shape))
              with open(path / corpus_file, 'wb') as outf:
                pickle.dump(encoded_corpus, outf, protocol = 4)
              with open(path / "pre_lengths_cache.json", 'w') as outf:
                json.dump(cache_lengths, outf)
              ch_idx += 1
              encoded_corpus = []
            bar.update(1)
          if encoded_corpus:
            encoded_corpus = np.array(encoded_corpus)
            l.logger().info("Storing chunk {}, len: {}".format(ch_idx, encoded_corpus.shape))
            corpus_file = "pre_corpus_{}.pkl".format(ch_idx)
            cache_lengths[corpus_file] = len(encoded_corpus)
            with open(path / corpus_file, 'wb') as outf:
              pickle.dump(encoded_corpus, outf, protocol = 4)
            with open(path / "pre_lengths_cache.json", 'w') as outf:
              json.dump(cache_lengths, outf)
          kernel_length_monitor.plot()
          pool.close()
          distrib.barrier()
        else:
          distrib.barrier()
          if len(glob.glob(str(path / "pre_corpus_*.pkl"))) > 0:
            return []
          else:
            raise FileNotFoundError(glob.glob(str(path / "pre_corpus_*.pkl")))
        return encoded_corpus
      else:
        encoded_corpus  = self.corpus.GetTrainingData(sequence_length = effect_seq_length if not self.config.truncate_large_kernels else None)

    if self.config.datapoint_type == "kernel":
      # Reject larger than sequence length
      initial_length = copy.deepcopy(len(encoded_corpus))

      if not self.pre_train:
        # Get features of fitting dataset within sequence length
        for feature in self.corpus.GetTrainingFeatures(effect_seq_length):
          for ftype, fvector in feature.items():
            feature_monitors[ftype].register(fvector)

      if self.config.truncate_large_kernels:
        encoded_corpus     = [list(x[:effect_seq_length]) for x in encoded_corpus if len(x[:effect_seq_length]) <= effect_seq_length] # Account for start and end token
      else:
        encoded_corpus     = [list(x) for x in encoded_corpus if len(x) <= effect_seq_length] # Account for start and end token

      reduced_length       = copy.deepcopy(len(encoded_corpus))
      # Add start/end tokens
      if self.config.use_start_end:
        encoded_corpus     = [self._addStartEndToken(kf) for kf in encoded_corpus]
      # Register the actual lengths before padding.
      kernel_length_monitor.register([len(x) for x in encoded_corpus])
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
      if self.num_train_steps:
        self.num_epochs      = self.num_train_steps // self.config.steps_per_epoch
      self.steps_per_epoch = self.config.steps_per_epoch
      assert shaped_corpus.ndim     == 2, "corpus dim: {}".format(shaped_corpus.shape)
      assert shaped_corpus.shape[1] == sequence_length, "Dim 1 shape mismatch: {}, target: {}".format(shaped_corpus.shape[1], sequence_length)

      l.logger().info("{} kernels were rejected (larger than sequence_length)".format(initial_length - reduced_length))
      l.logger().info(
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
      if self.num_train_steps:
        self.num_epochs            = self.num_train_steps // self.steps_per_epoch

      # clipped_corpus_length       = dupe_factor * self.steps_per_epoch * batch_size * sequence_length
      clipped_corpus_length = self.steps_per_epoch * batch_size * sequence_length
      clipped_corpus        = encoded_corpus[:clipped_corpus_length]

      # shaped_corpus = np.split(clipped_corpus, batch_size * self.steps_per_epoch * dupe_factor, 0)
      shaped_corpus = np.split(clipped_corpus, batch_size * self.steps_per_epoch, 0)

      # Register the actual lengths before padding.
      kernel_length_monitor.register([len(x) for x in shaped_corpus])

      np_corpus = np.asarray(shaped_corpus)
      assert np_corpus.ndim == 2, "Wrong dimensions for shaped_corpus: {}".format(np_corpus.shape)
      assert np_corpus.shape[1] == sequence_length, "Second dimension is not equal to sequence length: {}".format(np_corpus.shape[1])

      l.logger().info(
        "Loaded corpus of {} tokens (clipped last {} tokens) in {} ms.".format(
                  humanize.intcomma(clipped_corpus_length),
                  humanize.intcomma(len(encoded_corpus) - clipped_corpus_length),
                  humanize.intcomma(int((time.time() - start_time) * 1000)),
              )
      )
    else:
      raise ValueError("Unrecognized datapoint_type: {}".format(self.config.datapoint_type))

    kernel_length_monitor.plot()
    if not self.pre_train:
      for fm in feature_monitors.values():
        fm.plot()
    with open(path / corpus_file, 'wb') as outf:
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

    def apply_dupe_factor(arr: np.array, iters: int) -> np.array:
      if iters == 0:
        return np.asarray([], dtype = arr.dtype)
      start_len = len(arr)
      arr = np.expand_dims(arr, 0) # 2D->3D
      arr = np.repeat(arr, iters, 0) # -> Repeat 2D blocks over 3D space
      arr = arr.reshape(iters * start_len, -1) # Flatten repetitive 2D blocks, into 2D array
      return arr

    extended_corpus  = apply_dupe_factor(corpus, iterations)
    remaining_corpus = apply_dupe_factor(corpus, remaining)

    l.logger().info("Estimated element size: {}. Dupe factor {} split into {} iterations of {} (plus {} remaining)".format(
        humanize.naturalsize(single_item_bytes), self.training_opts.dupe_factor, iterations, max_dupe, remaining
      )
    )
    pool = multiprocessing.Pool()
    distribution = None
    # Specify the desired masking routine
    if config.HasField("hole"):
      distribution = distributions.Distribution.FromHoleConfig(
        config.hole, path, "hole_length_{}".format(set_name)
      )
      maskedSeq    = lambda c: pool.imap_unordered(
        functools.partial(self.hole_func,
                          train_set            = train_set,
                          max_predictions      = max_predictions,
                          pickled_distribution = pickle.dumps(distribution),
                          pickled_tokenizer    = pickle.dumps(self.tokenizer),
                          training_opts        = self.training_opts,
                          is_torch             = self.is_torch,
                          ),
        c
      )
    elif config.HasField("mask_seq"):
      distribution = distributions.Distribution.FromHoleConfig(
        config.mask_seq, path, "mask_seq_length_{}".format(set_name)
      )
      maskedSeq    = lambda c: pool.imap_unordered(
        functools.partial(self.mask_seq_func,
                          train_set            = train_set,
                          max_predictions      = max_predictions,
                          pickled_distribution = pickle.dumps(distribution),
                          pickled_tokenizer    = pickle.dumps(self.tokenizer),
                          training_opts        = self.training_opts,
                          is_torch             = self.is_torch,
                          ),
        c
      )
    elif config.HasField("mask"):
      maskedSeq    = lambda c: pool.imap_unordered(
        functools.partial(self.mask_func,
                          train_set          = train_set,
                          max_predictions    = max_predictions,
                          config             = config,
                          pickled_tokenizer  = pickle.dumps(self.tokenizer),
                          training_opts      = self.training_opts,
                          is_torch           = self.is_torch,
                          ),
        c
      )
    else:
      raise AttributeError("target predictions can only be mask or hole {}".format(self.config))

    # Token frequency distribution monitor.
    token_monitor         = monitors.NormalizedFrequencyMonitor(path, "{}_token_distribution".format(set_name))
    # Monitor counts target idxs of a hole as absolute index value.
    abs_start_idx_monitor = monitors.FrequencyMonitor(path, "{}_abs_target_mask_idx".format(set_name))
    # Monitors count of target indices (in percentile) that were hidden by a hole.
    start_idx_monitor     = monitors.FrequencyMonitor(path, "{}_target_mask_idx".format(set_name))
    # Monitor counts all absolute indices hidden by a hole.
    abs_idx_monitor       = monitors.FrequencyMonitor(path, "{}_abs_target_mask_idx".format(set_name))
    # Monitors count of indices (in percentile) that were hidden by a hole.
    idx_monitor           = monitors.FrequencyMonitor(path, "{}_mask_idx".format(set_name))
    # Monitors if left or right direction was picked for a hole expansion.
    direction_monitor     = monitors.FrequencyMonitor(path, "{}_masking_direction".format(set_name))

    if FLAGS.store_datasets_to_DB:
      lm_db = lm_database.LMDatabase("sqlite:///{}".format(self.cache.path / "{}.db".format(set_name)))

    ## Core loop of masking.
    masked_corpus = []
    bar = tqdm.tqdm(total = len(corpus) * self.training_opts.dupe_factor, desc = "Masking datapoints")
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
        for kernel, masked_idxs in multiproc_corpus:
          if distribution:
            distribution.register([mid.hole_length for mid in masked_idxs])

          try:
            if self.is_torch:
              actual_length = np.where(kernel['original_input'] == self.tokenizer.padToken)[0][0]
            else:
              actual_length = np.where(kernel.original_input == self.tokenizer.padToken)[0][0]
          except IndexError:
            actual_length = len(kernel['original_input'])

          token_monitor.register([
            self.tokenizer.decoder[int(x)]
            for x in kernel['input_ids'] if x != self.tokenizer.padToken]
          )
          for hole in masked_idxs:
            hole_idx = hole.pos_index
            selected_idx = hole.pos_index
            if hole.extend_left:
              selected_idx += hole.hole_length - 1 if hole.hole_length != 0 else 0
            abs_start_idx_monitor.register(selected_idx)
            start_idx_monitor.register(int(2 * round(100.0 * (selected_idx / actual_length) / 2.0)))
            abs_idx_monitor.register([hole_idx + i for i in range(hole.hole_length)])
            idx_monitor.register([int(2 * round(100.0 * ((hole_idx + i) / actual_length) / 2.0)) for i in range(hole.hole_length)])
            direction_monitor.register(1 if hole.extend_left else 0)

          masked_corpus.append(kernel)
          bar.update(1)
          kernel_idx += 1
          if kernel_idx == 1:
            self.LogBatchTelemetry(
              self.training_opts.batch_size, self.training_opts.sequence_length,
              max_predictions, self.steps_per_epoch, self.num_epochs
              )

        if FLAGS.store_datasets_to_DB:
          with lm_db.Session(commit = True) as s:
            count = lm_db.count
            for idx, kernel in enumerate(masked_corpus):
              s.add(
                lm_database.LMInstance(**lm_database.LMInstance.FromArgs(
                  id = count + idx,
                  original_input = self.tokenizer.tokensToString(kernel['original_input'], ignore_token = self.tokenizer.padToken),
                  input_ids = self.tokenizer.tokensToString(kernel['input_ids'],           ignore_token = self.tokenizer.padToken),
                  masked_lm_lengths = kernel['masked_lm_lengths'],
                  masked_lm_predictions = [self.tokenizer.tokensToString([x]) for x in kernel['mask_labels'] if x != -100],
                ))
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
    token_monitor.plot()
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
      l.logger().info(
        "Memory: {} per batch, {} per epoch, {} total.".format(
                humanize.naturalsize(self.estimatedSize(1, sequence_length, max_predictions_per_seq) * batch_size, binary = True),
                humanize.naturalsize(self.estimatedSize(1, sequence_length, max_predictions_per_seq) * batch_size * steps_per_epoch, binary = True),
                humanize.naturalsize(self.estimatedSize(1, sequence_length, max_predictions_per_seq) * batch_size * steps_per_epoch * num_epochs, binary = True),
            )
      )
    else:
      l.logger().info(
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
                          np.array([self.tokenizer.padToken] * 
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
    assert self.tokenizer.padToken not in inp, "Use this function before padding a sequence!"

    start = [self.tokenizer.startToken] if inp[0]  != self.tokenizer.startToken else []
    end   = [self.tokenizer.endToken  ] if inp[-1] != self.tokenizer.endToken   else []
    if isinstance(inp, list):
      return start + inp + end
    elif isinstance(inp, np.ndarray):
      raise NotImplementedError

  def GetShortSummary(self) -> str:
    return (
      "Data Generator: "
      "\n"
      f"  dupe_factor: {self.training_opts.dupe_factor}"
      "\n"
      f"  sequence_length: {self.training_opts.sequence_length}"
      "\n"
      f"  batch_size: {self.training_opts.batch_size}"
      "\n"
      "LM config:"
      "\n"
      f"  {self.config.hole if True else self.config.mask}"
      "\n"
    )
