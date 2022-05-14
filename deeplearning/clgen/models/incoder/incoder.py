"""API Calls to FAIR-Incoder."""
import os
import shutil
import humanize
import typing
import pathlib
import datetime
import time
import numpy as np
from absl import flags
import tqdm

from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.samplers import samplers
from deeplearning.clgen.samplers import sample_observers
from deeplearning.clgen.util import pbutil
from deeplearning.clgen.util import plotter
from deeplearning.clgen.util import distrib
from deeplearning.clgen.util import environment
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import sampler_pb2
from deeplearning.clgen.features import extractor
from deeplearning.clgen.models import backends
from deeplearning.clgen.models import telemetry
from deeplearning.clgen.preprocessors import opencl

from deeplearning.clgen.util import logging as l

class Incoder(backends.BackendBase):
  """
  API Class for incoder collected from huggingface.
  """
  def __init__(self, *args, **kwargs):
    super(Incoder, self).__init__(*args, **kwargs)

    from deeplearning.clgen.util import pytorch
    if not pytorch.initialized:
      pytorch.initPytorch()

    self.pytorch             = pytorch
    self.torch               = pytorch.torch
    self.torch_tpu_available = pytorch.torch_tpu_available

    self.torch.manual_seed(self.config.training.random_seed)
    self.torch.cuda.manual_seed_all(self.config.training.random_seed)

    self.incoder_version   = kwargs.pop("incoder_version")

    # self.bertAttrs         = {}
    # self.featureAttrs      = {}
    # self.bert_config       = None

    self.train             = None
    self.sample            = None
    self.predict_generator = None
    self.sampler           = None

    self.train_batch_size  = None
    self.eval_batch_size   = None
    self.learning_rate     = None
    self.num_train_steps   = None

    self.ckpt_path         = self.cache.path / "checkpoints"
    self.sample_path       = self.cache.path / "samples"

    self.logfile_path      = self.cache.path / "logs"
    if self.config.HasField("pre_train_corpus"):
      self.pre_logfile_path = self.logfile_path / "pre_train"

    self.telemetry         = telemetry.TrainingLogger(self.logfile_path)
    if self.config.HasField("pre_train_corpus"):
      self.pre_telemetry   = telemetry.TrainingLogger(self.logfile_path / "pre_train")

    self.is_validated      = False
    self.trained           = False
    l.logger().info("{} initialized".format(self.incoder_version))
    return

  def _ConfigModelParams(self, is_sampling):
    """General model hyperparameters initialization."""
    ##! Placeholder for now. If need be, will be populated.
    return

  def _ConfigSampleParams(self,
                          data_generator: torchLMDataGenerator,
                          sampler: samplers.Sampler,
                          ) -> None:
    """
    Model parameter initialization for inference.
    """
    self._ConfigModelParams(is_sampling = True)
    self.sampler = sampler
    self.temperature = sampler.temperature

    # if sampler.sequence_length > self.bertAttrs['max_position_embeddings']:
    #   raise ValueError(
    #       "Cannot use sequence length %d because the BERT model "
    #       "was only trained up to sequence length %d" %
    #       (sampler.sequence_length, self.bertAttrs['max_position_embeddings']))

    kwargs = {}
    if self.incoder_version == "facebook/incoder-6B":
      # the arguments added below will load a half precision version of the model,
      # which requires less RAM than loading the full float32 version.  this 
      # should fit in ~16GB of RAM
      # NOTE: half precision should *not* be used if you plan to fine-tune the
      # model. You'll need full precision and a lot of GPU memory. We have not
      # tested fine-tuning in `transformers` (the model was trained in fairseq)
      kwargs = dict(
          revision="float16", 
          torch_dtype=torch.float16,
          low_cpu_mem_usage=True,
      )
    m = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(self.pytorch.offset_device)

    if self.pytorch.num_nodes > 1:
      m = self.torch.nn.parallel.DistributedDataParallel(
        m,
        device_ids = [self.pytorch.offset_device],
        output_device = self.pytorch.offset_device,
      )
    elif self.pytorch.num_gpus > 1:
      m = self.torch.nn.DataParallel(m)

    self.sample = torchBert.SampleBertEstimator(m, data_generator)
    l.logger().info("Initialized model sampler in {}".format(self.sampler.cache.path))
    return

  def PreTrain(self, *args, **kwargs) -> None:
    l.logger().warn("Pre-training is not supported yet for Incoder. Moving on.")
    return

  def Train(self, *args, **kwargs) -> None:
    l.logger().warn("Pre-training is not supported yet for Incoder. Moving on.")
    return

  def Validate(self, *args, **kwargs) -> None:
    l.logger().warn("Pre-training is not supported yet for Incoder. Moving on.")
    return

  def InitSampling(self,
                   sampler : samplers.Sampler,
                   seed    : typing.Optional[int] = None,
                   corpus = None,
                   ) -> None:
    """This is called only once. Performs basic initialization of sampling"""
    sample_batch_size = sampler.batch_size
    ##! TODO: Replace with incoder data generator
    raise NotImplementedError
    data_generator = torchLMDataGenerator.SampleMaskLMBatchGenerator(
                       self.config.training, sampler, self.tokenizer, seed, sample_batch_size,
                       self.config.architecture.max_position_embeddings, self.cache.path, corpus,
                      #  self.feature_encoder,
                      #  self.feature_tokenizer,
                      #  self.feature_sequence_length,
                     )
    ##! TODO: Maybe initialize inline here instead of elaborating in separate function.
    raise NotImplementedError
    self._ConfigSampleParams(data_generator, sampler)
    if self.pytorch.num_gpus > 0:
      self.torch.cuda.empty_cache()
    self.step_inputs   = None
    self.loader        = None
    self.pred_iterator = None
    l.logger().info("Initialized model samples in {}".format(self.sample_path / self.sampler.hash))
    return

  def InitSampleBatch(self, sampler: samplers.Sampler, **kwargs) -> None:
    """Batch-specific initialization. Called once when a new batch is going to be generated"""
    workload_size = kwargs.get('workload_size', None)
    if self.loader is None:
      if self.torch_tpu_available:
        self.loader = self.pytorch.torch_ploader.ParallelLoader(
                          self.sample.data_generator.dataloader, [self.pytorch.device]
                    ).per_device_loader(self.pytorch.device)
      else:
        self.loader = self.sample.data_generator.dataloader

    if not sampler.is_active:
      if self.pred_iterator is None:
        self.pred_iterator = iter(self.loader)
      try:
        inputs = next(self.pred_iterator)
      except StopIteration:
        self.pred_iterator = iter(self.loader)
        inputs = next(self.pred_iterator)

      if workload_size is None:
        ## I think this dictionary holds tensors of the following size:
        ## [num_gpus x batch_size x seq_len] if only one node works.
        ## Otherwise, [1 x batch_size x seq_len] since each process manages its own GPU.
        padded_wsize = self.pytorch.num_gpus if environment.WORLD_SIZE == 1 else 1
      else:
        ## If a workload is specified, then after you pad to the dimension of GPU or num processes
        ## Divide the size by GPU size or num processes size.
        padded_wsize = (
          (max(1, workload_size // (self.pytorch.num_gpus * sampler.batch_size))) * self.pytorch.num_gpus
          if environment.WORLD_SIZE == 1
          else (workload_size // (self.pytorch.num_nodes * sampler.batch_size)) * self.pytorch.num_nodes)
      self.step_inputs = {
        x: inputs[x].unsqueeze(0).repeat(padded_wsize, 1, 1)
        for x in inputs
      }

      # This loop below is purely for proper printing reasons:
      sample_text = set(
        [self.tokenizer.tokensToString(
            seq.cpu().numpy(), ignore_token = self.tokenizer.padToken
          ) for seq in inputs['input_ids']]
      )
      for seq in sample_text:
        self.sampler.setStartText(seq)
        self.sampler.Specialize(self.tokenizer)
    return

  def SampleNextIndices(
    self, *unused_args, **unused_kwargs
  ) -> typing.Tuple[np.array, np.array, np.array, np.array]:
    """Called iteratively to build a single batch of samples, until termination criteria stops calling"""
    del unused_kwargs
    del unused_args

    if self.sample is None:
      raise ValueError("Incoder sampler has not been initialized.")

    with self.torch.no_grad():
      if self.sampler.is_active:
        try:
          return self.sample.data_generator.ActiveGeneration(self, self.sample)
        except StopIteration:
          raise StopIteration
      else:
        ##!TODO: just call model's forward function. No need to do more.
        raise NotImplementedError
        step_out, time = self.sample_model_step(
            self.sample.model,
            self.step_inputs,
            is_live = self.sampler.is_live
        )
        if self.pytorch.num_nodes > 1:
          self.torch.distributed.barrier()
          generated_samples = [self.torch.zeros(tuple(step_out['generated_samples'].shape), dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
          sample_indices    = [self.torch.zeros(tuple(step_out['sample_indices'   ].shape), dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
          self.torch.distributed.all_gather(generated_samples, step_out["generated_samples"])
          self.torch.distributed.all_gather(sample_indices,    step_out["sample_indices"])
          raise NotImplementedError("This will not work because generated_samples and sample indices are lists and not tensors")
        else:
          generated_samples = step_out['generated_samples']
          sample_indices    = step_out['sample_indices']

        if self.sampler.is_live and input("Show logits figure ? [y/!y]") == "y":
          if self.pytorch.num_nodes > 1:
            prediction_scores = [self.torch.zeros(tuple(step_out['prediction_scores'].shape), dtype = self.torch.float32).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
            self.torch.distributed.barrier()
            self.torch.distributed.all_gather(prediction_scores, step_out["prediction_scores"])
          else:
            prediction_scores = step_out['prediction_scores'].cpu()

          for hole, indcs in zip(prediction_scores, sample_indices):
            plotter.LogitsStepsDistrib(
              x = self.torch.nn.Softmax(dim = 1)(self.torch.FloatTensor(hole[:10])).numpy(),
              atoms = [self.tokenizer.decoder[i] for i in range(self.tokenizer.vocab_size)],
              sample_indices = [self.tokenizer.decoder[i] for i in indcs[0]][:10],
              plot_name = "sampling_distrib",
              title = "Sampling distribution dim 1",
              x_name = "Probs / sample step",
            )
        return (
          self.step_inputs['original_input'].cpu().view(-1, self.step_inputs['original_input'].shape[2]).numpy(),
          self.step_inputs['input_ids'].cpu().view(-1, self.sampler.sequence_length).numpy(),
          generated_samples,
          sample_indices
        )
class Incoder1B(Incoder):
  """
  Specified class for 'small' 1B parameter Incoder.
  """
  def __init__(self, *args, **kwargs):
    kwargs["incoder_version"] = "facebook/incoder-1B"
    super(Incoder1B, self).__init__(*args, **kwargs)
    return

class Incoder6B(Incoder):
  """
  Specified class for regular 6B parameter Incoder.
  """
  def __init__(self, *args, **kwargs):
    kwargs["incoder_version"] = "facebook/incoder-6B"
    super(Incoder6B, self).__init__(*args, **kwargs)
    return
