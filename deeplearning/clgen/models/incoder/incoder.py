"""API Calls to FAIR-Incoder."""
import typing
import time
import tqdm
import transformers
import numpy as np
from absl import flags

from deeplearning.clgen.samplers import samplers
from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.util import plotter
from deeplearning.clgen.util import environment
from deeplearning.clgen.models import backends
from deeplearning.clgen.models import telemetry
from deeplearning.clgen.util import distrib
from deeplearning.clgen.models.incoder import example_api
from deeplearning.clgen.models.torch_bert.data_generator import torchLMDataGenerator

from deeplearning.clgen.util import logging as l

transformers.set_seed(np.random.RandomState().randint(0, 2**32-1) % (1 + environment.WORLD_RANK))

class Incoder(backends.BackendBase):
  """
  API Class for incoder collected from huggingface.
  """
  class TrainEstimator(typing.NamedTuple):
    """Named tuple to wrap Incoder pipeline."""
    model          : typing.TypeVar('nn.Module')
    data_generator : torchLMDataGenerator
    optimizer      : typing.Any
    scheduler      : typing.Any

  class SampleEstimator(typing.NamedTuple):
    """Named tuple for sampling Incoder."""
    model          : typing.List[typing.TypeVar('nn.Module')]
    data_generator : torchLMDataGenerator

  def __init__(self, *args, **kwargs):
    super(Incoder, self).__init__(*args, **kwargs)

    from deeplearning.clgen.util import pytorch
    if not pytorch.initialized:
      pytorch.initPytorch()

    self.pytorch             = pytorch
    self.torch               = pytorch.torch
    self.torch_tpu_available = pytorch.torch_tpu_available

    self.torch.manual_seed(np.random.RandomState().randint(0, 2**32-1) % (1 + environment.WORLD_RANK))
    self.torch.cuda.manual_seed_all(np.random.RandomState().randint(0, 2**32-1) % (1 + environment.WORLD_RANK))

    self.incoder_version   = kwargs.pop("incoder_version")

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
    m = transformers.AutoModelForCausalLM.from_pretrained(
      "/checkpoint/foivos/unsharded_checkpoints/incoder1B_m2.fp16.transformer_lm_gpt.adam.beta0.9_0.98.wd0.01.clip1.0.lr0.0008.warmup1500.sampletok2048.breakeos_blocked.bs8.updatefreq1.seed3.ngpu128", **kwargs
    ).to(self.pytorch.offset_device)

    if self.pytorch.num_nodes > 1:
      # m = self.torch.nn.parallel.DistributedDataParallel(
      #   m,
      #   device_ids = [self.pytorch.offset_device],
      #   output_device = self.pytorch.offset_device,
      # )
      pass
    elif self.pytorch.num_gpus > 1:
      l.logger().warn("HuggingFace 'generate' function does not support DataParallel. If you want multi-GPU sampling, go to DDP.")

    self.sample = Incoder.SampleEstimator(m, data_generator)
    l.logger().info("Initialized model sampler in {}".format(self.sampler.cache.path))
    return

  def samplesWithCategorical(self):
    return True

  def model_step(self) -> 'torch.Tensor':
    raise NotImplementedError
    return

  def sample_model_step(self,
                        model     : typing.List[typing.TypeVar('torch.nn.Module')],
                        inputs    : typing.Dict[str, typing.TypeVar('torch.Tensor')],
                        is_live   : bool = False,
                        iteration : int = None,
                        ) -> typing.Dict[str, typing.List[typing.List[int]]]:
    """
    Specialized forward function.
    Dispatches model replicas across all GPUs, one process each.

    Inputs must be three-dimensional:
    workload_size x batch_size x sequence_length
    """
    start = time.time()
    total_seqs = len(inputs['input_ids']) * self.sampler.batch_size
    max_to_generate = self.sampler.sequence_length - 3
    outputs = {
      'generated_samples': self.torch.zeros((total_seqs, self.sampler.sequence_length), dtype = self.torch.int64).to(self.pytorch.device),
      'sample_indices': self.torch.zeros((total_seqs, max_to_generate), dtype = self.torch.int64).to(self.pytorch.device),
      'input_ids': [], 'masked_lm_lengths': []
    }
    if iteration is not None:
      desc = "Sampling iteration: {}".format(iteration)
    else:
      desc = "Sampling"
    encoding_time = 0.0
    inference_time = 0.0
    decoding_time = 0.0
    s_idx = 0
    for batch in tqdm.tqdm(inputs['input_ids'], total = len(inputs['input_ids']), desc = desc):
      for seq in batch:
        t1 = time.time()
        seq = [x for x in seq if x != self.tokenizer.padToken]
        # if seq[-1] == self.tokenizer.holeToken:
          # incode = self.tokenizer.ArrayToCode(seq[:-1])
        # else:
        incode = self.tokenizer.ArrayToCode(seq).replace("<|mask:0|>", "<insert>") # This is a text where pad has been stripped off.
        incode = "<| file ext=.cl |>\n{}\n<|/ file |>".format(incode[incode.index("kernel void"):])
        t2 = time.time()
        encoding_time += t2 - t1
        incoded = example_api.infill(
          model,
          incode,
          self.tokenizer.get_hf_tokenizer(),
          max_to_generate = max_to_generate - len(seq) - 13,
          temperature     = self.temperature,
          extra_sentinel  = True,
          max_retries     = 1,
        )
        t3 = time.time()
        inference_time += t3 - t2
        try:
          text    = opencl.ExtractSingleKernels(incoded['text'])[0] # Collect only the first kernel generated, ignore the rest.
        except IndexError:
          l.logger().warn(incoded['text'])
          text = incoded['text']
        text = text.replace("<| file ext=.cl |>\n", "").replace("\n<|/ file |>", "")
        sample  = self.tokenizer.TokenizeString(text)[:self.sampler.sequence_length]
        sample += [self.tokenizer.padToken] * (self.sampler.sequence_length - len(sample))
        sample  = self.torch.LongTensor(sample).to(self.pytorch.device)

        indices  = self.tokenizer.TokenizeString(incoded['infills'][0])[:max_to_generate]
        indices += [self.tokenizer.padToken] * (max_to_generate - len(indices))
        indices  = self.torch.LongTensor(indices).to(self.pytorch.device)

        outputs['generated_samples'][s_idx] = sample
        outputs['sample_indices'][s_idx]    = indices
        s_idx += 1
        t4 = time.time()
        decoding_time += t4 - t3

    t5 = time.time()
    outputs['input_ids']         = inputs['input_ids'].reshape(-1, self.sampler.sequence_length).to(self.pytorch.device)
    outputs['masked_lm_lengths'] = inputs['masked_lm_lengths'].reshape(-1, 1).to(self.pytorch.device)

    if self.pytorch.num_nodes > 1:
      distrib.barrier()
      generated_samples = [self.torch.zeros(tuple(outputs['generated_samples'].shape), dtype = self.torch.int64).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
      sample_indices    = [self.torch.zeros(tuple(outputs['sample_indices'].shape),    dtype = self.torch.int64).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
      input_ids         = [self.torch.zeros(tuple(outputs['input_ids'].shape),         dtype = self.torch.int64).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]
      masked_lm_lengths = [self.torch.zeros(tuple(outputs['masked_lm_lengths'].shape), dtype = self.torch.int64).to(self.pytorch.device) for _ in range(self.torch.distributed.get_world_size())]

      self.torch.distributed.all_gather(generated_samples, outputs["generated_samples"])
      self.torch.distributed.all_gather(sample_indices,    outputs["sample_indices"])
      self.torch.distributed.all_gather(input_ids,         outputs["input_ids"])
      self.torch.distributed.all_gather(masked_lm_lengths, outputs["masked_lm_lengths"])

      outputs['generated_samples'] = self.torch.cat(generated_samples)
      outputs['sample_indices']    = self.torch.cat(sample_indices)
      outputs['input_ids']         = self.torch.cat(input_ids)
      outputs['masked_lm_lengths'] = self.torch.cat(masked_lm_lengths)

    outputs['generated_samples'] = list(outputs['generated_samples'].cpu().numpy())
    outputs['sample_indices']    = list(outputs['sample_indices'].cpu().numpy())
    outputs['input_ids']         = list(outputs['input_ids'].cpu().numpy())
    outputs['masked_lm_lengths'] = list(outputs['masked_lm_lengths'].cpu().numpy())

    end = time.time()
    post_process_time = end - t5
    l.logger().warn("Profiling times:\n\tencoding_time: {}\n\tinference_time: {}\n\tdecoding_time: {}\n\tpost_process_time: {}".format(
      encoding_time, inference_time, decoding_time, post_process_time,
      )
    )
    return outputs, end-start

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
    data_generator = torchLMDataGenerator.SampleMaskLMBatchGenerator(
                       self.config.training, sampler, self.tokenizer, seed, sample_batch_size,
                       sampler.sequence_length, self.cache.path, corpus,
                      #  self.feature_encoder,
                      #  self.feature_tokenizer,
                      #  self.feature_sequence_length,
                     )
    ##! TODO: Maybe initialize inline here instead of elaborating in separate function.
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
        step_out, time = self.sample_model_step(
            self.sample.model,
            self.step_inputs,
            is_live = self.sampler.is_live
        )
        if self.pytorch.num_nodes > 1:
          distrib.barrier()
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
            distrib.barrier()
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
  
  def __repr__(self) -> str:
    return "Incoder1B"

class Incoder6B(Incoder):
  """
  Specified class for regular 6B parameter Incoder.
  """
  def __init__(self, *args, **kwargs):
    kwargs["incoder_version"] = "facebook/incoder-6B"
    super(Incoder6B, self).__init__(*args, **kwargs)
    return

  def __repr__(self) -> str:
    return "Incoder6B"
