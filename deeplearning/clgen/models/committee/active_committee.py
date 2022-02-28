"""
Array of NN models used for Active Learning Query-By-Committee.

This module handles
a) the passive training of the committee,
b) the confidence level of the committee for a datapoint (using entropy)
"""
import typing
import pathlib
import copy

from deeplearning.clgen.models import backends
from deeplearning.clgen.util.pytorch import torch

class ActiveCommittee(backends.BackendBase):

  class CommitteeEstimator(typing.NamedTuple):
    """Named tuple to wrap BERT pipeline."""
    model          : typing.TypeVar('nn.Module')
    data_generator : torchLMDataGenerator
    optimizer      : typing.Any
    scheduler      : typing.Any

  class SampleCommitteeEstimator(typing.NamedTuple):
    """Named tuple for sampling BERT."""
    model          : typing.List[typing.TypeVar('nn.Module')]
    data_generator : torchLMDataGenerator

  def __init__(self, *args, **kwargs):

    super(torchBert, self).__init__(*args, **kwargs)
    
    from deeplearning.clgen.util import pytorch
    if not pytorch.initialized:
      pytorch.initPytorch()

    self.pytorch             = pytorch
    self.torch               = pytorch.torch
    self.torch_tpu_available = pytorch.torch_tpu_available

    self.torch.manual_seed(self.config.training.random_seed)
    self.torch.cuda.manual_seed_all(self.config.training.random_seed)

    self.ckpt_path         = self.cache.path / "checkpoints"
    self.sample_path       = self.cache.path / "samples"

    self.logfile_path      = self.cache.path / "logs"

    self.is_validated      = False
    self.trained           = False
    l.logger().info("Active Committee config initialized in {}".format(self.cache.path))
    return

  def _ConfigModelParams(self, is_sampling):
    """General model hyperparameters initialization."""
    raise NotImplementedError
    self.bertAttrs = {
          "vocab_size"                   : self.tokenizer.vocab_size,
          "hidden_size"                  : self.config.architecture.hidden_size,
          "num_hidden_layers"            : self.config.architecture.num_hidden_layers,
          "num_attention_heads"          : self.config.architecture.num_attention_heads,
          "intermediate_size"            : self.config.architecture.intermediate_size,
          "hidden_act"                   : self.config.architecture.hidden_act,
          "hidden_dropout_prob"          : self.config.architecture.hidden_dropout_prob,
          "attention_probs_dropout_prob" : self.config.architecture.attention_probs_dropout_prob,
          "max_position_embeddings"      : self.config.architecture.max_position_embeddings,
          "type_vocab_size"              : self.config.architecture.type_vocab_size,
          "initializer_range"            : self.config.architecture.initializer_range,
          "layer_norm_eps"               : self.config.architecture.layer_norm_eps,
          "pad_token_id"                 : self.tokenizer.padToken,
    }
    if self.feature_encoder:
      self.featureAttrs = {
        "feature_encoder"                 : self.feature_encoder,
        "feature_sequence_length"         : self.feature_sequence_length,
        "feature_embedding_size"          : self.config.architecture.feature_embedding_size,
        "feature_pad_idx"                 : self.feature_tokenizer.padToken,
        "feature_dropout_prob"            : self.config.architecture.feature_dropout_prob,
        "feature_vocab_size"              : len(self.feature_tokenizer),
        "feature_num_attention_heads"     : self.config.architecture.feature_num_attention_heads,
        "feature_transformer_feedforward" : self.config.architecture.feature_transformer_feedforward,
        "feature_layer_norm_eps"          : self.config.architecture.feature_layer_norm_eps,
        "feature_num_hidden_layers"       : self.config.architecture.feature_num_hidden_layers,
      }
    self.bert_config = config.BertConfig.from_dict(
      self.bertAttrs,
      **self.featureAttrs,
      xla_device         = self.torch_tpu_available,
      reward_compilation = FLAGS.reward_compilation,
      is_sampling        = is_sampling,
    )
    return

  def _ConfigTrainParams(self, 
                         data_generator: torchLMDataGenerator,
                         ) -> None:
    """
    Model parameter initialization for training and validation.
    """
    self._ConfigModelParams(is_sampling = False)

    self.train_batch_size                 = self.config.training.batch_size
    self.eval_batch_size                  = self.config.training.batch_size
    self.learning_rate                    = self.config.training.adam_optimizer.initial_learning_rate_micros / 1e6
    self.num_warmup_steps                 = self.config.training.num_warmup_steps
    self.max_grad_norm                    = 1.0

    self.steps_per_epoch                  = data_generator.steps_per_epoch
    self.current_step                     = None
    self.num_epochs                       = data_generator.num_epochs
    self.num_train_steps                  = self.steps_per_epoch * self.num_epochs
    self.max_eval_steps                   = FLAGS.max_eval_steps

    self.validation_results_file          = "val_results.txt"
    self.validation_results_path          = os.path.join(str(self.logfile_path), self.validation_results_file)

    self.train = []

    for model in model_committee:
      m = model(self.config).to(self.pytorch.offset_device)
      if self.pytorch.num_nodes > 1:
        distrib.barrier()
        m = self.torch.nn.parallel.DistributedDataParallel(
          m,
          device_ids    = [self.pytorch.offset_device],
          output_device = self.pytorch.offset_device,
        )
      elif self.pytorch.num_gpus > 1:
        m = self.torch.nn.DataParallel(m)

      opt, lr_scheduler = optimizer.create_optimizer_and_scheduler(
        model           = m,
        num_train_steps = self.num_train_steps,
        warmup_steps    = self.num_warmup_steps,
        learning_rate   = self.learning_rate,
      )
      self.train.append(
        ActiveCommittee.CommitteeEstimator(
          m, copy.deepcopy(data_generator), opt, lr_scheduler
        )
      )
    l.logger().info(self.GetShortSummary())
    return

  def Train(self, corpus, **kwargs) -> None:
    """
    Training point of active learning committee.
    """
    raise NotImplementedError

    self._ConfigTrainParams(
      torchLMDataGenerator.TrainMaskLMBatchGenerator(
        corpus, self.config.training,
        self.cache.path,
        self.config.training.num_pretrain_steps if pre_train else None,
        pre_train,
        self.feature_encoder,
        self.feature_tokenizer,
        self.feature_sequence_length,
      ), pre_train
    )

    return