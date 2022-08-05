# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team and Foivos Tsimpourlas.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
import re
import typing

from deeplearning.benchpress.util import pytorch
from deeplearning.benchpress.util.pytorch import torch
from deeplearning.benchpress.models.torch_bert import generation_utils

from deeplearning.benchpress.util import logging as l

def find_pruneable_heads_and_indices(
  heads: typing.List[int], n_heads: int, head_size: int, already_pruned_heads: typing.Set[int]
) -> typing.Tuple[typing.Set[int], torch.LongTensor]:
  """
  Finds the heads and their indices taking :obj:`already_pruned_heads` into account.

  Args:
    heads (:obj:`typing.List[int]`): typing.List of the indices of heads to prune.
    n_heads (:obj:`int`): The number of heads in the model.
    head_size (:obj:`int`): The size of each head.
    already_pruned_heads (:obj:`typing.Set[int]`): A set of already pruned heads.

  Returns:
    :obj:`typing.Tuple[typing.Set[int], torch.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
  """
  mask = torch.ones(n_heads, head_size)
  heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
  for head in heads:
    # Compute how many pruned heads are before the head and move the index accordingly
    head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
    mask[head] = 0
  mask = mask.view(-1).contiguous().eq(1)
  index: torch.LongTensor = torch.arange(len(mask))[mask].long()
  return heads, index

class ModuleUtilsMixin:
  """
  A few utilities for :obj:`torch.torch.nn.Modules`, to be used as a mixin.
  """

  def num_parameters(self, only_trainable: bool = False) -> int:
    """
    Get the number of (optionally, trainable) parameters in the model.

    Args:
      only_trainable (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether or not to return only the number of trainable parameters

    Returns:
      :obj:`int`: The number of parameters.
    """
    params = filter(lambda x: x.requires_grad, self.parameters()) if only_trainable else self.parameters()
    return sum(p.numel() for p in params)

  @staticmethod
  def _hook_rss_memory_pre_forward(module, *args, **kwargs):
    try:
      import psutil
    except (ImportError):
      raise ImportError("You need to install psutil (pip install psutil) to use memory tracing.")

    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    module.mem_rss_pre_forward = mem.rss
    return None

  @staticmethod
  def _hook_rss_memory_post_forward(module, *args, **kwargs):
    try:
      import psutil
    except (ImportError):
      raise ImportError("You need to install psutil (pip install psutil) to use memory tracing.")

    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    module.mem_rss_post_forward = mem.rss
    mem_rss_diff = module.mem_rss_post_forward - module.mem_rss_pre_forward
    module.mem_rss_diff = mem_rss_diff + (module.mem_rss_diff if hasattr(module, "mem_rss_diff") else 0)
    return None

  def add_memory_hooks(self):
    """
    Add a memory hook before and after each sub-module forward pass to record increase in memory consumption.

    Increase in memory consumption is stored in a :obj:`mem_rss_diff` attribute for each module and can be reset to
    zero with :obj:`model.reset_memory_hooks_state()`.
    """
    for module in self.modules():
      module.register_forward_pre_hook(self._hook_rss_memory_pre_forward)
      module.register_forward_hook(self._hook_rss_memory_post_forward)
    self.reset_memory_hooks_state()

  def reset_memory_hooks_state(self):
    """
    Reset the :obj:`mem_rss_diff` attribute of each module (see
    :func:`~transformers.modeling_utils.ModuleUtilsMixin.add_memory_hooks`).
    """
    for module in self.modules():
      module.mem_rss_diff = 0
      module.mem_rss_post_forward = 0
      module.mem_rss_pre_forward = 0

  @property
  def device(self) -> pytorch.device:
    """
    :obj:`torch.device`: The device on which the module is (assuming that all the module parameters are on the same
    device).
    """
    try:
      return next(self.parameters()).pytorch.device
    except StopIteration:
      # For torch.nn.DataParallel compatibility in PyTorch 1.5

      def find_tensor_attributes(module: torch.nn.Module) -> typing.List[typing.Tuple[str, torch.Tensor]]:
        tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
        return tuples

      gen = self._named_members(get_members_fn=find_tensor_attributes)
      first_tuple = next(gen)
      return first_tuple[1].pytorch.device

  @property
  def dtype(self) -> torch.dtype:
    """
    :obj:`torch.torch.dtype`: The torch.dtype of the module (assuming that all the module parameters have the same torch.dtype).
    """
    try:
      return next(self.parameters()).dtype
    except StopIteration:
      # For torch.nn.DataParallel compatibility in PyTorch 1.5

      def find_tensor_attributes(module: torch.nn.Module) -> typing.List[typing.Tuple[str, torch.Tensor]]:
        tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
        return tuples

      gen = self._named_members(get_members_fn=find_tensor_attributes)
      first_tuple = next(gen)
      return first_tuple[1].dtype

  def invert_attention_mask(self, encoder_attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Invert an attention mask (e.g., switches 0. and 1.).

    Args:
      encoder_attention_mask (:obj:`torch.Tensor`): An attention mask.

    Returns:
      :obj:`torch.Tensor`: The inverted attention mask.
    """
    if encoder_attention_mask.dim() == 3:
      encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    if encoder_attention_mask.dim() == 2:
      encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
    # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
    # /transformer/transformer_layers.py#L270
    # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
    # encoder_extended_attention_mask.transpose(-1, -2))
    encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility

    if self.dtype == torch.float16:
      encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e4
    elif self.dtype == torch.float32:
      encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
    else:
      raise ValueError(
        "{} not recognized. `torch.dtype` should be set to either `torch.float32` or `torch.float16`".format(
          self.dtype
        )
      )

    return encoder_extended_attention_mask

  def get_extended_attention_mask(self, attention_mask: torch.Tensor, input_shape: typing.Tuple[int], device: device) -> torch.Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
      attention_mask (:obj:`torch.Tensor`):
        Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
      input_shape (:obj:`typing.Tuple[int]`):
        The shape of the input to the model.
      device: (:obj:`torch.device`):
        The device of the input to the model.

    Returns:
      :obj:`torch.Tensor` The extended attention mask, with a the same torch.dtype as :obj:`attention_mask.torch.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
      extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
      # Provided a padding mask of dimensions [batch_size, seq_length]
      # - if the model is a decoder, apply a causal mask in addition to the padding mask
      # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
      if self.config.is_decoder:
        batch_size, seq_length = input_shape
        seq_ids = torch.arange(seq_length, device=device)
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        # causal and attention masks must have same type with pytorch version < 1.3
        causal_mask = causal_mask.to(attention_mask.dtype)
        extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
      else:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
      raise ValueError(
        "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
          input_shape, attention_mask.shape
        )
      )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype = self.dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask

  def get_head_mask(
    self, head_mask: typing.Optional[torch.Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
  ) -> torch.Tensor:
    """
    Prepare the head mask if needed.

    Args:
      head_mask (:obj:`torch.Tensor` with shape :obj:`[num_heads]` or :obj:`[num_hidden_layers x num_heads]`, `optional`):
        The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
      num_hidden_layers (:obj:`int`):
        The number of hidden layers in the model.
      is_attention_chunked: (:obj:`bool`, `optional, defaults to :obj:`False`):
        Whether or not the attentions scores are computed by chunks or not.

    Returns:
      :obj:`torch.Tensor` with shape :obj:`[num_hidden_layers x batch x num_heads x seq_length x seq_length]`
      or list with :obj:`[None]` for each layer.
    """
    if head_mask is not None:
      head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
      if is_attention_chunked is True:
        head_mask = head_mask.unsqueeze(-1)
    else:
      head_mask = [None] * num_hidden_layers

    return head_mask

  def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
    """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
    if head_mask.dim() == 1:
      head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
      head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
      head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
    assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
    head_mask = head_mask.to(dtype = self.dtype)  # switch to fload if need + fp16 compatibility
    return head_mask

class PreTrainedModel(torch.nn.Module, ModuleUtilsMixin, generation_utils.GenerationMixin):
  r"""
  Base class for all models.

  :class:`~transformers.PreTrainedModel` takes care of storing the configuration of the models and handles methods
  for loading, downloading and saving models as well as a few methods common to all models to:

    * resize the input embeddings,
    * prune heads in the self-attention heads.

  Class attributes (overridden by derived classes):
    - **config_class** (:class:`~transformers.PretrainedConfig`) -- A subclass of
      :class:`~transformers.PretrainedConfig` to use as configuration class for this model architecture.
    - **load_tf_weights** (:obj:`typing.Callable`) -- A python `method` for loading a torch.TensorFlow checkpoint in a
      PyTorch model, taking as arguments:

      - **model** (:class:`~transformers.PreTrainedModel`) -- An instance of the model on which to load the
        torch.TensorFlow checkpoint.
      - **config** (:class:`~transformers.PreTrainedConfig`) -- An instance of the configuration associated
        to the model.
      - **path** (:obj:`str`) -- A path to the torch.TensorFlow checkpoint.

    - **base_model_prefix** (:obj:`str`) -- A string indicating the attribute associated to the base model in
      derived classes of the same architecture adding modules on top of the base model.
    - **authorized_missing_keys** (:obj:`typing.Optional[typing.List[str]]`) -- A list of re pattern of tensor names to ignore
      when loading the model (and avoid unnecessary warnings).
  """
  config_class = None
  base_model_prefix = ""
  authorized_missing_keys = None

  @property
  def dummy_inputs(self) -> typing.Dict[str, torch.Tensor]:
    """
    :obj:`typing.Dict[str, torch.Tensor]`: Dummy inputs to do a forward pass in the network.
    """
    return {"input_ids": torch.tensor(DUMMY_INPUTS)}

  def __init__(self, config, *inputs, **kwargs):
    super().__init__()
    # Save config in model
    self.config = config

  @property
  def base_model(self) -> torch.nn.Module:
    """
    :obj:`torch.torch.nn.Module`: The main body of the model.
    """
    return getattr(self, self.base_model_prefix, self)

  def get_input_embeddings(self) -> torch.nn.Module:
    """
    Returns the model's input embeddings.

    Returns:
      :obj:`torch.nn.Module`: A torch module mapping vocabulary to hidden states.
    """
    base_model = getattr(self, self.base_model_prefix, self)
    if base_model is not self:
      return base_model.get_input_embeddings()
    else:
      raise NotImplementedError

  def set_input_embeddings(self, value: torch.nn.Module):
    """
    typing.Set model's input embeddings

    Args:
      value (:obj:`torch.nn.Module`): A module mapping vocabulary to hidden states.
    """
    base_model = getattr(self, self.base_model_prefix, self)
    if base_model is not self:
      base_model.set_input_embeddings(value)
    else:
      raise NotImplementedError

  def get_output_embeddings(self) -> torch.nn.Module:
    """
    Returns the model's output embeddings.

    Returns:
      :obj:`torch.nn.Module`: A torch module mapping hidden states to vocabulary.
    """
    return None  # Overwrite for models with output embeddings

  def tie_weights(self):
    """
    Tie the weights between the input embeddings and the output embeddings.

    If the :obj:`torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning
    the weights instead.
    """
    output_embeddings = self.get_output_embeddings()
    if output_embeddings is not None:
      self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

    if self.config.is_encoder_decoder and self.config.tie_encoder_decoder:
      self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)

  @staticmethod
  def _tie_encoder_decoder_weights(encoder: torch.nn.Module, decoder: torch.nn.Module, base_model_prefix: str):
    uninitialized_encoder_weights: typing.List[str] = []
    assert decoder.__class__ == encoder.__class__, f"{decoder.__class__} and {encoder.__class__} have to be equal."

    def tie_encoder_to_decoder_recursively(
      decoder_pointer: torch.nn.Module,
      encoder_pointer: torch.nn.Module,
      module_name: str,
      uninitialized_encoder_weights: typing.List[str],
      depth=0,
    ):
      assert isinstance(decoder_pointer, torch.nn.Module) and isinstance(
        encoder_pointer, torch.nn.Module
      ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.torch.nn.Module"
      if hasattr(decoder_pointer, "weight"):
        assert hasattr(encoder_pointer, "weight")
        encoder_pointer.weight = decoder_pointer.weight
        if hasattr(decoder_pointer, "bias"):
          assert hasattr(encoder_pointer, "bias")
          encoder_pointer.bias = decoder_pointer.bias
        return

      encoder_modules = encoder_pointer._modules
      decoder_modules = decoder_pointer._modules
      if len(decoder_modules) > 0:
        assert (
          len(encoder_modules) > 0
        ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

        all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
        encoder_layer_pos = 0
        for name, module in decoder_modules.items():
          if name.isdigit():
            encoder_name = str(int(name) + encoder_layer_pos)
            decoder_name = name
            if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])):
              # this can happen if the name corresponds to the position in a list module list of layers
              # in this case the decoder has added a cross-attention that the encoder does not have
              # thus skip this step and substract one layer pos from encoder
              encoder_layer_pos -= 1
              continue
          elif name not in encoder_modules:
            continue
          elif depth > 500:
            raise ValueError(
              "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `torch.nn.Modules` of your model."
            )
          else:
            decoder_name = encoder_name = name
          tie_encoder_to_decoder_recursively(
            decoder_modules[decoder_name],
            encoder_modules[encoder_name],
            module_name + "/" + name,
            uninitialized_encoder_weights,
            depth=depth + 1,
          )
          all_encoder_weights.remove(module_name + "/" + encoder_name)

        uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights)
    if len(uninitialized_encoder_weights) > 0:
      l.logger().warning(
        f"The following encoder weights were not tied to the decoder {uninitialized_encoder_weights}"
      )

  def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
    """ Tie or clone module weights depending of whether we are using TorchScript or not
    """
    if self.config.torchscript:
      output_embeddings.weight = torch.nn.Parameter(input_embeddings.weight.clone())
    else:
      output_embeddings.weight = input_embeddings.weight

    if getattr(output_embeddings, "bias", None) is not None:
      output_embeddings.bias.data = torch.torch.nn.functional.pad(
        output_embeddings.bias.data,
        (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],),
        "constant",
        0,
      )
    if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
      output_embeddings.out_features = input_embeddings.num_embeddings

  def resize_token_embeddings(self, new_num_tokens: typing.Optional[int] = None) -> torch.torch.nn.Embedding:
    """
    Resizes input token embeddings matrix of the model if :obj:`new_num_tokens != config.vocab_size`.

    Takes care of tying weights embeddings afterwards if the model class has a :obj:`tie_weights()` method.

    Arguments:
      new_num_tokens (:obj:`int`, `optional`):
        The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
        vectors at the end. Reducing the size will remove vectors from the end. If not provided or :obj:`None`,
        just returns a pointer to the input tokens :obj:`torch.torch.nn.Embedding` module of the model wihtout doing
        anything.

    Return:
      :obj:`torch.torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
    """
    base_model = getattr(self, self.base_model_prefix, self)  # get the base model if needed
    model_embeds = base_model._resize_token_embeddings(new_num_tokens)
    if new_num_tokens is None:
      return model_embeds

    # Update base model and current model config
    self.config.vocab_size = new_num_tokens
    base_model.vocab_size = new_num_tokens

    # Tie weights again if needed
    self.tie_weights()

    return model_embeds

  def _resize_token_embeddings(self, new_num_tokens):
    old_embeddings = self.get_input_embeddings()
    new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
    self.set_input_embeddings(new_embeddings)
    return self.get_input_embeddings()

  def _get_resized_embeddings(
    self, old_embeddings: torch.torch.nn.Embedding, new_num_tokens: typing.Optional[int] = None
  ) -> torch.torch.nn.Embedding:
    """
    Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
    initialized vectors at the end. Reducing the size will remove vectors from the end

    Args:
      old_embeddings (:obj:`torch.torch.nn.Embedding`):
        Old embeddings to be resized.
      new_num_tokens (:obj:`int`, `optional`):
        New number of tokens in the embedding matrix.

        Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
        vectors from the end. If not provided or :obj:`None`, just returns a pointer to the input tokens
        :obj:`torch.torch.nn.Embedding`` module of the model wihtout doing anything.

    Return:
      :obj:`torch.torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
      :obj:`new_num_tokens` is :obj:`None`
    """
    if new_num_tokens is None:
      return old_embeddings

    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    if old_num_tokens == new_num_tokens:
      return old_embeddings

    # Build new embeddings
    new_embeddings = torch.nn.Embedding(new_num_tokens, old_embedding_dim)
    new_embeddings.to(old_embeddings.weight.pytorch.device)

    # initialize all new embeddings (in particular added tokens)
    self._init_weights(new_embeddings)

    # Copy token embeddings from the previous weights
    num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
    new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]

    return new_embeddings

  def init_weights(self):
    """
    Initializes and prunes weights if needed.
    """
    # Initialize weights
    self.apply(self._init_weights)

    # Prune heads if needed
    if self.config.pruned_heads:
      self.prune_heads(self.config.pruned_heads)

    # Tie weights if needed
    self.tie_weights()

  def prune_heads(self, heads_to_prune: typing.Dict[int, typing.List[int]]):
    """
    Prunes heads of the base model.

    Arguments:
      heads_to_prune (:obj:`typing.Dict[int, typing.List[int]]`):
        typing.Dictionary with keys being selected layer indices (:obj:`int`) and associated values being the list
        of heads to prune in said layer (list of :obj:`int`). For instance {1: [0, 2], 2: [2, 3]} will
        prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.
    """
    # save new sets of pruned heads as union of previously stored pruned heads and newly pruned heads
    for layer, heads in heads_to_prune.items():
      union_heads = set(self.config.pruned_heads.get(layer, [])) | set(heads)
      self.config.pruned_heads[layer] = list(union_heads)  # Unfortunately we have to store it as list for JSON

    self.base_model._prune_heads(heads_to_prune)

  def save_pretrained(self, save_directory):
    """
    Save a model and its configuration file to a directory, so that it can be re-loaded using the
    `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

    Arguments:
      save_directory (:obj:`str`):
        Directory to which to save. Will be created if it doesn't exist.
    """
    if os.path.isfile(save_directory):
      l.logger().error("Provided path ({}) should be a directory, not a file".format(save_directory))
      return
    os.makedirs(save_directory, exist_ok=True)

    # Only save the model itself if we are using distributed training
    model_to_save = self.module if hasattr(self, "module") else self

    # Attach architecture to the config
    model_to_save.config.architectures = [model_to_save.__class__.__name__]

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(save_directory, WEIGHTS_NAME)

    if getattr(self.config, "xla_device", False):

      if pytorch.xla_model.is_master_ordinal():
        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)
      # pytorch.xla_model.save takes care of saving only from master
      pytorch.xla_model.save(model_to_save.state_dict(), output_model_file)
    else:
      model_to_save.config.save_pretrained(save_directory)
      torch.save(model_to_save.state_dict(), output_model_file)

    l.logger().info("Model weights saved in {}".format(output_model_file))

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    r"""
    Instantiate a pretrained pytorch model from a pre-trained model configuration.

    The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated).
    To train the model, you should first set it back in training mode with ``model.train()``.

    The warning `Weights from XXX not initialized from pretrained model` means that the weights of XXX do not come
    pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
    task.

    The warning `Weights from XXX not used in YYY` means that the layer XXX is not used by YYY, therefore those
    weights are discarded.

    Parameters:
      pretrained_model_name_or_path (:obj:`str`, `optional`):
        Can be either:

          - A string with the `shortcut name` of a pretrained model to load from cache or download, e.g.,
            ``bert-base-uncased``.
          - A string with the `identifier name` of a pretrained model that was user-uploaded to our S3, e.g.,
            ``dbmdz/bert-base-german-cased``.
          - A path to a `directory` containing model weights saved using
            :func:`~transformers.PreTrainedModel.save_pretrained`, e.g., ``./my_model_directory/``.
          - A path or url to a `tensorflow index checkpoint file` (e.g, ``./tf_model/model.ckpt.index``). In
            this case, ``from_tf`` should be set to :obj:`True` and a configuration object should be provided
            as ``config`` argument. This loading path is slower than converting the torch.TensorFlow checkpoint in
            a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
          - :obj:`None` if you are both providing the configuration and state dictionary (resp. with keyword
            arguments ``config`` and ``state_dict``).
      model_args (sequence of positional arguments, `optional`):
        All remaning positional arguments will be passed to the underlying model's ``__init__`` method.
      config (:obj:`typing.Union[PretrainedConfig, str]`, `optional`):
        Can be either:

          - an instance of a class derived from :class:`~transformers.PretrainedConfig`,
          - a string valid as input to :func:`~transformers.PretrainedConfig.from_pretrained`.

        Configuration for the model to use instead of an automatically loaded configuation. Configuration can
        be automatically loaded when:

          - The model is a model provided by the library (loaded with the `shortcut name` string of a
            pretrained model).
          - The model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded
            by suppling the save directory.
          - The model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a
            configuration JSON file named `config.json` is found in the directory.
      state_dict (:obj:`typing.Dict[str, torch.Tensor]`, `optional`):
        A state dictionary to use instead of a state dictionary loaded from saved weights file.

        This option can be used if you want to create a model from a pretrained configuration but load your own
        weights. In this case though, you should check if using
        :func:`~transformers.PreTrainedModel.save_pretrained` and
        :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.
      cache_dir (:obj:`str`, `optional`):
        Path to a directory in which a downloaded pretrained model configuration should be cached if the
        standard cache should not be used.
      from_tf (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Load the model weights from a torch.TensorFlow checkpoint save file (see docstring of
        ``pretrained_model_name_or_path`` argument).
      force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether or not to force the (re-)download of the model weights and configuration files, overriding the
        cached versions if they exist.
      resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether or not to delete incompletely received files. Will attempt to resume the download if such a
        file exists.
      proxies (:obj:`typing.Dict[str, str], `optional`):
        A dictionary of proxy servers to use by protocol or endpoint, e.g.,
        :obj:`{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each
        request.
      output_loading_info(:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether ot not to also return a dictionnary containing missing keys, unexpected keys and error
        messages.
      local_files_only(:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether or not to only look at local files (e.g., not try doanloading the model).
      use_cdn(:obj:`bool`, `optional`, defaults to :obj:`True`):
        Whether or not to use Cloudfront (a Content Delivery Network, or CDN) when searching for the model on
        our S3 (faster). Should be set to :obj:`False` for checkpoints larger than 20GB.
      kwargs (remaining dictionary of keyword arguments, `optional`):
        Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
        :obj:`output_attention=True`). Behaves differently depending on whether a ``config`` is provided or
        automatically loaded:

          - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the
            underlying model's ``__init__`` method (we assume all relevant updates to the configuration have
            already been done)
          - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class
            initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of
            ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute
            with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration
            attribute will be passed to the underlying model's ``__init__`` function.

    Examples::

      from transformers import BertConfig, BertModel
      # Download model and configuration from S3 and cache.
      model = BertModel.from_pretrained('bert-base-uncased')
      # Model was saved using `save_pretrained('./test/saved_model/')` (for example purposes, not runnable).
      model = BertModel.from_pretrained('./test/saved_model/')
      # Update configuration during loading.
      model = BertModel.from_pretrained('bert-base-uncased', output_attention=True)
      assert model.config.output_attention == True
      # Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
      config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
      model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_tf=True, config=config)
    """
    config = kwargs.pop("config", None)
    state_dict = kwargs.pop("state_dict", None)
    cache_dir = kwargs.pop("cache_dir", None)
    from_tf = kwargs.pop("from_tf", False)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    output_loading_info = kwargs.pop("output_loading_info", False)
    local_files_only = kwargs.pop("local_files_only", False)
    use_cdn = kwargs.pop("use_cdn", True)

    model_kwargs = kwargs

    # Load model
    if pretrained_model_name_or_path is not None:
      if os.path.isdir(pretrained_model_name_or_path):
        if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
          # Load from a TF 1.0 checkpoint
          archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
        elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
          # Load from a TF 2.0 checkpoint
          archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
        elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
          # Load from a PyTorch checkpoint
          archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        else:
          raise EnvironmentError(
            "Error no file named {} found in directory {} or `from_tf` set to False".format(
              [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME + ".index"],
              pretrained_model_name_or_path,
            )
          )
      elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
        archive_file = pretrained_model_name_or_path
      elif os.path.isfile(pretrained_model_name_or_path + ".index"):
        assert (
          from_tf
        ), "We found a torch.TensorFlow checkpoint at {}, please set from_tf to True to load from this checkpoint".format(
          pretrained_model_name_or_path + ".index"
        )
        archive_file = pretrained_model_name_or_path + ".index"
      else:
        archive_file = hf_bucket_url(
          pretrained_model_name_or_path,
          filename=(TF2_WEIGHTS_NAME if from_tf else WEIGHTS_NAME),
          use_cdn=use_cdn,
        )

      try:
        # Load from URL or cache if already cached
        resolved_archive_file = cached_path(
          archive_file,
          cache_dir=cache_dir,
          force_download=force_download,
          proxies=proxies,
          resume_download=resume_download,
          local_files_only=local_files_only,
        )
        if resolved_archive_file is None:
          raise EnvironmentError
      except EnvironmentError:
        msg = (
          f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
          f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
          f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named one of {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME}.\n\n"
        )
        raise EnvironmentError(msg)

      if resolved_archive_file == archive_file:
        l.logger().info("loading weights file {}".format(archive_file))
      else:
        l.logger().info("loading weights file {} from cache at {}".format(archive_file, resolved_archive_file))
    else:
      resolved_archive_file = None

    # Instantiate model.
    model = cls(config, *model_args, **model_kwargs)

    if state_dict is None and not from_tf:
      try:
        state_dict = torch.load(resolved_archive_file, map_location="cpu")
      except Exception:
        raise OSError(
          "Unable to load weights from pytorch checkpoint file. "
          "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
        )

    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    if from_tf:
      if resolved_archive_file.endswith(".index"):
        # Load from a torch.TensorFlow 1.X checkpoint - provided by original authors
        model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
      else:
        # Load from our torch.TensorFlow 2.0 checkpoints
        try:
          from transformers import load_tf2_checkpoint_in_pytorch_model

          model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
        except ImportError:
          l.logger().error(
            "Loading a torch.TensorFlow model in PyTorch, requires both PyTorch and torch.TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
          )
          raise
    else:
      # Convert old format to new format if needed from a PyTorch state_dict
      old_keys = []
      new_keys = []
      for key in state_dict.keys():
        new_key = None
        if "gamma" in key:
          new_key = key.replace("gamma", "weight")
        if "beta" in key:
          new_key = key.replace("beta", "bias")
        if new_key:
          old_keys.append(key)
          new_keys.append(new_key)
      for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

      # copy state_dict so _load_from_state_dict can modify it
      metadata = getattr(state_dict, "_metadata", None)
      state_dict = state_dict.copy()
      if metadata is not None:
        state_dict._metadata = metadata

      # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
      # so we need to apply the function recursively.
      def load(module: torch.nn.Module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
          state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs,
        )
        for name, child in module._modules.items():
          if child is not None:
            load(child, prefix + name + ".")

      # Make sure we are able to load base models as well as derived models (with heads)
      start_prefix = ""
      model_to_load = model
      has_prefix_module = any(s.startswith(cls.base_model_prefix) for s in state_dict.keys())
      if not hasattr(model, cls.base_model_prefix) and has_prefix_module:
        start_prefix = cls.base_model_prefix + "."
      if hasattr(model, cls.base_model_prefix) and not has_prefix_module:
        model_to_load = getattr(model, cls.base_model_prefix)

      load(model_to_load, prefix=start_prefix)

      if model.__class__.__name__ != model_to_load.__class__.__name__:
        base_model_state_dict = model_to_load.state_dict().keys()
        head_model_state_dict_without_base_prefix = [
          key.split(cls.base_model_prefix + ".")[-1] for key in model.state_dict().keys()
        ]
        missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)

      # Some models may have keys that are not in the state by design, removing them before needlessly warning
      # the user.
      if cls.authorized_missing_keys is not None:
        for pat in cls.authorized_missing_keys:
          missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

      if len(unexpected_keys) > 0:
        l.logger().warning(
          f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when "
          f"initializing {model.__class__.__name__}: {unexpected_keys}\n"
          f"- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task "
          f"or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).\n"
          f"- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect "
          f"to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
        )
      else:
        l.logger().info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
      if len(missing_keys) > 0:
        l.logger().warning(
          f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} "
          f"and are newly initialized: {missing_keys}\n"
          f"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
        )
      else:
        l.logger().info(
          f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.\n"
          f"If your task is similar to the task the model of the checkpoint was trained on, "
          f"you can already use {model.__class__.__name__} for predictions without further training."
        )
      if len(error_msgs) > 0:
        raise RuntimeError(
          "Error(s) in loading state_dict for {}:\n\t{}".format(
            model.__class__.__name__, "\n\t".join(error_msgs)
          )
        )
    # make sure token embedding weights are still tied if needed
    model.tie_weights()

    # typing.Set model in evaluation mode to deactivate DropOut modules by default
    model.eval()

    if output_loading_info:
      loading_info = {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "error_msgs": error_msgs,
      }
      return model, loading_info

    if hasattr(config, "xla_device") and config.xla_device and is_torch_tpu_available():
      model = pytorch.xla_model.send_cpu_data_to_device(model, pytorch.xla_model.xla_device())
      model.to(pytorch.xla_model.xla_device())

    return model

def prune_linear_layer(layer: torch.torch.nn.Linear, index: torch.LongTensor, dim: int = 0) -> torch.torch.nn.Linear:
  """
  Prune a linear layer to keep only entries in index.

  Used to remove heads.

  Args:
    layer (:obj:`torch.torch.nn.Linear`): The layer to prune.
    index (:obj:`torch.LongTensor`): The indices to keep in the layer.
    dim (:obj:`int`, `optional`, defaults to 0): The dimension on which to keep the indices.

  Returns:
    :obj:`torch.torch.nn.Linear`: The pruned layer as a new layer with :obj:`requires_grad=True`.
  """
  index = index.to(layer.weight.pytorch.device)
  W = layer.weight.index_select(dim, index).clone().detach()
  if layer.bias is not None:
    if dim == 1:
      b = layer.bias.clone().detach()
    else:
      b = layer.bias[index].clone().detach()
  new_size = list(layer.weight.size())
  new_size[dim] = len(index)
  new_layer = torch.nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.pytorch.device)
  new_layer.weight.requires_grad = False
  new_layer.weight.copy_(W.contiguous())
  new_layer.weight.requires_grad = True
  if layer.bias is not None:
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(b.contiguous())
    new_layer.bias.requires_grad = True
  return new_layer

def apply_chunking_to_forward(
  forward_fn: typing.Callable[..., torch.Tensor], chunk_size: int, chunk_dim: int, *input_tensors
) -> torch.Tensor:
  """
  This function chunks the :obj:`input_tensors` into smaller input tensor parts of size :obj:`chunk_size` over the
  dimension :obj:`chunk_dim`. It then applies a layer :obj:`forward_fn` to each chunk independently to save memory.

  If the :obj:`forward_fn` is independent across the :obj:`chunk_dim` this function will yield the same result as
  directly applying :obj:`forward_fn` to :obj:`input_tensors`.

  Args:
    forward_fn (:obj:`typing.Callable[..., torch.Tensor]`):
      The forward function of the model.
    chunk_size (:obj:`int`):
      The chunk size of a chunked tensor: :obj:`num_chunks = len(input_tensors[0]) / chunk_size`.
    chunk_dim (:obj:`int`):
      The dimension over which the :obj:`input_tensors` should be chunked.
    input_tensors (:obj:`typing.Tuple[torch.Tensor]`):
      The input tensors of ``forward_fn`` which will be chunked.
  Returns:
    :obj:`torch.Tensor`: A tensor with the same shape as the :obj:`foward_fn` would have given if applied`.


  Examples::

    # rename the usual forward() fn to forward_chunk()
    def forward_chunk(self, hidden_states):
      hidden_states = self.decoder(hidden_states)
      return hidden_states

    # implement a chunked forward function
    def forward(self, hidden_states):
      return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
  """

  assert len(input_tensors) > 0, "{} has to be a tuple/list of tensors".format(input_tensors)
  tensor_shape = input_tensors[0].shape
  assert all(
    input_tensor.shape == tensor_shape for input_tensor in input_tensors
  ), "All input tenors have to be of the same shape"

  # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compability
  num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
  assert num_args_in_forward_chunk_fn == len(
    input_tensors
  ), "forward_chunk_fn expects {} arguments, but only {} input tensors are given".format(
    num_args_in_forward_chunk_fn, len(input_tensors)
  )

  if chunk_size > 0:
    assert (
      input_tensors[0].shape[chunk_dim] % chunk_size == 0
    ), "The dimension to be chunked {} has to be a multiple of the chunk size {}".format(
      input_tensors[0].shape[chunk_dim], chunk_size
    )

    num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

    # chunk input tensor into tuples
    input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
    # apply forward fn to every tuple
    output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
    # concatenate output at same dimension
    return torch.cat(output_chunks, dim=chunk_dim)

  return forward_fn(*input_tensors)
