# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and Foivos Tsimpourlas.
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
""" Configuration base class and utilities."""
class BertConfig(object):

  r"""
    This is the configuration class to store the configuration of a :class:`~transformers.BertModel`.
    It is used to instantiate an BERT model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the BERT `bert-base-uncased <https://huggingface.co/bert-base-uncased>`__ architecture.

    Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
    to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
    for more information.


    Args:
      vocab_size (:obj:`int`, optional, defaults to 30522):
        Vocabulary size of the BERT model. Defines the different tokens that
        can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.BertModel`.
      hidden_size (:obj:`int`, optional, defaults to 768):
        Dimensionality of the encoder layers and the pooler layer.
      num_hidden_layers (:obj:`int`, optional, defaults to 12):
        Number of hidden layers in the Transformer encoder.
      num_attention_heads (:obj:`int`, optional, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
      intermediate_size (:obj:`int`, optional, defaults to 3072):
        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
      hidden_act (:obj:`str` or :obj:`function`, optional, defaults to "gelu"):
        The non-linear activation function (function or string) in the encoder and pooler.
        If string, "gelu", "relu", "swish" and "gelu_new" are supported.
      hidden_dropout_prob (:obj:`float`, optional, defaults to 0.1):
        The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob (:obj:`float`, optional, defaults to 0.1):
        The dropout ratio for the attention probabilities.
      max_position_embeddings (:obj:`int`, optional, defaults to 512):
        The maximum sequence length that this model might ever be used with.
        Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
      type_vocab_size (:obj:`int`, optional, defaults to 2):
        The vocabulary size of the `token_type_ids` passed into :class:`~transformers.BertModel`.
      initializer_range (:obj:`float`, optional, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
      layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
      gradient_checkpointing (:obj:`bool`, optional, defaults to False):
        If True, use gradient checkpointing to save memory at the expense of slower backward pass.

    Example::

      >>> from transformers import BertModel, BertConfig

      >>> # Initializing a BERT bert-base-uncased style configuration
      >>> configuration = BertConfig()

      >>> # Initializing a model from the bert-base-uncased style configuration
      >>> model = BertModel(configuration)

      >>> # Accessing the model configuration
      >>> configuration = model.config

    Note:
      A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to
      initialize a model does **not** load the model weights.
      It only affects the model's configuration.

    Class attributes (overridden by derived classes)
      - **model_type** (:obj:`str`): An identifier for the model type, serialized into the JSON file, and used to
        recreate the correct object in :class:`~transformers.AutoConfig`.

    Args:
      output_hidden_states (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether or not the model should return all hidden-states.
      output_attentions (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether or not the model should returns all attentions.
      use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
        Whether or not the model should return the last key/values attentions (not used by all models).
      return_dict (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether or not the model should return a :class:`~transformers.file_utils.ModelOutput` instead of a
        plain tuple.
      is_encoder_decoder (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether the model is used as an encoder/decoder or not.
      is_decoder (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether the model is used as decoder or not (in which case it's used as an encoder).
      add_cross_attention (:obj:`bool`, `optional`, defaults to :obj:`False`):
        Whether cross-attention layers should be added to the model. Note, this option is only relevant for models that can be used as decoder models within the `:class:~transformers.EncoderDecoderModel` class, which consists of all models in ``AUTO_MODELS_FOR_CAUSAL_LM``.
      tie_encoder_decoder (:obj:`bool`, `optional`, defaults to :obj:`False`)
        Whether all encoder weights should be tied to their equivalent decoder weights. This requires the encoder and decoder model to have the exact same parameter names.
      prune_heads (:obj:`Dict[int, List[int]]`, `optional`, defaults to :obj:`{}`):
        Pruned heads of the model. The keys are the selected layer indices and the associated values, the list
        of heads to prune in said layer.

        For instance ``{1: [0, 2], 2: [2, 3]}`` will prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer
        2.
      xla_device (:obj:`bool`, `optional`):
        A flag to indicate if TPU are available or not.
      chunk_size_feed_forward (:obj:`int`, `optional`, defaults to :obj:`0`):
        The chunk size of all feed forward layers in the residual attention blocks.
        A chunk size of :obj:`0` means that the feed forward layer is not chunked.
        A chunk size of n means that the feed forward layer processes :obj:`n` < sequence_length embeddings at a time.
        For more information on feed forward chunking, see `How does Feed Forward Chunking work? <../glossary.html#feed-forward-chunking>`__ .

    Parameters for sequence generation
      - **max_length** (:obj:`int`, `optional`, defaults to 20) -- Maximum length that will be used by
        default in the :obj:`generate` method of the model.
      - **min_length** (:obj:`int`, `optional`, defaults to 10) -- Minimum length that will be used by
        default in the :obj:`generate` method of the model.
      - **do_sample** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Flag that will be used by default in
        the :obj:`generate` method of the model. Whether or not to use sampling ; use greedy decoding otherwise.
      - **early_stopping** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Flag that will be used by
        default in the :obj:`generate` method of the model. Whether to stop the beam search when at least
        ``num_beams`` sentences are finished per batch or not.
      - **num_beams** (:obj:`int`, `optional`, defaults to 1) -- Number of beams for beam search that will be
        used by default in the :obj:`generate` method of the model. 1 means no beam search.
      - **temperature** (:obj:`float`, `optional`, defaults to 1) -- The value used to module the next token
        probabilities that will be used by default in the :obj:`generate` method of the model. Must be strictly
        positive.
      - **top_k** (:obj:`int`, `optional`, defaults to 50) -- Number of highest probability vocabulary tokens to
        keep for top-k-filtering that will be used by default in the :obj:`generate` method of the model.
      - **top_p** (:obj:`float`, `optional`, defaults to 1) --  Value that will be used by default in the
        :obj:`generate` method of the model for ``top_p``. If set to float < 1, only the most probable tokens
        with probabilities that add up to ``top_p`` or higher are kept for generation.
      - **repetition_penalty** (:obj:`float`, `optional`, defaults to 1) -- Parameter for repetition penalty
        that will be used by default in the :obj:`generate` method of the model. 1.0 means no penalty.
      - **length_penalty** (:obj:`float`, `optional`, defaults to 1) -- Exponential penalty to the length that
        will be used by default in the :obj:`generate` method of the model.
      - **no_repeat_ngram_size** (:obj:`int`, `optional`, defaults to 0) -- Value that will be used by default
        in the :obj:`generate` method of the model for ``no_repeat_ngram_size``. If set to int > 0, all ngrams of
        that size can only occur once.
      - **bad_words_ids** (:obj:`List[int]`, `optional`) -- List of token ids that are not allowed to be
        generated that will be used by default in the :obj:`generate` method of the model. In order to get the
        tokens of the words that should not appear in the generated text, use
        :obj:`tokenizer.encode(bad_word, add_prefix_space=True)`.
      - **num_return_sequences** (:obj:`int`, `optional`, defaults to 1) -- Number of independently computed
        returned sequences for each element in the batch that will be used by default in the :obj:`generate`
        method of the model.

    Parameters for fine-tuning tasks
      - **architectures** (:obj:`List[str]`, `optional`) -- Model architectures that can be used with the
        model pretrained weights.
      - **finetuning_task** (:obj:`str`, `optional`) -- Name of the task used to fine-tune the model. This can be
        used when converting from an original (TensorFlow or PyTorch) checkpoint.
      - **id2label** (:obj:`List[str]`, `optional`) -- A map from index (for instance prediction index, or target
        index) to label.
      - **label2id** (:obj:`Dict[str, int]`, `optional`) -- A map from label to index for the model.
      - **num_labels** (:obj:`int`, `optional`) -- Number of labels to use in the last layer added to the model,
        typically for a classification task.
      - **task_specific_params** (:obj:`Dict[str, Any]`, `optional`) -- Additional keyword arguments to store for
        the current task.

    Parameters linked to the tokenizer
      - **prefix** (:obj:`str`, `optional`) -- A specific prompt that should be added at the beginning of each
        text before calling the model.
      - **bos_token_id** (:obj:`int`, `optional`)) -- The id of the `beginning-of-stream` token.
      - **pad_token_id** (:obj:`int`, `optional`)) -- The id of the `padding` token.
      - **eos_token_id** (:obj:`int`, `optional`)) -- The id of the `end-of-stream` token.
      - **decoder_start_token_id** (:obj:`int`, `optional`)) -- If an encoder-decoder model starts decoding with
        a different token than `bos`, the id of that token.

    PyTorch specific parameters
      - **torchscript** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Whether or not the model should be
        used with Torchscript.

    TensorFlow specific parameters
      - **use_bfloat16** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Whether or not the model should
        use BFloat16 scalars (only used by some TensorFlow models).
  """
  model_type = "bert"

  def __init__(
    self,
    vocab_size,
    hidden_size,
    num_hidden_layers,
    num_attention_heads,
    intermediate_size,
    hidden_act,
    hidden_dropout_prob,
    attention_probs_dropout_prob,
    max_position_embeddings,
    pad_token_id,
    type_vocab_size,
    initializer_range,
    layer_norm_eps,
    **kwargs
  ):

    ## Bert-specific attributes
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range
    self.layer_norm_eps = layer_norm_eps
    self.gradient_checkpointing = kwargs.pop("gradient_checkpointing", False)

    # Attributes with defaults
    self.reward_compilation = kwargs.pop("reward_compilation", -1)
    self.is_sampling = kwargs.pop("is_sampling", False)
    self.return_dict = kwargs.pop("return_dict", False)
    self.output_hidden_states = kwargs.pop("output_hidden_states", False)
    self.output_attentions = kwargs.pop("output_attentions", False)
    self.use_cache = kwargs.pop("use_cache", True)  # Not used by all models
    self.torchscript = kwargs.pop("torchscript", False)  # Only used by PyTorch models
    self.use_bfloat16 = kwargs.pop("use_bfloat16", False)
    self.pruned_heads = kwargs.pop("pruned_heads", {})

    # Attributes for feature vector encoding
    self.feature_encoder                 = kwargs.pop("feature_encoder", False)
    self.feature_sequence_length         = kwargs.pop("feature_sequence_length", 256)
    self.feature_embedding_size          = kwargs.pop("feature_embedding_size", 512)
    self.feature_pad_idx                 = kwargs.pop("feature_pad_idx", -1)
    self.feature_dropout_prob            = kwargs.pop("feature_dropout_prob", 0.1)
    self.feature_vocab_size              = kwargs.pop("feature_vocab_size",  768)
    self.feature_num_attention_heads     = kwargs.pop("feature_num_attention_heads", 4)
    self.feature_transformer_feedforward = kwargs.pop("feature_transformer_feedforward", 2048)
    self.feature_layer_norm_eps          = kwargs.pop("feature_layer_norm_eps", 1e-5)
    self.feature_num_hidden_layers       = kwargs.pop("feature_num_hidden_layers", 2)

    # Is decoder is used in encoder-decoder models to differentiate encoder from decoder
    self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
    self.is_decoder = kwargs.pop("is_decoder", False)
    self.add_cross_attention = kwargs.pop("add_cross_attention", False)
    self.tie_encoder_decoder = kwargs.pop("tie_encoder_decoder", False)

    # Parameters for sequence generation
    self.max_length = kwargs.pop("max_length", 20)
    self.min_length = kwargs.pop("min_length", 0)
    self.do_sample = kwargs.pop("do_sample", False)
    self.early_stopping = kwargs.pop("early_stopping", False)
    self.num_beams = kwargs.pop("num_beams", 1)
    self.temperature = kwargs.pop("temperature", 1.0)
    self.top_k = kwargs.pop("top_k", 50)
    self.top_p = kwargs.pop("top_p", 1.0)
    self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
    self.length_penalty = kwargs.pop("length_penalty", 1.0)
    self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
    self.bad_words_ids = kwargs.pop("bad_words_ids", None)
    self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
    self.chunk_size_feed_forward = kwargs.pop("chunk_size_feed_forward", 0)

    # Fine-tuning task arguments
    self.architectures = kwargs.pop("architectures", None)
    self.finetuning_task = kwargs.pop("finetuning_task", None)
    self.id2label = kwargs.pop("id2label", None)
    self.label2id = kwargs.pop("label2id", None)
    if self.id2label is not None:
      kwargs.pop("num_labels", None)
      self.id2label = dict((int(key), value) for key, value in self.id2label.items())
      # Keys are always strings in JSON so convert ids to int here.
    else:
      self.num_labels = kwargs.pop("num_labels", 2)

    # Tokenizer arguments TODO: eventually tokenizer and models should share the same config
    self.prefix = kwargs.pop("prefix", None)
    self.bos_token_id = kwargs.pop("bos_token_id", None)
    self.pad_token_id = kwargs.pop("pad_token_id", None)
    self.eos_token_id = kwargs.pop("eos_token_id", None)
    self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)
    self.chunk_size_feed_forward = kwargs.pop("chunk_size_feed_forwar", 0)

    # task specific arguments
    self.task_specific_params = kwargs.pop("task_specific_params", None)

    # TPU arguments
    self.xla_device = kwargs.pop("xla_device", None)

    # Additional attributes without default values
    for key, value in kwargs.items():
      try:
        setattr(self, key, value)
      except AttributeError as err:
        l.logger().error("Can't set {} with value {} for {}".format(key, value, self))
        raise err

  @classmethod
  def from_dict(cls, bert_dict, **extra_args):
    config = BertConfig(
      vocab_size                   = bert_dict['vocab_size'],
      hidden_size                  = bert_dict['hidden_size'],
      num_hidden_layers            = bert_dict['num_hidden_layers'],
      num_attention_heads          = bert_dict['num_attention_heads'],
      intermediate_size            = bert_dict['intermediate_size'],
      hidden_act                   = bert_dict['hidden_act'],
      hidden_dropout_prob          = bert_dict['hidden_dropout_prob'],
      attention_probs_dropout_prob = bert_dict['attention_probs_dropout_prob'],
      max_position_embeddings      = bert_dict['max_position_embeddings'],
      type_vocab_size              = bert_dict['type_vocab_size'],
      initializer_range            = bert_dict['initializer_range'],
      layer_norm_eps               = bert_dict['layer_norm_eps'],
      pad_token_id                 = bert_dict['pad_token_id'],
      **extra_args,
      )
    return config

  @property
  def use_return_dict(self) -> bool:
    """
    :obj:`bool`: Whether or not return :class:`~transformers.file_utils.ModelOutput` instead of tuples.
    """
    # If torchscript is set, force `return_dict=False` to avoid jit errors
    return self.return_dict and not self.torchscript

  @property
  def num_labels(self) -> int:
    """
    :obj:`int`: The number of labels for classification models.
    """
    return len(self.id2label)

  @num_labels.setter
  def num_labels(self, num_labels: int):
    self.id2label = {i: "LABEL_{}".format(i) for i in range(num_labels)}
    self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))
