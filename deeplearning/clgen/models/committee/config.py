""" Configuration base class for committee models."""
class CommitteeConfig(object):

  model_type = "committee"

  @classmethod
  def FromConfig(cls, config: active_learning_pb2.Committee, downstream_task: downstream_tasks.DownstreamTask):
    config = CommitteeConfig(config, downstream_task)
    return config

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
