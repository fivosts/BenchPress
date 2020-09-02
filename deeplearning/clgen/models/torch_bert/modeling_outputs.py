from dataclasses import dataclass
from collections import OrderedDict
import typing
import torch

class ModelOutput(OrderedDict):
  """
  Base class for all model outputs as dataclass. Has a ``__getitem__`` that allows indexing by integer or slice (like
  a tuple) or strings (like a dictionnary) that will ignore the ``None`` attributes. Otherwise behaves like a
  regular python dictionary.

  .. warning::
    You can't unpack a :obj:`ModelOutput` directly. Use the :meth:`~transformers.file_utils.ModelOutput.to_tuple`
    method to convert it to a tuple before.
  """

  def __post_init__(self):
    class_fields = fields(self)

    # Safety and consistency checks
    assert len(class_fields), f"{self.__class__.__name__} has no fields."
    assert all(
      field.default is None for field in class_fields[1:]
    ), f"{self.__class__.__name__} should not have more than one required field."

    first_field = getattr(self, class_fields[0].name)
    other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

    if other_fields_are_none and not is_tensor(first_field):
      try:
        iterator = iter(first_field)
        first_field_iterator = True
      except TypeError:
        first_field_iterator = False

      # if we provided an iterator as first field and the iterator is a (key, value) iterator
      # set the associated fields
      if first_field_iterator:
        for element in iterator:
          if (
            not isinstance(element, (list, tuple))
            or not len(element) == 2
            or not isinstance(element[0], str)
          ):
            break
          setattr(self, element[0], element[1])
          if element[1] is not None:
            self[element[0]] = element[1]
    else:
      for field in class_fields:
        v = getattr(self, field.name)
        if v is not None:
          self[field.name] = v

  def __delitem__(self, *args, **kwargs):
    raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

  def setdefault(self, *args, **kwargs):
    raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

  def pop(self, *args, **kwargs):
    raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

  def update(self, *args, **kwargs):
    raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

  def __getitem__(self, k):
    if isinstance(k, str):
      inner_dict = {k: v for (k, v) in self.items()}
      return inner_dict[k]
    else:
      return self.to_tuple()[k]

  def to_tuple(self) -> typing.Tuple[typing.Any]:
    """
    Convert self to a tuple containing all the attributes/keys that are not ``None``.
    """
    return tuple(self[k] for k in self.keys())

@dataclass
class BaseModelOutput(ModelOutput):
  """
  Base class for model's outputs, with potential hidden states and attentions.

  Args:
    last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
      Sequence of hidden-states at the output of the last layer of the model.
    hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
      Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
      of shape :obj:`(batch_size, sequence_length, hidden_size)`.

      Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
      Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
      :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

      Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
      heads.
  """

  last_hidden_state: torch.FloatTensor
  hidden_states: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
  attentions: typing.Optional[typing.Tuple[torch.FloatTensor]] = None


@dataclass
class BaseModelOutputWithPooling(ModelOutput):
  """
  Base class for model's outputs that also contains a pooling of the last hidden states.

  Args:
    last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
      Sequence of hidden-states at the output of the last layer of the model.
    pooler_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
      Last layer hidden-state of the first token of the sequence (classification token)
      further processed by a Linear layer and a Tanh activation function. The Linear
      layer weights are trained from the next sentence prediction (classification)
      objective during pretraining.
    hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
      Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
      of shape :obj:`(batch_size, sequence_length, hidden_size)`.

      Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
      Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
      :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

      Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
      heads.
  """

  last_hidden_state: torch.FloatTensor
  pooler_output: torch.FloatTensor = None
  hidden_states: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
  attentions: typing.Optional[typing.Tuple[torch.FloatTensor]] = None


@dataclass
class BaseModelOutputWithPast(ModelOutput):
  """
  Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

  Args:
    last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
      Sequence of hidden-states at the output of the last layer of the model.

      If `past_key_values` is used only the last hidden-state of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
    past_key_values (:obj:`typing.List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
      typing.List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
      :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

      Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
      ``past_key_values`` input) to speed up sequential decoding.
    hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
      Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
      of shape :obj:`(batch_size, sequence_length, hidden_size)`.

      Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
      Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
      :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

      Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
      heads.
  """

  last_hidden_state: torch.FloatTensor
  past_key_values: typing.Optional[typing.List[torch.FloatTensor]] = None
  hidden_states: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
  attentions: typing.Optional[typing.Tuple[torch.FloatTensor]] = None


@dataclass
class Seq2SeqModelOutput(ModelOutput):
  """
  Base class for model encoder's outputs that also contains : pre-computed hidden states that can speed up sequential
  decoding.

  Args:
    last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
      Sequence of hidden-states at the output of the last layer of the decoder of the model.

      If ``decoder_past_key_values`` is used only the last hidden-state of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
    decoder_past_key_values (:obj:`typing.List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
      typing.List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
      :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

      Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
      used (see ``decoder_past_key_values`` input) to speed up sequential decoding.
    decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
      Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
      of shape :obj:`(batch_size, sequence_length, hidden_size)`.

      Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
    decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
      Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
      :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

      Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
      self-attention heads.
    encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
      Sequence of hidden-states at the output of the last layer of the encoder of the model.
    encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
      Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
      of shape :obj:`(batch_size, sequence_length, hidden_size)`.

      Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
    encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
      Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
      :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

      Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
      self-attention heads.
  """

  last_hidden_state: torch.FloatTensor
  decoder_past_key_values: typing.Optional[typing.List[torch.FloatTensor]] = None
  decoder_hidden_states: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
  decoder_attentions: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
  encoder_last_hidden_state: typing.Optional[torch.FloatTensor] = None
  encoder_hidden_states: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
  encoder_attentions: typing.Optional[typing.Tuple[torch.FloatTensor]] = None


@dataclass
class CausalLMOutput(ModelOutput):
  """
  Base class for causal language model (or autoregressive) outputs.

  Args:
    loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
      Language modeling loss (for next-token prediction).
    logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
      Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
      Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
      of shape :obj:`(batch_size, sequence_length, hidden_size)`.

      Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
      Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
      :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

      Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
      heads.
  """

  loss: typing.Optional[torch.FloatTensor]
  logits: torch.FloatTensor = None
  hidden_states: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
  attentions: typing.Optional[typing.Tuple[torch.FloatTensor]] = None


@dataclass
class CausalLMOutputWithPast(ModelOutput):
  """
  Base class for causal language model (or autoregressive) outputs.

  Args:
    loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
      Language modeling loss (for next-token prediction).
    logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
      Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (:obj:`typing.List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
      typing.List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
      :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

      Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
      ``past_key_values`` input) to speed up sequential decoding.
    hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
      Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
      of shape :obj:`(batch_size, sequence_length, hidden_size)`.

      Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
      Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
      :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

      Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
      heads.
  """

  loss: typing.Optional[torch.FloatTensor] = None
  logits: torch.FloatTensor = None
  past_key_values: typing.Optional[typing.List[torch.FloatTensor]] = None
  hidden_states: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
  attentions: typing.Optional[typing.Tuple[torch.FloatTensor]] = None


@dataclass
class MaskedLMOutput(ModelOutput):
  """
  Base class for masked language models outputs.

  Args:
    loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
      Masked languaged modeling (MLM) loss.
    logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
      Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
      Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
      of shape :obj:`(batch_size, sequence_length, hidden_size)`.

      Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
      Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
      :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

      Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
      heads.
  """

  loss: typing.Optional[torch.FloatTensor] = None
  logits: torch.FloatTensor = None
  hidden_states: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
  attentions: typing.Optional[typing.Tuple[torch.FloatTensor]] = None


@dataclass
class Seq2SeqLMOutput(ModelOutput):
  """
  Base class for sequence-to-sequence language models outputs.

  Args:
    loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
      Languaged modeling loss.
    logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
      Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    decoder_past_key_values (:obj:`typing.List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
      typing.List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
      :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

      Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
      used (see ``decoder_past_key_values`` input) to speed up sequential decoding.
    decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
      Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
      of shape :obj:`(batch_size, sequence_length, hidden_size)`.

      Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
    decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
      Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
      :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

      Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
      self-attention heads.
    encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
      Sequence of hidden-states at the output of the last layer of the encoder of the model.
    encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
      Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
      of shape :obj:`(batch_size, sequence_length, hidden_size)`.

      Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
    encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
      Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
      :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

      Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
      self-attention heads.
  """

  loss: typing.Optional[torch.FloatTensor] = None
  logits: torch.FloatTensor = None
  decoder_past_key_values: typing.Optional[typing.List[torch.FloatTensor]] = None
  decoder_hidden_states: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
  decoder_attentions: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
  encoder_last_hidden_state: typing.Optional[torch.FloatTensor] = None
  encoder_hidden_states: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
  encoder_attentions: typing.Optional[typing.Tuple[torch.FloatTensor]] = None


@dataclass
class NextSentencePredictorOutput(ModelOutput):
  """
  Base class for outputs of models predicting if two sentences are consecutive or not.

  Args:
    loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`next_sentence_label` is provided):
      Next sequence prediction (classification) loss.
    logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
      Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
    hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
      Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
      of shape :obj:`(batch_size, sequence_length, hidden_size)`.

      Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
      Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
      :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

      Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
      heads.
  """

  loss: typing.Optional[torch.FloatTensor] = None
  logits: torch.FloatTensor = None
  hidden_states: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
  attentions: typing.Optional[typing.Tuple[torch.FloatTensor]] = None


@dataclass
class SequenceClassifierOutput(ModelOutput):
  """
  Base class for outputs of sentence classification models.

  Args:
    loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
      Classification (or regression if config.num_labels==1) loss.
    logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
      Classification (or regression if config.num_labels==1) scores (before SoftMax).
    hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
      Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
      of shape :obj:`(batch_size, sequence_length, hidden_size)`.

      Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
      Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
      :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

      Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
      heads.
  """

  loss: typing.Optional[torch.FloatTensor] = None
  logits: torch.FloatTensor = None
  hidden_states: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
  attentions: typing.Optional[typing.Tuple[torch.FloatTensor]] = None


@dataclass
class Seq2SeqSequenceClassifierOutput(ModelOutput):
  """
  Base class for outputs of sequence-to-sequence sentence classification models.

  Args:
    loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
      Classification (or regression if config.num_labels==1) loss.
    logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
      Classification (or regression if config.num_labels==1) scores (before SoftMax).
    decoder_past_key_values (:obj:`typing.List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
      typing.List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
      :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

      Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
      used (see ``decoder_past_key_values`` input) to speed up sequential decoding.
    decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
      Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
      of shape :obj:`(batch_size, sequence_length, hidden_size)`.

      Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
    decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
      Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
      :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

      Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
      self-attention heads.
    encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
      Sequence of hidden-states at the output of the last layer of the encoder of the model.
    encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
      Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
      of shape :obj:`(batch_size, sequence_length, hidden_size)`.

      Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
    encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
      Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
      :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

      Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
      self-attention heads.
  """

  loss: typing.Optional[torch.FloatTensor] = None
  logits: torch.FloatTensor = None
  decoder_past_key_values: typing.Optional[typing.List[torch.FloatTensor]] = None
  decoder_hidden_states: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
  decoder_attentions: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
  encoder_last_hidden_state: typing.Optional[torch.FloatTensor] = None
  encoder_hidden_states: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
  encoder_attentions: typing.Optional[typing.Tuple[torch.FloatTensor]] = None


@dataclass
class MultipleChoiceModelOutput(ModelOutput):
  """
  Base class for outputs of multiple choice models.

  Args:
    loss (:obj:`torch.FloatTensor` of shape `(1,)`, `optional`, returned when :obj:`labels` is provided):
      Classification loss.
    logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
      `num_choices` is the second dimension of the input tensors. (see `input_ids` above).

      Classification scores (before SoftMax).
    hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
      Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
      of shape :obj:`(batch_size, sequence_length, hidden_size)`.

      Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
      Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
      :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

      Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
      heads.
  """

  loss: typing.Optional[torch.FloatTensor] = None
  logits: torch.FloatTensor = None
  hidden_states: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
  attentions: typing.Optional[typing.Tuple[torch.FloatTensor]] = None


@dataclass
class TokenClassifierOutput(ModelOutput):
  """
  Base class for outputs of token classification models.

  Args:
    loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
      Classification loss.
    logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`):
      Classification scores (before SoftMax).
    hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
      Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
      of shape :obj:`(batch_size, sequence_length, hidden_size)`.

      Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
      Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
      :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

      Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
      heads.
  """

  loss: typing.Optional[torch.FloatTensor] = None
  logits: torch.FloatTensor = None
  hidden_states: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
  attentions: typing.Optional[typing.Tuple[torch.FloatTensor]] = None


@dataclass
class QuestionAnsweringModelOutput(ModelOutput):
  """
  Base class for outputs of question answering models.

  Args:
    loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
      Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
    start_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
      Span-start scores (before SoftMax).
    end_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
      Span-end scores (before SoftMax).
    hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
      Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
      of shape :obj:`(batch_size, sequence_length, hidden_size)`.

      Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
      Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
      :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

      Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
      heads.
  """

  loss: typing.Optional[torch.FloatTensor] = None
  start_logits: torch.FloatTensor = None
  end_logits: torch.FloatTensor = None
  hidden_states: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
  attentions: typing.Optional[typing.Tuple[torch.FloatTensor]] = None


@dataclass
class Seq2SeqQuestionAnsweringModelOutput(ModelOutput):
  """
  Base class for outputs of sequence-to-sequence question answering models.

  Args:
    loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
      Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
    start_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
      Span-start scores (before SoftMax).
    end_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
      Span-end scores (before SoftMax).
    decoder_past_key_values (:obj:`typing.List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
      typing.List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
      :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

      Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be
      used (see ``decoder_past_key_values`` input) to speed up sequential decoding.
    decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
      Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
      of shape :obj:`(batch_size, sequence_length, hidden_size)`.

      Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
    decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
      Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
      :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

      Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
      self-attention heads.
    encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
      Sequence of hidden-states at the output of the last layer of the encoder of the model.
    encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
      Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
      of shape :obj:`(batch_size, sequence_length, hidden_size)`.

      Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
    encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
      Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
      :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

      Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
      self-attention heads.
  """

  loss: typing.Optional[torch.FloatTensor] = None
  start_logits: torch.FloatTensor = None
  end_logits: torch.FloatTensor = None
  decoder_past_key_values: typing.Optional[typing.List[torch.FloatTensor]] = None
  decoder_hidden_states: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
  decoder_attentions: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
  encoder_last_hidden_state: typing.Optional[torch.FloatTensor] = None
  encoder_hidden_states: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
  encoder_attentions: typing.Optional[typing.Tuple[torch.FloatTensor]] = None
