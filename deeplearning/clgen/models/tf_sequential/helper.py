
import tensorflow_addons as tfa
import tensorflow_probability as tfp

from deeplearning.clgen.util import logging as l
from deeplearning.clgen.util import tf as local_tf

tf = local_tf.tf

class CustomInferenceHelper(tfa.seq2seq.sampler.TrainingSampler):
  """An inference helper that takes a seed text"""

  def __init__(self, seed_length, embedding, temperature):
    super(CustomInferenceHelper, self).__init__(time_major=False)

    self._seed_length = seed_length
    self._xlate = embedding
    self.softmax_temperature = temperature

  def initialize(self, inputs, sequence_length, name=None):
    return super(CustomInferenceHelper, self).initialize(inputs = inputs,
                                                         sequence_length = sequence_length,
                                                         mask = None
                                                         )

  # def sample(self, time, outputs, state, name=None):
  def sample(self, time, outputs, state):
    if self.softmax_temperature is not None:
      outputs = outputs / self.softmax_temperature

    sampler = tfp.distributions.Categorical(logits=outputs)
    sample_ids = sampler.sample()
    return sample_ids

  ## Only this function requires refactoring
  # def next_inputs(self, time, outputs, state, sample_ids, name = "CIHNextInputs"):
  def next_inputs(self, time, outputs, state, sample_ids):
    # with tf.name_scope(name, "CIHNextInputs", [time, outputs, state]):
    next_time = time + 1
    finished = next_time >= self.sequence_length
    all_finished = tf.reduce_all(finished)
    seed_done = next_time >= self._seed_length

    # def read_from_ta(inp):
    #   return inp.read(next_time)

    next_inputs = tf.case(  ## tf.case maybe deprecated
      [
        ( 
          all_finished,
          lambda: self.zero_inputs
        ),
        ( 
          tf.math.logical_not(seed_done),
          lambda: tf.nest.map_structure(lambda inp: inp.read(next_time), self.input_tas),
        ),
      ],
      default=lambda: tf.stop_gradient(
        tf.nn.embedding_lookup(self._xlate, sample_ids)
      ),
    )
    return (finished, next_inputs, state)
