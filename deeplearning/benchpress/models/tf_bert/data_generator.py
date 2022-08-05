# coding=utf-8
# Copyright 2022 Foivos Tsimpourlas.
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
"""This file defines the streaming generators for model training data.

We train models on overlapping one-hot encoded text sequences. For a corpus of
a reasonable size, the full training data may not fit in memory. This modules
provides Python Generator classes for use by a sequential Keras model's
fit_generator() method to stream batches of training data.
"""
import typing
import random
import collections
import glob
import numpy as np

from deeplearning.benchpress.util.tf import tf
from deeplearning.benchpress.proto import model_pb2
from deeplearning.benchpress.models import lm_data_generator
from deeplearning.benchpress.models import sequence_masking
from absl import flags
from deeplearning.benchpress.util import logging as l

FLAGS = flags.FLAGS

class tfLMDataGenerator(lm_data_generator.MaskLMDataGenerator):

  @classmethod
  def TrainMaskLMBatchGenerator(cls,
                               corpus: "corpuses.Corpus",
                               training_opts: model_pb2.TrainingOptions,
                               cache_path,
                               ) -> "data_generator.MaskLMBatchGenerator":
    """Initializes data generator for training."""
    return super(tfLMDataGenerator, tfLMDataGenerator()).TrainMaskLMBatchGenerator(
              corpus, training_opts, cache_path
            )

  @classmethod
  def SampleMaskLMBatchGenerator(cls,
                                model_opts,
                                sampler,
                                tokenizer,
                                seed: int,
                                max_position_embeddings: int,
                                cache_path,
                                ) -> "data_generator.MaskLMBatchGenerator":
    """Initializes data generator for inference."""
    d = super(tfLMDataGenerator, tfLMDataGenerator()).SampleMaskLMBatchGenerator(
          model_opts, sampler, tokenizer, seed, max_position_embeddings, cache_path
        )
    d.tfRecordSampler = d.tfRecordSampleGenerator()
    return d

  def __init__(self):
    super(tfLMDataGenerator, self).__init__("tf_record")
    self.sampleBatch             = None
    self.sampleIndices           = None
    self.tfRecordSampler         = None
    return

  def generateTfDataset(self,
                      sequence_length: int,
                      is_training    : bool,
                      num_cpu_threads: int,
                      eval_set       : typing.List = None,
                      use_tpu        : bool = False,
                      ) -> "tf.Dataset":
    """Wrapper function that constructs a tf.Dataset used for training BERT."""

    def input_fn(params):
      """
      function used by tf.estimator to generate inputs for training.
      """
      def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        ## This function assumes record is still a file (expressed as TF dataset)
        ## It decodes this record to tf scalars.
        example = tf.io.parse_single_example(record, name_to_features)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
          t = example[name]
          if t.dtype == tf.int64:
            t = tf.cast(t, dtype = tf.int32)
          example[name] = t
        return example

      batch_size = params["batch_size"]
      name_to_features = {
          "seen_in_training"        : tf.io.FixedLenFeature([1], tf.int64),
          "original_input"          : tf.io.FixedLenFeature([sequence_length], tf.int64),
          "input_ids"               : tf.io.FixedLenFeature([sequence_length], tf.int64),
          "input_mask"              : tf.io.FixedLenFeature([sequence_length], tf.int64),
          "masked_lm_positions"     : tf.io.FixedLenFeature([self.training_opts.max_predictions_per_seq], tf.int64),
          "masked_lm_ids"           : tf.io.FixedLenFeature([self.training_opts.max_predictions_per_seq], tf.int64),
          "masked_lm_weights"       : tf.io.FixedLenFeature([self.training_opts.max_predictions_per_seq], tf.float32),
          "masked_lm_lengths"       : tf.io.FixedLenFeature([self.training_opts.max_predictions_per_seq], tf.int64),
          "next_sentence_labels"    : tf.io.FixedLenFeature([1], tf.int64),
      }

      # For training, we want a lot of parallel reading and shuffling.
      # For eval, we want no shuffling and parallel reading doesn't matter.
      if is_training:
        dataset = tf.io.gfile.glob([str(p) for p in self.dataset['train_dataset']['file']])
        d = tf.data.Dataset.from_tensor_slices(tf.constant(dataset))
        if self.training_opts.shuffle_corpus_contentfiles_between_epochs:
          d = d.shuffle(buffer_size = len(dataset), reshuffle_each_iteration=True)
        d = d.repeat()

        # `cycle_length` is the number of parallel files that get read.
        cycle_length = min(num_cpu_threads, len(dataset))

        # `sloppy` mode means that the interleaving is not exact. This adds
        # even more randomness to the training pipeline.
        d = d.apply(
            tf.data.experimental.parallel_interleave(
                tf.data.TFRecordDataset,
                sloppy=is_training,
                cycle_length=cycle_length))
      else:
        if eval_set is None:
          dataset = tf.io.gfile.glob(
            [str(path) for tf_set in self.dataset for path in self.dataset[tf_set]['file']]
          )
        else:
          dataset = tf.io.gfile.glob([str(tf_set) for tf_set in eval_set])
        d = tf.data.TFRecordDataset(dataset)
        # Since we evaluate for a fixed number of steps we don't want to encounter
        # out-of-range exceptions.
        d = d.repeat()

      # We must `drop_remainder` on training because the TPU requires fixed
      # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
      # and we *don't* want to drop the remainder, otherwise we wont cover
      # every sample.
      d = d.apply(
          tf.data.experimental.map_and_batch(
              lambda record: _decode_record(record, name_to_features),
              batch_size=batch_size,
              num_parallel_batches=num_cpu_threads,
              drop_remainder=use_tpu))
      return d
    return input_fn

  def generateTfSamples(self):
    """
    Contains input_fn closure function for estimator
    
    Returns:
      input_fn callable.
    """

    def input_fn(params):
      """
      function used by tf.estimator to generate inputs for inference.
      """

      def sample_gen(batch_size: int):
        """
        This generator yields iteratively the inference input blob for each step.
        In the first iteration, it yields sampler.encoded_start_text and then for each step,
        self.sampleBatch is updated with the model's current output through self.updateVocabulary.
        The generator stops when the model has filled all mask or hole tokens with predictions.

        Arguments:
          batch_size: The batch size used during inference.

        Yields:
          Current step's inference input for model.

        Returns:
          None
        """
        assert batch_size == len(self.sampleBatch), "{}, {}".format(batch_size, len(self.sampleBatch))
        original_input = [sample for sample in self.sampleBatch]
        while True:

          (input_ids, input_mask, masked_lm_positions,
          masked_lm_ids, masked_lm_weights, masked_lm_lengths) = [], [], [], [], [], []

          max_mask_len = max(
          [len(np.where(np.in1d(np.asarray(x), [self.tokenizer.maskToken, self.tokenizer.holeToken]))[0]) for x in self.sampleBatch]
          )
          if max_mask_len == 0:
            return
          for sample in self.sampleBatch:
            sample_masks = np.where(np.in1d(sample, [self.tokenizer.maskToken, self.tokenizer.holeToken]))[0]
            actual_mask_len = len(sample_masks)
            len_offset     = max_mask_len - actual_mask_len
            pad_idx      = np.where(sample == self.tokenizer.padToken)[0]
            inp_mask     = np.ones(len(sample), dtype = np.int32)
            if len(pad_idx) > 0:
              inp_mask[pad_idx[0]:] = 0

            input_ids.append(list(sample))
            input_mask.append(list(inp_mask))
            masked_lm_positions.append(list(sample_masks) + [0] * len_offset)
            masked_lm_ids.append([self.tokenizer.maskToken] * actual_mask_len + [self.tokenizer.padToken] * len_offset)
            masked_lm_weights.append([0.0] * (actual_mask_len + len_offset))
            masked_lm_lengths.append([-1] * (actual_mask_len + len_offset))
          yield (np.full([batch_size, 1], -1), original_input, 
            input_ids, input_mask,
            masked_lm_positions, masked_lm_ids,
            masked_lm_weights, masked_lm_lengths,
            np.zeros([batch_size, 1]))

      batch_size = params['batch_size']
      sample = tf.data.Dataset.from_generator(
                lambda: sample_gen(batch_size), 
                output_types = sequence_masking.tfSequence.tfTypes(),
                output_shapes = sequence_masking.tfSequence.tfShapes(batch_size, self.sampler.sequence_length)
                )

      it = tf.compat.v1.data.make_one_shot_iterator(sample)
      (seen_in_training, original_input, 
        input_ids, input_mask,
        masked_lm_positions, masked_lm_ids,
        masked_lm_weights, masked_lm_lengths, next_sentence_labels) = it.get_next()

      return {
          'seen_in_training'      : seen_in_training,
          'original_input'        : original_input,
          'input_ids'             : input_ids,
          'input_mask'            : input_mask,
          'masked_lm_positions'   : masked_lm_positions,
          'masked_lm_ids'         : masked_lm_ids,
          'masked_lm_weights'     : masked_lm_weights,
          'masked_lm_lengths'     : masked_lm_lengths,
          'next_sentence_labels'  : next_sentence_labels,
      }
    return input_fn

  def tfRecordSampleGenerator(self):

    if self.sampler.isFixedStr:
      return None
    assert not self.sampler.config.HasField("start_text")

    path_list = self.configSampleSets()
    if len(path_list) == 0:
      raise FileNotFoundError(path_list)

    for path in path_list:
      for example in tf.compat.v1.io.tf_record_iterator(path):
        input_ids = np.asarray(tf.train.Example.FromString(example).features.feature['input_ids'].int64_list.value)
        if self.tokenizer.padToken in input_ids:
          yield input_ids[:np.where(input_ids == self.tokenizer.padToken)[0][0]]
        else:
          yield input_ids

  def InitSampleBatch(self) -> None:
    """
      Initializes data_generator for inference.
      self.sampleBatch is initialized with sampler.encoded_start_text
    """
    if not self.sampler.isFixedStr:
      try:
        start_text = next(self.tfRecordSampler)[:self.sampler.sequence_length]
      except StopIteration:
        l.logger().warn("Repeating iterator on dataset...")
        self.tfRecordSampler = self.tfRecordSampleGenerator()
        try:
          start_text = next(self.tfRecordSampler)[:self.sampler.sequence_length]
        except Exception as e:
          raise e
      self.sampler.setStartText(self.tokenizer.tokensToString(start_text))
      self.sampler.Specialize(self.tokenizer)
    
    assert self.sampler.sequence_length <= self.max_position_embeddings, "Sampler sequence length exceeds max position embeddings."
    input_sample = self.sampler.encoded_start_text
    assert np.ndim(input_sample) == 1, "Input samples have to be one-dimensional. {} given.".format(input_sample.shape)

    target_idx = np.where(np.in1d(input_sample, [self.tokenizer.maskToken, self.tokenizer.holeToken]))[0]
    assert len(target_idx) != 0, "No target prediction in sample text"

    num_masks = np.count_nonzero(input_sample == self.tokenizer.maskToken)
    num_holes = np.count_nonzero(input_sample == self.tokenizer.holeToken)
    num_targets = num_masks + num_holes

    padded_sample = self._padToMaxPosition(input_sample)
    padded_sample = padded_sample[:self.sampler.sequence_length]
    self.sampleBatch   = np.repeat(padded_sample[None, :], self.sampler.batch_size, axis = 0)
    self.sampleIndices = [[[] for i in range(num_targets)] for j in range(self.sampler.batch_size)]
    return

  def updateSampleBatch(self, 
                        input_ids     : np.array,
                        masked_lm_ids : np.array,
                        ) -> np.array:
    """
    Updates self.sampleBatch with the model's output prediction.
    The output, if still contains hole or mask tokens, is fed back
    to the model's input through the input_fn's sample_gen generator.
    """
    assert len(input_ids) == len(masked_lm_ids), "Inputs and predictions do not have the same batch size."

    updated_sequence = []
    done = True
    for batch_idx, _ in enumerate(input_ids):
      batch = []
      mask_id_index     = 0
      closed_hole_index = 0
      for idx, token in enumerate(input_ids[batch_idx]):
        if   token == self.tokenizer.maskToken:
          mt = masked_lm_ids[batch_idx][mask_id_index]
          if mt == self.tokenizer.maskToken or mt == self.tokenizer.holeToken:
            continue
          if len(self.sampleIndices[batch_idx][mask_id_index]) > 0:
            while(self.sampleIndices[batch_idx][mask_id_index + closed_hole_index][-1]) == self.tokenizer.endholeToken:
              closed_hole_index += 1
          self.sampleIndices[batch_idx][mask_id_index + closed_hole_index].append(mt)
          mask_id_index += 1
          batch.append(mt)
        elif token == self.tokenizer.holeToken:
          mt = masked_lm_ids[batch_idx][mask_id_index]
          if mt == self.tokenizer.maskToken or mt == self.tokenizer.holeToken:
            continue
          if len(self.sampleIndices[batch_idx][mask_id_index]) > 0:
            while(self.sampleIndices[batch_idx][mask_id_index + closed_hole_index][-1]) == self.tokenizer.endholeToken:
              closed_hole_index += 1
          self.sampleIndices[batch_idx][mask_id_index + closed_hole_index].append(mt)
          mask_id_index += 1
          if mt != self.tokenizer.endholeToken:
            batch.append(mt)
            batch.append(self.tokenizer.holeToken)
            done = False
        else:
          batch.append(token)
      batch = np.asarray(batch)
      batch = self._padToMaxPosition(batch)
      # TODO, chop sequence for now, but TODO it: 
      # If a sequence is bigger than it should, crop one or both edges,
      # save them and send max_position_embeddings for next step.
      # Then, concat it back.
      if self.sampler.sequence_length > len(batch):
        l.logger().warn("Cropped {} tokens from sample batch".format(self.sampler.sequence_length - len(batch)))
      batch = batch[:self.sampler.sequence_length]
      updated_sequence.append(batch)

    self.sampleBatch = np.asarray(updated_sequence)
    return self.sampleBatch, self.sampleIndices

  def toTensorFormat(self,
                     datapoint: typing.TypeVar("#TODO")
                     ) -> typing.TypeVar("#TODO"):
    raise NotImplementedError("#TODO!")

  def _saveCorpusRecord(self, masked_corpus: typing.Dict) -> None:
    """Converts corpus nparrays to tf Features and stores corpus to TfRecord"""
     
    writer = tf.io.TFRecordWriter(str(masked_corpus['file']))
    if FLAGS.write_text_dataset:
      file_writer = open(masked_corpus['txt'], 'w')

    for (inst_index, instance) in enumerate(masked_corpus['corpus']):
      seen_in_training = instance.seen_in_training
      original_input   = instance.original_input
      input_ids        = instance.input_ids
      input_mask       = instance.input_mask

      assert len(input_ids) == self.training_opts.sequence_length, "len(input_ids):  {}, sequence_length: {}".format(len(input_ids), self.training_opts.sequence_length)

      masked_lm_positions   = instance.masked_lm_positions
      masked_lm_ids         = instance.masked_lm_ids
      masked_lm_weights     = instance.masked_lm_weights
      masked_lm_lengths     = instance.masked_lm_lengths
      next_sentence_label   = instance.next_sentence_label
      features              = collections.OrderedDict()

      features["seen_in_training"]      = tf.train.Feature(int64_list = tf.train.Int64List(
                                                                value = list([seen_in_training])))

      features["original_input"]        = tf.train.Feature(int64_list = tf.train.Int64List(
                                                                value = list(original_input)))

      features["input_ids"]             = tf.train.Feature(int64_list = tf.train.Int64List(
                                                                value = list(input_ids)))

      features["input_mask"]            = tf.train.Feature(int64_list = tf.train.Int64List(
                                                                value = list(input_mask)))

      features["masked_lm_positions"]   = tf.train.Feature(int64_list = tf.train.Int64List(
                                                                value = list(masked_lm_positions)))

      features["masked_lm_ids"]         = tf.train.Feature(int64_list = tf.train.Int64List(
                                                                value = list(masked_lm_ids)))

      features["masked_lm_weights"]     = tf.train.Feature(float_list = tf.train.FloatList(
                                                                value = list(masked_lm_weights)))

      features["masked_lm_lengths"]     = tf.train.Feature(int64_list = tf.train.Int64List(
                                                                value = list(masked_lm_lengths)))

      features["next_sentence_labels"]  = tf.train.Feature(int64_list = tf.train.Int64List(
                                                                value = list([next_sentence_label])))

      tf_example = tf.train.Example(features = tf.train.Features(feature = features))
      writer.write(tf_example.SerializeToString())
      if FLAGS.write_text_dataset:
        file_writer.write("'seen_in_training': {}\n'original_input': {}\n'input_ids': {}\n'input_mask': {}\n'masked_lm_positions': {}\n'masked_lm_ids': {}\n'masked_lm_weights': {}\n'masked_lm_lengths': {}\n'next_sentence_labels': {}\n\n"
                            .format((True if seen_in_training == 1 else False),
                                    self.tokenizer.tokensToString(original_input, ignore_token = self.tokenizer.padToken),
                                    self.tokenizer.tokensToString(input_ids,      ignore_token = self.tokenizer.padToken),
                                    input_mask, 
                                    masked_lm_positions, 
                                    self.tokenizer.tokensToString(masked_lm_ids), 
                                    masked_lm_weights, 
                                    masked_lm_lengths, 
                                    next_sentence_label)
                            )
    writer.close()
    if FLAGS.write_text_dataset:
      file_writer.close()
    l.logger().info("Wrote {} instances ({} batches of {} datapoints) to {}"
                      .format(inst_index + 1, self.steps_per_epoch, self.training_opts.batch_size, masked_corpus['file']))
    return
