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
"""
Shared absl flags between Pytorch and Tensorflow
BERT models.
"""
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer(
  "monitor_frequency",
  5000,
  "Choose frequency (in steps) in which tensors will be logged during training. "
  "Default: 5000"
)

flags.DEFINE_integer(
  "select_checkpoint_step",
  -1,
  "Select step checkpoint for sample. Re-training with this flag is not supported. "
  "To restart from specific checkpoint, you have to manually remove the checkpoints after the desired one."
  "Default: -1, flag ignored and latest checkpoint is loaded."
)

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_boolean("force_eval", False, "Run Validation no matter what.")

flags.DEFINE_integer("sample_per_epoch", 3, "Set above zero to sample model after every epoch.")

flags.DEFINE_boolean("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_boolean("categorical_sampling", True, "Use categorical distribution on logits when sampling.")
