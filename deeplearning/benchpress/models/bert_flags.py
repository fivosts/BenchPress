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
