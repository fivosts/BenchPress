
tf = None
if tf is None:
  import tensorflow
  print("AHYAHA\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
  tf = tensorflow
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  tf.compat.v1.disable_eager_execution()
else:
  print("AOUA\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")