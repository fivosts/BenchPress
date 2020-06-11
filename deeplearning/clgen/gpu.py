from deeplearning.clgen.tf import tf

def numGPUs():
  devices = tf.python.client.device_lib.list_local_devices()
  return len([x for x in devices if x.device_type == 'GPU'])