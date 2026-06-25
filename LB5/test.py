import os
import tensorflow as tf

print("LD_LIBRARY_PATH =", os.environ.get("LD_LIBRARY_PATH"))
print("Available GPUs:", tf.config.list_physical_devices("GPU"))
