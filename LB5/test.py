import os
print("LD_LIBRARY_PATH =", os.environ.get('LD_LIBRARY_PATH'))
import tensorflow as tf
print("Найденные GPU:", tf.config.list_physical_devices('GPU'))
