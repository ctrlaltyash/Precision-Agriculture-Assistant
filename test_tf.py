import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
print("TF Imported")
try:
    print(tf.reduce_sum(tf.random.normal([1000, 1000])))
    print("TF Computation Success")
except Exception as e:
    print(f"TF Computation Failed: {e}")
