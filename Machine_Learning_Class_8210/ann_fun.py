from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf

# Create a network with 1 linear unit
model = keras.Sequential([
    layers.Dense(units=2, input_shape=[1])
])

my_X = np.array([0.05, 0.1])
my_Y = np.array([0.01, 0.99])

x = tf.linspace(-1.0, 1.0, 100)

y = model.predict()

print(y)