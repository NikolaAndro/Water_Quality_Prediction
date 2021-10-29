##
## Programmer: Nikola Andric
## Email: namdd@mst.edu
## Last Eddited: 10/29/2021
##
##

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

# Running the code on the GPU. Making sure tensorflow is correctly identifying the GPU as well as enable memory growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Number of GPUs available: ",len(physical_devices))
#tf.config.experimental.set_memory_growth(physical_devices[0],True)

#Building a sequential model
model = Sequential([
    Dense(units = 16, input_shape = (1,0), activation = 'relu'),
    Dense(units = 32, activation = 'relu'),
    Dense(units = 2, activation='softmax')
])

model.summary()