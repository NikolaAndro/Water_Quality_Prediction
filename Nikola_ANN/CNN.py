##
## Programmer: Nikola Andric
## Email: namdd@mst.edu
## Last Eddited: 10/29/2021
##
##

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

import pandas as pd
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

train_labels = []
train_samples = []



my_dataset = pd.read_csv(
    r'./preprocessed_dataset.csv',
    usecols=[
        "DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col","WQI"
    ]
)[["DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col","WQI"]]

x = my_dataset[["DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col"]]
y = my_dataset["WQI"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 1)

print(my_dataset.head())

# Running the code on the GPU. Making sure tensorflow is correctly identifying the GPU as well as enable memory growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Number of GPUs available: ",len(physical_devices))
#tf.config.experimental.set_memory_growth(physical_devices[0],True)

#np.array(my_dataset)

#print(my_dataset.head())
print(my_dataset.shape)

#Building a sequential model
model = Sequential([
    Dense(units = 16, input_dim = 7, activation = 'relu'),
    Dense(units = 32, activation = 'relu'),
    Dense(units = 24, activation = 'relu'),
    Dense(units = 32, activation = 'relu'),
    Dense(units = 16, activation = 'relu'),
    Dense(units = 32, activation = 'relu'),
    Dense(units = 24, activation = 'relu'),
    Dense(units = 14, activation = 'relu'),
    Dense(units = 32, activation = 'relu'),
    Dense(units = 23, activation = 'relu'),
    Dense(units = 32, activation = 'relu'),
    Dense(units = 16, activation = 'relu'),
    Dense(units = 10, activation = 'relu'),
    Dense(units = 20, activation = 'relu'),
    Dense(units = 16, activation = 'relu'),
    Dense(units = 1, activation='linear')
])

#model.summary()

#prepare the model for training
model.compile(optimizer="adam", loss='mse', metrics=['accuracy'])

#fitting the data
model.fit(x_train, y_train, epochs = 300, batch_size = 80)

#print(model.predict(x_test))
prediction = model.predict(x_test)
from sklearn.metrics import mean_absolute_error
#this has to be close to 0
print(model.predict(x_test))
print(mean_absolute_error(y_test,prediction))


