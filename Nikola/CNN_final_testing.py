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
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import keras_tuner as kt

from sklearn.metrics import mean_absolute_error

train_labels = []
train_samples = []

# read in the file
training_dataset = pd.read_csv(
    r'./waters_datasets/Cleaned_Datasets/Merged_Dataset_For_Trainging.csv',
    usecols=[
        "DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col","WQI"
    ]
)[["DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col","WQI"]]

x_train = training_dataset[["DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col"]]
y_train = training_dataset["WQI"]





# Testing dataset 1
testing_dataset_1 = pd.read_csv(r'./waters_datasets/Cleaned_Datasets/testing_data_1.csv',
    usecols=[
        "DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col","WQI",
    ]
)[["DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col","WQI"]]

# testing cloumns for x 
x_test_1 = testing_dataset_1[["DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col"]]

# testing cloumns for y
y_test_1 = testing_dataset_1["WQI"]






#Use the hp argument to define the hyperparameters during model creation.
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Dense(
        hp.Choice('units', [8, 16, 32, 64, 128, 256, 512]),
        activation='relu'))
    model.add(keras.layers.Dense(
      hp.Choice('units', [8, 16, 32, 64, 128, 256, 512]),
      activation='relu'))
    model.add(keras.layers.Dense(
      hp.Choice('units', [8, 16, 32, 64, 128, 256, 512]),
      activation='relu'))
    model.add(keras.layers.Dense(
      hp.Choice('units', [8, 16, 32, 64, 128, 256,  512]),
      activation='relu'))
    model.add(keras.layers.Dense(
      hp.Choice('units', [8, 16, 32, 64, 128, 256, 512]),
      activation='relu'))
   
    model.add(keras.layers.Dense(1, activation='linear'))
    model.compile(loss='mse')
    return model

physical_devices = tf.config.experimental.list_physical_devices('GPU')

#add hyperparameter tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=20)

tuner.search(x_train, y_train, epochs=500, validation_data=(x_test_1, y_test_1))
model = tuner.get_best_models()[0]

prediction = model.predict(x_test_1)

# Save the predictinos into a csv file so you can add them to the orginal file.
np.savetxt("predictions_CNN_test_1.csv", prediction)

#Root mean square error
rmse = np.sqrt(mean_squared_error(y_test_1, prediction, squared=False))
print("RMSE: ", rmse)




