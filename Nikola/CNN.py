##
## Programmer: Nikola Andric
## Email: namdd@mst.edu
## Last Eddited: 11/06/2021
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
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import keras_tuner as kt

train_labels = []
train_samples = []

# read in the file
training_dataset = pd.read_csv(
    r'./waters_datasets/Cleaned_Datasets/Merged_Dataset_For_Trainging.csv',
    usecols=[
        "DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col","WQI","WQI clf"
    ]
)[["DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col","WQI","WQI clf"]]

x_train = training_dataset[["DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col","WQI clf"]]
y = training_dataset["WQI"]


# Testing dataset 1
testing_dataset_1 = pd.read_csv(r'./waters_datasets/Cleaned_Datasets/testing_data_1.csv',
    usecols=[
        "DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col","WQI","WQI clf"
    ]
)[["DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col","WQI","WQI clf"]]

# Teseting dataset 2 
testing_dataset_2 = pd.read_csv(r'./waters_datasets/Cleaned_Datasets/testing_data_2.csv',
    usecols=[
        "DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col","WQI","WQI clf"
    ]
)[["DO","PH","Conductivity","BOD","NI","Fec_col","Tot_col","WQI","WQI clf"]]


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 1)

# remove the WQI clf from x_train and X_test and store them in new arrays (they hold true values for the ranges 0-3)
x_train_WQI_clf = x_train['WQI clf']
x_test_WQI_clf = x_test['WQI clf']

del x_train['WQI clf']
del x_test['WQI clf']

print(training_dataset.head())

# Running the code on the GPU. Making sure tensorflow is correctly identifying the GPU as well as enable memory growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Number of GPUs available: ",len(physical_devices))
#tf.config.experimental.set_memory_growth(physical_devices[0],True)

#np.array(training_dataset)

#print(training_dataset.head())
print(training_dataset.shape)

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


tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=20)

tuner.search(x_train, y_train, epochs=500, validation_data=(x_test, y_test))
model = tuner.get_best_models()[0]

prediction = model.predict(x_test)




from sklearn.metrics import mean_absolute_error
#this has to be close to 0
print(model.predict(x_test))

#y_test are true labels
#x_test true values
print(mean_absolute_error(y_test,prediction))

#Root mean square error
rmse = mean_squared_error(y_test, prediction, squared=False)
print("RMSE: ", rmse)

prediction_1d = prediction.ravel() #convert ndarray to a regular array
d = {'WQI_predict': prediction_1d, 'True_WQI': y_test}
df_check = pd.DataFrame(data= d)
df_check.to_csv("check_regression_redone.csv")

print(type(prediction_1d))

plt.ylim(0, 1000)
plt.plot(prediction)
plt.plot(y_test.tolist())
plt.savefig('figures/regression.png')
plt.show()


WQI_range = []
list_prediction = prediction_1d.tolist()
for i in range (0, len(list_prediction)):
    if list_prediction[i] < 26:
        WQI_range.append(3)
    elif list_prediction[i] <51:
        WQI_range.append(2)
    elif list_prediction[i] < 76:
        WQI_range.append(1)
    else:
        WQI_range.append(0)

d = {'WQI_predict_range': WQI_range, 'True_WQI_range': x_test_WQI_clf}
df_range_check = pd.DataFrame(data = d)
df_range_check.to_csv("check_range_redone.csv")

#compare ranges column
print("the meaan absolute error btw ranges is: ", mean_absolute_error(WQI_range,x_test_WQI_clf))


plt.plot(WQI_range)
plt.plot(x_test_WQI_clf.to_list())
plt.savefig('figures/ranges.png')
plt.show()

