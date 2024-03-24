import numpy as np
import matplotlib.pyplot as plt

import keras

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout


#Getting the data 
(X_train, y_train),(X_test, y_test) = mnist.load_data()

#Preprocessing the Data
#Normalizing the image [0,1] range
X_train = X_train/255
X_test = X_test/255

#Reshaping and expanding the dimensions of image to (28,28,1)
X_train = np.expand_dims(X_train,-1)
X_test = np.expand_dims(X_test,-1)

#converting classes to vectors
y_train = keras.utils.np_utils.to_categorical(y_train)
y_test = keras.utils.np_utils.to_categorical(y_test)


#Model 
model = Sequential()

model.add(Conv2D(32,(3,3), activation = 'relu',input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dropout(0.25))

model.add(Dense(10, activation= "softmax"))

model.compile(optimizer= 'adam', loss =keras.losses.categorical_crossentropy, metrics = 'accuracy')

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'val_accuracy', min_delta = 0.01, patience =4, verbose =1)

mc = ModelCheckpoint("./name_of_model.h5", monitor = "val_accuracy", verbose =1, save_best_only = True)
cb = [es,mc]

#Model Training
his = model.fit(X_train, y_train, epochs =20, validation_split = 0.3, callbacks = cb)

#Model Loading
model_S = keras.models.load_model("./name_of_model.h5")
score = model_S.evaluate(X_test, y_test)

#Model Accuracy
print(f"Model Accuracy: {score[1]}")
print(f"Model Loss: {score[0]}")