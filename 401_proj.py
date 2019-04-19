import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dense, MaxPooling2D, Flatten, Dropout
from sklearn.metrics import log_loss
from keras.optimizers import SGD
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from load_data import load_data, load_resized_data
import numpy as np
import os
from os.path import join
import time
import cv2


X_train, X_valid, Y_train, Y_valid = load_resized_data(100,100)

print(X_train.shape)
num_classes = 2
channel = 3 #RGB
batch_size = 128
num_epoch = 15

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(100,100,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

t=time.time()
hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epoch, verbose=1,
                     shuffle=False, validation_data= (X_valid,Y_valid))
#compute the training time
print('Training time: %s' % (time.time()-t))

model.save_weights('dog_cat.h5')

y_pred = model.predict(X_valid, batch_size=batch_size, verbose=1)

y_pred = np.argmax(y_pred, axis=1)
y_actual = np.argmax(Y_valid, axis=1)

correct = y_actual[y_actual == y_pred]
incorrect = y_actual[y_actual != y_pred]

print("Test Accuracy = ", len(correct)/len(y_actual), "%")
print("Test Inaccuracy = ", len(incorrect)/len(y_actual),"%")


