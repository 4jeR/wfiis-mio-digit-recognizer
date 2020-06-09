import tensorflow as tf
import matplotlib.pyplot as plt 
import pandas as pd 
import csv
import numpy as np


''' Training data'''
train_data = pd.read_csv("digit-recognizer/train.csv").to_numpy()
x_train = train_data[0:, 1:]
x_train_digit = train_data[0:, 0]



''' Test data '''
test_data = pd.read_csv("digit-recognizer/test.csv").to_numpy()
x_test = test_data[0:, 0:]


''' Predicting '''
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

''' 128 - liczba neuronow w poszczegolnej warstwie sieci'''
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))

''' 10 - liczba klasyfikacji - w naszym przypadku cyfry [0-9] '''
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, x_train_digit, epochs=20)



p = model.predict(x_test)

f = open("digit-recognizer/our_submission_network.csv", mode='w')
f.write("ImageId,Label\n")
for i in range(len(p)):
    f.write(f"{i+1},{np.argmax(p[i])}\n")

f.close()

