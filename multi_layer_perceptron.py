import tensorflow as tf
import matplotlib.pyplot as plt 
import pandas as pd 
import csv
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import time
from mlxtend.plotting import plot_confusion_matrix
import seaborn as sn


''' Training data'''
train_data = pd.read_csv("digit-recognizer/train.csv").to_numpy()
x_train = train_data[0:38000, 1:]
y_train_digit = train_data[0:38000, 0]


''' Test data '''
x_test = train_data[38000:, 1:]
y_test_digit = train_data[38000:, 0]



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


start = time.time()
model.fit(x_train, y_train_digit,batch_size=400, epochs=10)
predicted = model.predict(x_test)
end = time.time()


pred = []
for i in range(len(predicted)):
    pred.append(np.argmax(predicted[i]))

''' Error matrices '''
matrix = confusion_matrix(y_test_digit, pred)
fig, ax = plot_confusion_matrix(conf_mat=matrix)


''' Kaggle results '''
test_data = pd.read_csv("digit-recognizer/test.csv").to_numpy()
x_test = test_data[0:, 0:]
x_train = train_data[0:, 1:]
y_train_digit = train_data[0:, 0]


start = time.time()
model.fit(x_train, y_train_digit)
predicted = model.predict(x_test)
end = time.time()


print(f"Multilayer perceptron time = {end - start}")

pred = []
for i in range(len(predicted)):
    pred.append(np.argmax(predicted[i]))


''' Save output to submission file '''
f = open("digit-recognizer/our_submission_mlp.csv", mode='w')
f.write("ImageId,Label\n")
for i in range(len(predicted)):
    f.write(f"{i+1},{pred[i]}\n")

f.close()