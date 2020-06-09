import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
import csv


dtc = DecisionTreeClassifier()

''' Training data'''
train_data = pd.read_csv("digit-recognizer/train.csv").to_numpy()
x_train = train_data[0:, 1:]
x_train_digit = train_data[0:, 0]

dtc.fit(x_train, x_train_digit)


''' Test data '''
test_data = pd.read_csv("digit-recognizer/test.csv").to_numpy()
x_test = test_data[0:, 0:]


''' Predicting '''
p = dtc.predict(x_test)
plt.imshow(255-d, cmap='gray')
plt.show()


f = open("digit-recognizer/our_submission.csv", mode='w')
f.write("ImageId,Label\n")
for i in range(len(p)):
    f.write(f"{i+1},{p[i]}\n")

f.close()

