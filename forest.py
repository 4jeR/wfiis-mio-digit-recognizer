import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import csv
from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier()

''' Training data'''
train_data = pd.read_csv("digit-recognizer/train.csv").to_numpy()
x_train = train_data[0:, 1:]
x_train_digit = train_data[0:, 0]

model.fit(x_train, x_train_digit)


''' Test data '''
test_data = pd.read_csv("digit-recognizer/test.csv").to_numpy()
x_test = test_data[0:, 0:]


''' Predicting '''
p = model.predict(x_test)


f = open("digit-recognizer/our_submission_random_forest.csv", mode='w')
f.write("ImageId,Label\n")
for i in range(len(p)):
    f.write(f"{i+1},{p[i]}\n")

f.close()

