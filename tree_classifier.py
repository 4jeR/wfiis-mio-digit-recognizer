import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import csv
from sklearn.metrics import plot_confusion_matrix
import time



dtc = DecisionTreeClassifier()

''' Training data'''
train_data = pd.read_csv("digit-recognizer/train.csv").to_numpy()
x_train = train_data[0:10500, 1:]
y_train_digit = train_data[0:10500, 0]


''' Test data '''
x_test = train_data[10500:, 1:]
y_test_digit = train_data[10500:, 0]


dtc.fit(x_train, y_train_digit)

predicted = dtc.predict(x_test)




''' Dokladnosc oraz drzewo '''
print("Accuracy: ", dtc.score(x_test, y_test_digit))
fig, ax = plt.subplots(figsize=(12, 12))  # whatever size you want
tree.plot_tree(dtc, ax=ax, max_depth=1, fontsize=7)




''' Macierze pomylek '''
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(dtc, x_test, y_test_digit,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()


''' Kaggle results '''






test_data = pd.read_csv("digit-recognizer/test.csv").to_numpy()
x_test = test_data[0:, 0:]
x_train = train_data[0:, 1:]
y_train_digit = train_data[0:, 0]



start = time.time()
dtc.fit(x_train, y_train_digit)
predicted = dtc.predict(x_test)


end = time.time()
print(f"Decision Tree Clasifier time = {end - start}")








f = open("digit-recognizer/our_submission_dtc.csv", mode='w')
f.write("ImageId,Label\n")
for i in range(len(predicted)):
    f.write(f"{i+1},{predicted[i]}\n")

f.close()


