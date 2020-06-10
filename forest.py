import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix, classification_report, accuracy_score
import time

# 5 20 50 100 200

rfc = RandomForestClassifier(n_estimators=200)


''' Training data'''
train_data = pd.read_csv("digit-recognizer/train.csv").to_numpy()
x_train = train_data[0:38000, 1:]
y_train_digit = train_data[0:38000, 0]


''' Test data '''
x_test = train_data[38000:, 1:]
y_test_digit = train_data[38000:, 0]


rfc.fit(x_train, y_train_digit)

predicted = rfc.predict(x_test)


# print(classification_report(y_test_digit, predicted))
print("Accuracy: ", accuracy_score(y_test_digit, predicted))


''' Macierze pomylek '''
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(rfc, x_test, y_test_digit,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    # print(title)
    # print(disp.confusion_matrix)

plt.show()


''' Kaggle results '''
test_data = pd.read_csv("digit-recognizer/test.csv").to_numpy()
x_test = test_data[0:, 0:]

x_train = train_data[0:, 1:]
y_train_digit = train_data[0:, 0]



start = time.time()
rfc.fit(x_train, y_train_digit)
predicted = rfc.predict(x_test)


end = time.time()
print(f"Decision Tree Clasifier time = {end - start}")











f = open("digit-recognizer/our_submission_rfc.csv", mode='w')
f.write("ImageId,Label\n")
for i in range(len(predicted)):
    f.write(f"{i+1},{predicted[i]}\n")

f.close()



