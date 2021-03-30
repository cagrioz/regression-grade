import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle


data = pd.read_csv('student-mat.csv', sep=';')
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences', 'freetime', 'goout', 'Walc', 'Dalc']]
predict = 'G3'
x = np.array(data.drop([predict], 1))
y = np.array(data[[predict]])   
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = .1)
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(np.rint(x_test), y_test)
predictions = linear.predict(x_test)

total = [""]
for x in range(len(predictions)):
    new1 = np.rint(predictions[x][0])
    new2 = x_test[x][0]
    total.append("[" + str(new1) + ", ")
    total.append(str(new2) + ".0] - ")

res = ' '.join(total)

print(res)
print("ACCURACY: " + str(np.rint(acc * 100)))

