# Importing.. req. modules..
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
 

# Loading Datasets
iris = datasets.load_iris()
# print(list(iris.keys()))
# print(iris['data'].shape)
# print(iris['target'])
# print(iris['DESCR'])
# - Iris-Setosa
# - Iris-Versicolour
# - Iris-Virginica

x = iris['data'][:, 3:]
y = (iris['target'] == 2).astype(np.int)

# Train a logisitc_regrssion classifier 
clf = LogisticRegression()
clf.fit(x, y)

ex = clf.predict(([[2.6]]))
print(ex)
# Using matplotlib for visualization
x_new = np.linspace(0,3,1000).reshape(-1, 1)
y_prob = clf.predict_proba(x_new)
plt.plot(x_new,y_prob[:,1], "g-", label="viriginica")
plt.show()
# print(x_new)
print(y_prob)
# print(y)
# print(iris['data'])
# print(x)