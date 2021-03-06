import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
# diabetes = diabete

diabete = datasets.load_diabetes()
# (['data', 'target', 'frame', 'DESCR',
#  'feature_names', 'data_filename', 'target_filename']
# print(diabetes.DESCR)

diabete_X = np.array([[1], [2], [3]])

diabete_X_train = diabete_X
diabete_X_test = diabete_X

diabete_Y_train = np.array([3, 2, 4])
diabete_Y_test = np.array([3, 2, 4])

model = linear_model.LinearRegression()


model.fit(diabete_X_train, diabete_Y_train)

diabete_Y_predicted = model.predict(diabete_X_test)

# print(diabete_Y_predict)

print("Mean Squared error is : ", mean_squared_error(diabete_Y_test, diabete_Y_predicted))

print("Weights :", model.coef_)
print("Intercept :", model.intercept_)

# Visualization

plt.scatter(diabete_X_test, diabete_Y_test)
plt.plot(diabete_X_test, diabete_Y_predicted)

plt.show()
