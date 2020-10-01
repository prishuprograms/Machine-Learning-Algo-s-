# Loading req. modules...
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#  Laoding Dataset
iris = datasets.load_iris()

#  Printing data"s discription and features
# print(iris.DESCR)
features = iris.data
labels = iris.target
# print(features[0], labels[0])

#  Training Classifier
clf = KNeighborsClassifier()
clf.fit(features, labels)

# Predicition part 
preds = clf.predict([[9.1, 9.5, 5.4, 9.2]])
print(preds)




