
'''
from sklearn import datasets
import matplotlib.pyplot as plt


iris = datasets.load_iris()
X = iris.data[:,:2]
Y = iris.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.90, random_state=0)



from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)



score = logisticRegr.score(x_test, y_test)
print(score)



predictions = logisticRegr.predict(x_test)



from sklearn.inspection import DecisionBoundaryDisplay

# display = DecisionBoundaryDisplay(xx0=X[:,0], xx1=X[:,1], response=predictions)



disp = DecisionBoundaryDisplay.from_estimator( LogisticRegression, X, response_method="predict", alpha=0.5)
# disp.ax_.scatter(X[:, 0], X[:, 1], c=iris.target, edgecolor="k")

plt.show()






# Gráfica
plt.scatter(X[:50,0], X[:50,1], label = "1")
plt.scatter(X[50:100,0],X[50:100,1], label = "2")
plt.scatter(X[100:150,0],X[100:150,1], label = "3")
plt.legend()
# plt.show()
'''


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.inspection import DecisionBoundaryDisplay
# from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression



logisticRegr = LogisticRegression()
iris = load_iris()

# iris = datasets.load_iris()
X = iris.data[:,:2]
Y = iris.target

feature_1, feature_2 = np.meshgrid(
np.linspace(iris.data[:, 0].min(), iris.data[:, 0].max()),
np.linspace(iris.data[:, 1].min(), iris.data[:, 1].max()))

grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T

classifier = LogisticRegression().fit(X, iris.target)
# tree = DecisionTreeClassifier().fit(iris.data[:, :2], iris.target)

y_pred = np.reshape(classifier.predict(grid), feature_1.shape)
display = DecisionBoundaryDisplay(
xx0=feature_1, xx1=feature_2, response=y_pred )
display.plot()

display.ax_.scatter(
iris.data[:, 0], iris.data[:, 1], c=iris.target, edgecolor="black")

plt.show()

