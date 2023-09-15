
from sklearn.datasets import load_iris
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
X = iris.data[:,:2]
Y = iris.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

# Clasidicación con datos de entrenamiento.
logisticRegr = LogisticRegression()
clasificador = logisticRegr.fit(x_train, y_train)

# Crear una malla, la cual está pensada para testear el modelo.
feature_1, feature_2 = np.meshgrid(
np.linspace(X[:,0].min(), X[:,0].max(),1000),
np.linspace(X[:,1].min(), X[:,1].max(),1000) )

grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T

# Predicción
y_pred = np.reshape(clasificador.predict(grid), feature_1.shape)


# Graficación 
display = DecisionBoundaryDisplay(xx0=feature_1, xx1=feature_2, response=y_pred )
display.plot()

display.ax_.scatter(X[:,0], X[:, 1], c = Y, edgecolor="black")
plt.show()

# Precisión del modelo. Cantidad de datos correctos versus número total de datos.
score = logisticRegr.score(x_test, y_test)
print("Proporción de veces que acierta el modelo: ", score)


# # Gráfica
# plt.scatter(X[:50,0], X[:50,1], label = "1")
# plt.scatter(X[50:100,0],X[50:100,1], label = "2")
# plt.scatter(X[100:150,0],X[100:150,1], label = "3")
# plt.legend()
# # plt.show()
