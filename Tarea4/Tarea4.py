import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


np.random.seed(2) #21, 3, 2

# Inciso a)

# Leer base de datos
iris = load_iris()
X = iris.data[:,2:]
Y = iris.target

# Verdadera clasificación
plt.scatter(X[:,0] , X[:,1] , c = Y)
plt.xlabel('Largo de pétalo')
plt.ylabel('Ancho de pétalo')
plt.title('Clasificación predeterminada')
plt.show()


# K-means graficas
n = 2
kmeans = KMeans(n_clusters=n)
kmeans.fit(X)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'purple')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'orange')
plt.title('K-Means con K = 2')
plt.xlabel('Largo de pétalo')
plt.ylabel('Ancho de pétalo')
plt.show()


n = 3
kmeans = KMeans(n_clusters=n)
kmeans.fit(X)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'purple')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'orange')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green')
plt.title('K-Means con K = 3')
plt.xlabel('Largo de pétalo')
plt.ylabel('Ancho de pétalo')
plt.show()


n = 4
kmeans = KMeans(n_clusters=n)
kmeans.fit(X)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'purple')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'orange')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'blue')
plt.title('K-Means con K = 4')
plt.xlabel('Largo de pétalo')
plt.ylabel('Ancho de pétalo')
plt.show()


# Inciso b)

def riesgo(X,k):
    '''
    Input:
            X es base de datos
            k es el número de clusters.
    '''
    # Base de datos
    df = pd.DataFrame(X, columns=['Largo', 'Ancho'])

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    y_kmeans = kmeans.fit_predict(X)
    df['Cluster'] = y_kmeans

    riesgo = 0
    for i in range(k):

        #Filtrar por clase i
        d0 = df[df['Cluster'] == i]

        #Calculo de la media
        x = d0['Largo']-d0['Largo'].mean()
        y = d0['Ancho']-d0['Ancho'].mean()

        # plt.plot(d0['Largo'].mean(), d0['Ancho'].mean(), '*k') #plot
        dist = (x**2 + y**2).sum()
        riesgo += dist
        # print(dist)

    return riesgo


# Inciso c)

RiesgoVal = []
for j in range(2,11):
    RiesgoVal.append(riesgo(X,j))

clust = np.linspace(2,10,9)


plt.plot(clust,RiesgoVal)
plt.title('Riesgo')
plt.xlabel('Numero de clusters')


