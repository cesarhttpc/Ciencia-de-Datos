# %%
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import datasets
    
# load dataset and transform to pandas df

X, y = datasets.load_iris(return_X_y=True)
X = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(4)])
y = pd.DataFrame(y, columns=['labels'])
tot = pd.concat([X,y], axis=1)

# calculate class means
class_means = tot.groupby('labels').mean()
total_mean = X.mean()

x_mi = tot.transform(lambda x: x - class_means.loc[x['labels']], axis=1).drop('labels', axis=1)
# Calcula basicamente x-m_i (busca la media correspondiente al label y quita los labels

def variance(df, weights):
    dim = df.shape[1]
    S = np.zeros((dim,dim))
    for idx, row in df.iterrows():
        x_m = row.values.reshape(dim,1)
        S += weights[idx]*np.dot(x_m, x_m.T)
    return S

# Each x_mi is weighted with 1. Now we use the kronecker_and_sum function to calculate the within-class scatter matrix S_w
S_w = variance(x_mi, 150*[1])



mi_m = class_means.transform(lambda x: x - total_mean, axis=1)
# Each mi_m is weighted with the number of observations per class which is 50 for each class in this example. We use kronecker_and_sum to calculate the between-class scatter matrix.

S_b=variance(mi_m, 3*[50])



eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_w).dot(S_b))
print(eig_vals)

W = eig_vecs[:, :1]
X_trafo = np.dot(X, W)
tot_trafo = pd.concat([pd.DataFrame(X_trafo, index=range(len(X_trafo))), y], axis=1)

plt.scatter(tot_trafo[0], tot_trafo[0]*0, c=tot_trafo["labels"], s=50, cmap='viridis');
# plt.savefig("Fisher_transfo1D.pdf")
plt.clf()

W = eig_vecs[:, :2]
X_trafo = np.dot(X, W)
tot_trafo = pd.concat([pd.DataFrame(X_trafo, index=range(len(X_trafo))), y], axis=1)

plt.scatter(tot_trafo[0], tot_trafo[1], c=tot_trafo["labels"], s=50, cmap='viridis');

# plt.savefig("Fisher_transfo2D.pdf")
