#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# %%

# Imagine we have the following data

from sklearn.datasets import make_blobs

X, y = make_blobs(100, 2, centers = 2, random_state= 2, cluster_std= 1.5)
plt.scatter(X[:,0], X[:,1], c = y, s = 50, cmap= 'RdBu')



# %%
# Ajustamos el modelo
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X,y)

rgn = np.random.RandomState(0)
Xnew = [-6,-14] + [14,18] *rgn.rand(2000 ,2)
ynew = model.predict(Xnew)

plt.scatter(X[:,0],X[:,1], c=y, s = 50, cmap = 'RdBu')
lim = plt.axis()
plt.scatter(Xnew[:,0], Xnew[:,1], c = ynew, s = 20, cmap= 'RdBu', alpha= 0.1)
plt.axis(lim)

# %%

yprob = model.predict_proba(Xnew)
yprob[-8:].round(2)
