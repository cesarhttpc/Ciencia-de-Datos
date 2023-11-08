# %%
import numpy as np
import matplotlib.pylab as plt
import scipy.io as sio 

# %%
# Leer la base de datos
mat = sio.loadmat('data1.mat')
# mat = sio.loadmat('faces.mat')

X = mat['X']
X

# %%
# Graficar el scatter
# plt.figure()
# plt.scatter(X[:,0], X[:,1], facecolors = 'none', edgecolor='b')
# plt.show()

# Normalización del profe
def feature_normalize2(X):
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0, ddof= 1)
    X_norm = (X-mu)/sigma
    return X_norm, mu, sigma

def pca (X):

    sigma = X.T@X/X.shape[0]

    U,S,V = np.linalg.svd(sigma)
    return U,S,V

def draw_line(A,B,dash = False):
    if dash:
        plt.plot([A[0],B[0]], [A[1],B[1]], color = 'black',linestyle = '--')
    else:
        plt.plot([A[0],B[0]], [A[1],B[1]], color = 'black')

def project_data(X,U,K):
    Z = X @ U[:,0:K]
    return Z

def recover_data(Z,U,K): 
    X_rec = Z @ U[:,0:K].T
    return X_rec

# %%
# MAIN
X_norm ,mu, sigma = feature_normalize2(X)

plt.scatter(X_norm[:,0],X_norm[:,1])
plt.show()

U,S,V = pca(X_norm)

plt.scatter(X[:,0],X[:,1],facecolors = 'none', edgecolor='b')
draw_line(mu, mu + S[0]*U[:,0].T)
draw_line(mu, mu + S[1]*U[:,1].T)
plt.show()



K=1
Z = project_data(X_norm,U,K)
# print(Z.shape)

X_rec = recover_data(Z,U,K)
# print(X_rec)

plt.scatter(X_norm[:,0],X_norm[:,1],facecolors = 'none', edgecolor='b')
# draw_line( np.zeros([0,0]) , U[:,0].T)
# draw_line(mu, mu + S[1]*U[:,1].T)
plt.scatter(X_rec[:,0],X_rec[:,1])

for i in range(X_norm.shape[0]):
    draw_line(X_norm[i,:],X_rec[i,:],dash = True)

# plt.scatter(X_rec[:,0],X_rec[:,1])













# %%
# Ejercicio 2 ----------------------------------------
import numpy as np
import matplotlib.pylab as plt
import scipy.io as sio 
from display_data import display_data

# %%
# Leer la base de datos
mat = sio.loadmat('faces.mat')
X = mat['X']
print(X.shape)

# print(X)


display_data(X[:9,:])


K = 10
X_norm, mu, sigma = feature_normalize2(X)
U, S, V = pca(X_norm)

Z = project_data(X_norm, U ,K)
X_rec = recover_data(Z, U , K)

fig = plt.figure(figsize = (10,10))
ax1 = fig.add_subplot(1,2,1)    
display_data(X_norm[1:K,:], axes = ax1)
ax1.set_title('Caras normalizadas')
ax2 = fig.add_subplot(1,2,2)
display_data(X_rec[1:K,:], axes= ax2)
ax2.set_title('Caras proyectadas')
plt.show()





