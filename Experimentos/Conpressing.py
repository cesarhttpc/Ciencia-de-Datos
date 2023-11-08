from matplotlib.image import imread
import matplotlib.pyplot as plt
import scipy.linalg as ln
import numpy as np
import os
from PIL import Image
from math import log10, sqrt
plt.rcParams['figure.figsize'] = [16, 8]
# Import image
A = imread(os.path.join("Cat-1.jpg"))
X = A.dot([0.299, 0.5870, 0.114]) # Convert RGB to grayscale



# Descomposición SVD
U, S, VT = np.linalg.svd(X,full_matrices=False)
D = np.diag(S)
n = U.shape[0] # Numero de filas
d = VT.shape[1] # Numero de columnas

print(f"Imagen de {d}x{n} pixeles => {d*n} parámetros")

k= 10
print(f"Se guardan {np.round(k*(1+d+n)/(n*d)*100,2)}% de los parámetros")

print(S[k+1]/S[0])
Xapprox = U[:,:k] @ D[0:k,:k] @ VT[:k,:]
img = plt.imshow(Xapprox)
img.set_cmap('gray')
plt.axis('off')
plt.show()