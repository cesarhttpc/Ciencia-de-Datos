import numpy as np
from scipy.stats import poisson
import random


def factorial(n):
    f = n
    if n > 0 and n == n//1:
        while n > 1:
            f = f*(n-1)
            n = n-1
    elif n == 0:
        f = 1
    else:
        print('Entrada invalida')
    return f

def ppois(x,lamda):
    '''
    Función de distribución Poisson de parametro lambda.
    $\mathbb{P}[X \leq x]$
    con $X\sim Poisson(\lambda)$
    '''
    if lamda <= 0:
        print("Parametro lambda debe ser positivo")
        
        
    Fx = np.zeros(x+1) 
    for i in range(x+1):
        Fx[i] = np.exp(-lamda) * lamda**i / factorial(i)
    
    return sum(Fx)

# Main

p1 = ppois(12,10)
p2 = ppois(17,15) - ppois(12,15)
p3 = 1 - ppois(17,20)

pFracaso = 1 - (p1+p2+p3)/3   #Probabilidad de error
print("La probabilidad teorica es: ", pFracaso)



lamda = [10, 15, 20]
clases = [1,2,3]

error = []

m = 5000
for i in range(m):
    r = random.choice(clases)
    x = poisson.rvs(lamda[r-1])
    if r == 1:
        error.append(x >= 13)
    elif r == 2:
        error.append( (x <= 12) or (x>= 18))
    elif r == 3:
        error.append((x <= 17))
    
print('La proporción de errores es: ',sum(error)/len(error))