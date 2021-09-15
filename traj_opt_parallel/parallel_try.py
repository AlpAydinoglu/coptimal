from math import sqrt
from joblib import Parallel, delayed
import numpy as np
from path_matlab import*
from QP_function import*
from evaluate_system import*
import timeit


asd = Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))


mylist = range(10)

x = [[0],[0],[1.36],[0],[0.2],[0],[-0.3],[0],[-0.7],[0]]
x = np.asarray(x)
dt = 0.01
A,B,D,d,E,c,F,H = Evaluate_system( x, dt)


starttime = timeit.default_timer()

asd = Parallel(n_jobs=5)(delayed(  Evaluate_system )   (z,dt)  for   z in [x,x,x,x,x])

t_diff = timeit.default_timer() - starttime

print(t_diff)

starttime = timeit.default_timer()

for i in range(5):
    Evaluate_system(x,dt)

t_diff = timeit.default_timer() - starttime

print(t_diff)
    
input("Press enter")

#asd = Parallel(n_jobs=2)(delayed(  z * 5  )(z)  for z in range(1) )