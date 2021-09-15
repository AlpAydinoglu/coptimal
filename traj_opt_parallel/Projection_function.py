import numpy as np

def Projection_function(z, delta, omega, N):
    
    z = np.reshape(z, (np.size(z,0),1))
    n=4
    m=2
    k=1
    TOT = n+m+k
    #system parameters
    g = 9.81
    mp = 0.411
    mc = 0.978
    len_p = 0.6
    len_com = 0.4267
    d1 = 0.35
    d2 = -0.35
    ks= 50
    E = [[-1, len_p, 0, 0], [1, -len_p, 0, 0 ]]
    E = np.asarray(E)
    F = 1/ks * np.eye(2)
    F = np.asarray(F)
    c = [[d1], [-d2]]
    c = np.asarray(c)
    H = np.zeros((2,1))
    
    for i in range(N):
        cons = z[TOT*i:TOT*(i+1)] + omega[TOT*i:TOT*(i+1)]
        x = cons[0:n]
        lam = ks * np.maximum( np.zeros((2,1)), (- E @ x - c) ) 
        uu = cons[n+m:n+m+k]
        delta[TOT*i:TOT*(i+1)] = np.vstack((x, lam, uu))
        
    #for cartpole
    delta[0:n] = np.zeros((n,1))
        
    return delta
