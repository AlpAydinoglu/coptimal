from QP_function import*
from Projection_function import*
from Projection_function2 import*
from Projection_function3 import*
from Projection_function4 import*
from lemke_lcp import*
import matplotlib.pyplot as plt
import numpy as np
import numpy, scipy.io


#system_parameters
n = 6
m = 6
k = 4
N = 10
TOT = N*(n+m+k) + n
rho_scale = 1.2 #1.2  2 
admm_iter = 10 #10    5
rho = 1
system_iter = 3000 #3000
dt = 0.001 #0.001
   
g = 9.81
mu = 1
h = 0.1
A = [[1,h,0,0,0,0],[0, 1, 0, 0, 0, 0], [0, 0, 1, h, 0, 0 ], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, h], [0, 0, 0, 0, 0, 1]]
A = np.asarray(A)
B = [[0,0,0,0], [0, 0, 0, 0], [h*h, 0, 0, 0], [h, 0, 0, 0], [0, h*h, 0, 0], [0, h, 0, 0]]
B = np.asarray(B)
D = [[0, h*h, -h*h, 0, h*h, -h*h], [0, h, -h, 0, h, -h], [0, -h*h, h*h, 0, 0, 0], [0, -h, h, 0, 0, 0], [0, 0, 0, 0, -h*h, h*h], [0, 0, 0, 0, -h, h]]
D = np.asarray(D)
E = [[0, 0, 0, 0, 0, 0], [0, 1, 0, -1, 0, 0], [0, -1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, -1], [0, -1, 0, 0, 0, 1]]
E = np.asarray(E)
F = [[0, -1, -1, 0, 0, 0], [1, 2*h, -2*h, 0, h, -h], [1, -2*h, 2*h, 0, -h, h], [0, 0, 0, 0, -1,-1], [0, h, -h, 1, 2*h, -2*h], [0, -h, h, 1, -2*h, 2*h]]
F = np.asarray(F)
c = [[0],[-h*g], [h*g], [0], [-h*g], [h*g]]
c = np.asarray(c)
d = [[-g*h*h],[-g*h],[0],[0],[0],[0]]
d = np.asarray(d)
H = [[0, 0, mu, 0], [-h, 0, 0, 0], [h, 0, 0, 0], [0, 0, 0, mu], [0, -h, 0, 0], [0, h, 0, 0]]
H = np.asarray(H)

#######simulation

g = 9.81
mu = 1
h = dt
Asim = [[1,h,0,0,0,0],[0, 1, 0, 0, 0, 0], [0, 0, 1, h, 0, 0 ], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, h], [0, 0, 0, 0, 0, 1]]
Asim  = np.asarray(Asim )
Bsim  = [[0,0,0,0], [0, 0, 0, 0], [h*h, 0, 0, 0], [h, 0, 0, 0], [0, h*h, 0, 0], [0, h, 0, 0]]
Bsim  = np.asarray(Bsim )
Dsim  = [[0, h*h, -h*h, 0, h*h, -h*h], [0, h, -h, 0, h, -h], [0, -h*h, h*h, 0, 0, 0], [0, -h, h, 0, 0, 0], [0, 0, 0, 0, -h*h, h*h], [0, 0, 0, 0, -h, h]]
Dsim  = np.asarray(Dsim )
Esim  = [[0, 0, 0, 0, 0, 0], [0, 1, 0, -1, 0, 0], [0, -1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, -1], [0, -1, 0, 0, 0, 1]]
Esim  = np.asarray(Esim )
Fsim  = [[0, -1, -1, 0, 0, 0], [1, 2*h, -2*h, 0, h, -h], [1, -2*h, 2*h, 0, -h, h], [0, 0, 0, 0, -1,-1], [0, h, -h, 1, 2*h, -2*h], [0, -h, h, 1, -2*h, 2*h]]
Fsim  = np.asarray(Fsim )
csim  = [[0],[-h*g], [h*g], [0], [-h*g], [h*g]]
csim  = np.asarray(csim)
dsim  = [[-g*h*h],[-g*h],[0],[0],[0],[0]]
dsim  = np.asarray(dsim )
Hsim  = [[0, 0, mu, 0], [-h, 0, 0, 0], [h, 0, 0, 0], [0, 0, 0, mu], [0, -h, 0, 0], [0, h, 0, 0]]
Hsim = np.asarray(Hsim )


delta = np.zeros((N*(n+k+m) + n,1))
omega = np.zeros((N*(n+k+m) + n,1))
#x0 = [[0.1],[-0.3],[0],[0]]
x0 = [[-8],[0],[3],[0],[4],[0]]
x0 = np.asarray(x0)

x = np.zeros((n, system_iter+1))
lam = np.zeros((m, system_iter))
uu = np.zeros((k,system_iter))
x[:, [0]]  = x0

t_calc = np.zeros((1,system_iter))
t_qp = np.zeros((1,system_iter))
t_diff = 0


for i in range(system_iter):
    
    if i % 1000 == 0:
        print(i)
    
    
    if i % (0.1/dt) == 0:
    
        delta = 0*np.ones((N*(n+k+m) + n,1))
        
        for j in range(N):
            delta[(n+m+k)*j:(n+m+k)*j + n] = x[:, [i]]
        
        omega = 0*np.ones((N*(n+k+m) + n,1))
        G = rho*np.eye(n+m+k)
        #u sifirla
        #G[n+m+k-1,n+m+k-1] = 0
        #rho = 1
        
        
        #ADMM routine
        step = i
        starttime = timeit.default_timer()
        for j in range(admm_iter):
            #SOLVE THE QP
            z, t_diff_qp = QP_function(x[:, [i]] ,delta,omega, N,G, step)
            z = np.reshape(z, (np.size(z,0),1))
            
            #t_qp[:, [i]] = t_diff_qp
            
            #print(j)
    
            
            #PROJECTION STEP
            #delta = Projection_function(z, delta, omega, N)
            delta, t_diff = Projection_function2(z, delta, omega, N, G, A,B,D,d,E,c,F,H)
            #delta, t_diff = Projection_function4(z, delta, omega, N, G)
            #delta, t_diff = Projection_function3(z, delta, omega, N)
           
            
            #t_calc[:, [i]] = t_diff
            
            #UPDATE STEP
            omega = omega + z - delta
            #for cartpole
            #omega[0:6] = np.zeros((n,1))
            #rho = rho * rho_scale
            G = G * rho_scale
            omega = omega / rho_scale
    
        u = z[n+m:n+m+k]
        uu[:,[i]] = u
        t_diff2 = timeit.default_timer() - starttime
        print(t_diff2)
    
    qsim = Esim @ x[:, [i]] + Hsim @ u + csim
    
    eps = 10e-5
    qsim[0] = qsim[0] + eps
    qsim[3] = qsim[3] + eps

    #DEGISTIR
    sol_lcp = lemkelcp(Fsim,qsim)
    lam[:,[i]] = np.reshape( sol_lcp[0], (m,1))
    x[:,[i+1]] = Asim @ x[:, [i]] + Dsim @ lam[:,[i]] + dsim + Bsim @ u
    
    #uu[:,[i]] = u


time_x = np.arange(0, system_iter * dt + dt, dt)
plt.plot(time_x, x.T)
plt.show()


#print(np.mean(t_calc[t_calc != 0]  ))
#print(np.std(t_calc[t_calc != 0] ))

#print(np.mean(t_qp[t_qp != 0]  ))
#print(np.std(t_qp[t_qp != 0] ))


input("Press enter")

