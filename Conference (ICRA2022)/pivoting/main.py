from evaluate_system import*
from lemke_lcp import*
import numpy as np
import matplotlib.pyplot as plt
import torch
from path_matlab import*
from QP_function import*
from Projection_function import*
from Projection_function2 import*
from Projection_function3 import*
import numpy, scipy.io


n=10 
m=10
k=4
N = 10
TOT = N*(n+m+k) + n
rho_scale = 1.1   
admm_iter = 5   
rho = 0.02

dist_amp = 0.1

system_iter = 7000
eps = 1e-12
dt = 0.001

x = np.zeros((n, system_iter+1)) 
lam = np.zeros((m, system_iter))
uu = np.zeros((k,system_iter))

cp = np.zeros((1,system_iter))
phinext = np.zeros((1,system_iter))
linear_phinext = np.zeros((1,system_iter))


x0 = [[0],[0],[1.36],[0],[0.2],[0],[-0.3],[0],[-0.7],[0]]

x0 = np.asarray(x0)
x[:, [0]]  = x0


delta = np.zeros((N*(n+k+m) + n,1))
omega = np.zeros((N*(n+k+m) + n,1))

t_calc = np.zeros((1,system_iter))
t_qp = np.zeros((1,system_iter))
t_diff = 0


for i in range(system_iter):
    
    if np.mod(i,10) == 0:
        print(i)
    
    A,B,D,d,E,c,F,H = Evaluate_system( x[:, [i]], dt*10)
    
    xcur = x[:, [i]]
    
    
    if i % 150 == 0:
        delta = 0*np.ones((N*(n+k+m) + n,1))
    #ADMM
    if i % 70 == 0:

        for j in range(N):
                delta[(n+m+k)*j:(n+m+k)*j + n] = x[:, [i]]
                
        omega = 0*np.ones((N*(n+k+m) + n,1))
        G = rho*np.eye(n+m+k)
        
        #ADMM routine
        for j in range(admm_iter):
            #SOLVE THE QP
            z, t_diff_qp = QP_function(x[:, [i]] ,delta,omega, N,G, A,B,D,d,E,c,F,H)
            z = np.reshape(z, (np.size(z,0),1))
            t_qp[:, [i]] = t_diff_qp
            
    
            #PROJECTION STEP
            #delta = Projection_function(z, delta, omega, N)
            delta, t_diff = Projection_function2(z, delta, omega, N, G, A,B,D,d,E,c,F,H)
            #delta, t_diff = Projection_function3(z, delta, omega, N, A,B,D,d,E,c,F,H)
           
            
            t_calc[:, [i]] = t_diff
            
            #UPDATE STEP
            omega = omega + z - delta
            G = G * rho_scale
            omega = omega / rho_scale
    
        u = z[n+m:n+m+k]   

        if u[2] < 0:
            u[2] = 0
        if u[3] < 0:
            u[3] = 0
        
    uu[:,[i]] = u
    

    

    A,B,D,d,E,c,F,H = Evaluate_system( x[:, [i]], dt)        
    #calculate q
    q = E @ x[:, [i]] + H @ u + c


       
    cp[0,i] = xcur[2] - np.cos( xcur[4] ) - np.sin(  xcur[4] )
    
    #use lemkelcp
    sol_lcp = lemkelcp(F + eps*np.eye(m),q)
    lam[:,[i]] = np.reshape( sol_lcp[0], (m,1))
    
    #dist = dt*dist_amp*np.random.rand(10)
    dist = dist_amp*dt*np.random.normal(size=10)
    dist = np.reshape(dist, (1,10))
    
    #dynamics update
    x[:,[i+1]] = A @ x[:, [i]] + D @ lam[:,[i]] + d + B @ u + dist.T
    
    #plotting gap function    
    xnext = x[:,[i+1]]
    phinext[0,i] = xnext[2] - np.sin( xnext[4] ) - np.cos( xnext[4] )
    linear_phinext[0,i] = F[9,9] * lam[9,i] + q[9]
    linear_phinext[0,i] = linear_phinext[0,i] 
          

plt.figure()
time_x = np.arange(0, system_iter * dt + dt, dt)
plt.plot(time_x, x[[0,2,4,6,8],:].T)
plt.legend(['x','y','theta','finger_top','finger bottom'])
plt.show()



print(np.mean(t_calc[t_calc != 0]  ))


print(np.mean(t_qp[t_qp != 0]  ))






input("Press enter")

