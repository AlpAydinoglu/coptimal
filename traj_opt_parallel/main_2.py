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

n=10 
m=10
k=4
N = 10
TOT = N*(n+m+k) + n
rho_scale = 1.1 #1.2  2 
admm_iter = 5 #10    5
rho = 0.1

system_iter = 150
eps = 1e-5
dt = 0.01

x = np.zeros((n, system_iter+1))
lam = np.zeros((m, system_iter))
uu = np.zeros((k,system_iter))

cp = np.zeros((1,system_iter))
phinext = np.zeros((1,system_iter))
linear_phinext = np.zeros((1,system_iter))


x0 = [[0],[0],[1.36],[0],[0.5],[0],[0.5],[0],[0.5],[0]]
x0 = np.asarray(x0)
x[:, [0]]  = x0


delta = np.zeros((N*(n+k+m) + n,1))
omega = np.zeros((N*(n+k+m) + n,1))


for i in range(system_iter):
    
    if np.mod(i,10) == 0:
        print(i)
    
    A,B,D,d,E,c,F,H = Evaluate_system( x[:, [i]], dt)
    
    xcur = x[:, [i]]
    
    #ADMM
    if i % (0.1/dt) == 0:
        delta = 0*np.ones((N*(n+k+m) + n,1))

    for j in range(N):
            delta[(n+m+k)*j:(n+m+k)*j + n] = x[:, [i]]
            
    omega = 0*np.ones((N*(n+k+m) + n,1))
    G = rho*np.eye(n+m+k)
    
    #ADMM routine
    for j in range(admm_iter):
        #SOLVE THE QP
        z, t_diff_qp = QP_function(x[:, [i]] ,delta,omega, N,G, A,B,D,d,E,c,F,H)
        z = np.reshape(z, (np.size(z,0),1))
        
        #t_qp[:, [i]] = t_diff_qp
        
        #print(j)

        
        #PROJECTION STEP
        #delta = Projection_function(z, delta, omega, N)
        delta, t_diff = Projection_function2(z, delta, omega, N, G, A,B,D,d,E,c,F,H)
        #delta, t_diff = Projection_function3(z, delta, omega, N, A,B,D,d,E,c,F,H)
       
        
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
    
    

        
    #calculate q
    q = E @ x[:, [i]] + H @ u + c

    
    #use pathlcp
    # F = torch.from_numpy(F)
    # q = torch.from_numpy(q)
    # sol_lcp = solve_lcp_path(F, q)
    # sol_lcp = sol_lcp.detach().cpu().numpy()
    # lam[:,[i]] = sol_lcp
       
    cp[0,i] = xcur[2] - np.cos( xcur[4] ) - np.sin(  xcur[4] )
    
    #use lemkelcp
    sol_lcp = lemkelcp(F + eps*np.eye(m),q)
    lam[:,[i]] = np.reshape( sol_lcp[0], (m,1))
    
    #dynamics update
    x[:,[i+1]] = A @ x[:, [i]] + D @ lam[:,[i]] + d + B @ u
    
    #plotting gap function    
    xnext = x[:,[i+1]]
    phinext[0,i] = xnext[2] - np.sin( xnext[4] ) - np.cos( xnext[4] )
    linear_phinext[0,i] = F[9,9] * lam[9,i] + q[9]
    linear_phinext[0,i] = linear_phinext[0,i] 
      
    
time_x = np.arange(0, system_iter * dt + dt, dt)
plt.plot(time_x, x[[0,2,4,6,8],:].T)
plt.legend(['x','y','theta','finger_top','finger bottom'])
plt.show()

# plt.plot(cp.T)
# plt.show()

plt.figure()
plt.plot(phinext.T)
plt.plot(linear_phinext.T)
#plt.xlim([20,40])
#plt.ylim([-0.01, 0.02])
plt.legend(['real(gap)','linearization (gap)'])
plt.show()




