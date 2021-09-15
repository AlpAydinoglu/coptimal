from evaluate_system import*
from lemke_lcp import*
import numpy as np
import matplotlib.pyplot as plt
import torch
from path_matlab import*

n=10 
m=10
k=4

system_iter = 100
eps = 1e-5
dt = 0.01

x = np.zeros((n, system_iter+1))
lam = np.zeros((m, system_iter))
uu = np.zeros((k,system_iter))

cp = np.zeros((1,system_iter))
phinext = np.zeros((1,system_iter))
linear_phinext = np.zeros((1,system_iter))


x0 = [[0],[0],[1.5],[0],[0.5 ],[0],[1],[0],[1],[0]]
x0 = np.asarray(x0)
x[:, [0]]  = x0



for i in range(system_iter):
    
    if np.mod(i,50) == 0:
        print(i)
    
    A,B,D,d,E,c,F,H = Evaluate_system( x[:, [i]], dt)
    
    u = np.zeros((4,1))
    u[0] = 0
    u[2] = 0
    u[3] = 0
    
    xcur = x[:, [i]]
    
    # if xcur[4] > 0.01:
    #     u[2] = 2
    
    # if xcur[4] < -0.01:
    #     u[3] = 8
        
    #calculate q
    q = E @ x[:, [i]] + H @ u + c
    
    #q[9] = q[9] /dt   
    #F[9,:] = F[9,:] / dt
    
    #F = F + eps*np.eye(n)
    
    #use pathlcp
    F = torch.from_numpy(F)
    q = torch.from_numpy(q)
    sol_lcp = solve_lcp_path(F, q)
    sol_lcp = sol_lcp.detach().cpu().numpy()
    lam[:,[i]] = sol_lcp
    
    #lam[9,[i]] = max(0, -q[9]/F[9,9])

    
    cp[0,i] = xcur[2] - np.cos( xcur[4] ) - np.sin(  xcur[4] )
    
    #use lemkelcp
    # sol_lcp = lemkelcp(F + eps*np.eye(m),q)
    # lam[:,[i]] = np.reshape( sol_lcp[0], (m,1))
    
    #dynamics update
    x[:,[i+1]] = A @ x[:, [i]] + D @ lam[:,[i]] + d + B @ u
    
    
    xnext = x[:,[i+1]]
    phinext[0,i] = xnext[2] - np.sin( xnext[4] ) - np.cos( xnext[4] )
    linear_phinext[0,i] = F[9,9] * lam[9,i] + q[9]
    linear_phinext[0,i] = linear_phinext[0,i] 
      
    
time_x = np.arange(0, system_iter * dt + dt, dt)
plt.plot(time_x, x[[0,2,4],:].T)
plt.legend(['x','y','theta'])
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




