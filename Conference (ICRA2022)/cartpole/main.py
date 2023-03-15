from QP_function import*
from Projection_function import*
from Projection_function2 import*
from Projection_function3 import*
from lemke_lcp import*
import matplotlib.pyplot as plt
import numpy, scipy.io

#system_parameters
n = 4
m = 2
k = 1
N = 10
TOT = N*(n+m+k) + n
rho_scale = 2
admm_iter = 10
rho = 0.1
system_iter = 800
dt = 0.01
   
g = 9.81
mp = 0.411
mc = 0.978
len_p = 0.6
len_com = 0.4267
d1 = 0.35
d2 = -0.35
ks= 50
Ts = 0.01
A = [[0, 0, 1, 0], [0, 0, 0, 1], [0, g*mp/mc, 0, 0], [0, g*(mc+mp)/(len_com*mc), 0, 0]]
A = np.asarray(A)
B = [[0],[0],[1/mc],[1/(len_com*mc)]]
B = np.asarray(B)
D = [[0,0], [0,0], [(-1/mc) + (len_p/(mc*len_com)), (1/mc) - (len_p/(mc*len_com)) ], [(-1 / (mc*len_com) ) + (len_p*(mc+mp)) / (mc*mp*len_com*len_com)  , -((-1 / (mc*len_com) ) + (len_p*(mc+mp)) / (mc*mp*len_com*len_com))    ]]
D = np.asarray(D)
E = [[-1, len_p, 0, 0], [1, -len_p, 0, 0 ]]
E = np.asarray(E)
F = 1/ks * np.eye(2)
F = np.asarray(F)
c = [[d1], [-d2]]
c = np.asarray(c)
d = np.zeros((4,1))
H = np.zeros((2,1))
A = np.eye(n) + Ts * A
B = Ts*B
D = Ts*D
d = Ts*d

delta = np.zeros((N*(n+k+m) + n,1))
omega = np.zeros((N*(n+k+m) + n,1))

#x0 = [[0.1],[-0.3],[0],[0]]

save_z = np.zeros((N*(n+k+m)+n,3))

x0 = [[0.3],[0],[0.3],[0]]
#x0 = [[-0.3],[0.1],[0],[0]]
x0 = np.asarray(x0)

t_calc = np.zeros((1,system_iter))


x = np.zeros((n, system_iter+1))
lam = np.zeros((m, system_iter))
costt = np.zeros((1,system_iter))
x[:, [0]]  = x0

for i in range(system_iter):
    
    if i % 50 == 0:
        print(i)
    
    delta = 0*np.ones((N*(n+k+m) + n,1))
    omega = 0*np.ones((N*(n+k+m) + n,1))
    G = 0.1*np.eye(n+m+k)
    #u sifirla
    G[n+m+k-1,n+m+k-1] = 0
    rho = 0.1
    
    
    #ADMM routine
    for j in range(admm_iter):
        #SOLVE THE QP
        z = QP_function(x[:, [i]] ,delta,omega, N,G)
        z = np.reshape(z, (np.size(z,0),1))
        
        zres = np.reshape(z, (np.size(z,0),))
        if j == 0:
             save_z[:,0] = zres
           
        if j == 2:
            save_z[:,1] = zres
        
        if j == 9:
            save_z[:,2] = zres
        
        #PROJECTION STEP
        delta, t_diff = Projection_function(z, delta, omega, N)    #LCP Projection
        #delta, t_diff = Projection_function2(z, delta, omega, N, G) #MIQP Projection
        #delta, t_diff = Projection_function3(z, delta, omega, N)   #ADMM Projection
        t_calc[:, [i]] = t_diff
        
        #UPDATE STEP
        omega = omega + z - delta
        #for cartpole
        omega[0:4] = np.zeros((n,1))
        #rho = rho * rho_scale
        G = G * rho_scale
        omega = omega / rho_scale

    u = z[n+m:n+m+k]
    
    #calculate the cost
    u_all = z[6::7]
    xc = x[:, [i]]
    Q = [[10, 0, 0, 0], [0, 3, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    Q = np.asarray(Q)
    R = 1
    QN = linalg.solve_discrete_are(A, B, Q, R)
    costt[0,i] = 0
    for p in range(N):
        uc = u_all[p]
        costt[0,i] = xc.T @ Q @ xc + costt[0,i]  + uc * R * uc
        qc = E @ xc + H * uc + c
        sol_c = lemkelcp(F,qc)
        sol_c = np.reshape( sol_c[0], (m,1))
        xc = A @ xc + D @ sol_c + d + B * uc
    costt[0,i] =  costt[0,i] + xc.T @ QN @ xc
    
    
    q = E @ x[:, [i]] + H @ u + c
    
    # li = save_z
    # sub = li[4::7]
    # plt.plot(sub)
    # plt.legend((['z1'],['z3'],['z10']))
    # plt.show()
    # mdic = {"sub": sub, "li": li }
    # scipy.io.savemat("sub.mat",  mdic)
    
    sol_lcp = lemkelcp(F,q)
    lam[:,[i]] = np.reshape( sol_lcp[0], (m,1))
    x[:,[i+1]] = A @ x[:, [i]] + D @ lam[:,[i]] + d + B * u

time_x = np.arange(0, system_iter * dt + dt, dt)
plt.plot(time_x, x.T)
plt.show()

print(np.mean(t_calc))
print(np.std(t_calc))

print(np.mean(costt))

mdic = {"x": x , "t": time_x}
scipy.io.savemat("matlab_matrix.mat",  mdic)