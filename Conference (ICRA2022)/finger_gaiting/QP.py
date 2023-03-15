import osqp
import numpy as np
from scipy import sparse
from scipy.linalg import block_diag
from scipy import linalg


#system_parameters
n = 6
m = 6
k = 10
N = 10
TOT = N*(n+m+k) + n
rho_scale = 1.2
admm_iter = 10
rho = 1
system_iter = 10
dt = 0.1
   
g = 9.81
mu = 1
h = dt
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


#cost matrices
Q = np.eye(n)
S = np.zeros((m,m))
R = np.eye(k)
QN = linalg.solve_discrete_are(A, B, Q, R)

#setup for quadratic cost
Csetup_initX = np.zeros((n,n))
Csetup_initLAM = S
Csetup_initU = R
Csetup_init = block_diag(Csetup_initX, Csetup_initLAM, Csetup_initU)

Csetup_reg = block_diag(Q,  S, R)

Csetup_end = QN

C = Csetup_init

for i in range(N-1):
    C = block_diag(C, Csetup_reg)

C = block_diag(C, Csetup_end)

#scaling because of OSQP structure
C = 2*C

#setup for ADMM cost
G = np.eye(n+m+k)
Gsetup = G
for i in range(N-1):
    Gsetup = block_diag(Gsetup,G)
    
Gsetup = block_diag(Gsetup, np.zeros((n,n)))

#scaling because of OSQP structure
Gsetup = 2 * Gsetup

#DEGISECEKLER
rho = np.zeros((N*(n+k+m) + n,1))
omega = np.zeros((N*(n+k+m) + n,1))
cc = rho-omega

#LINEAR COST
q = -2 * cc.T @ Gsetup

#QUADRATIC COST
P = C + Gsetup

#DYNAMIC CONSTRAINTS
dyn_init1 = np.eye(n)
dyn_init2 = np.zeros((n, TOT-n))
dyn_init = np.hstack((dyn_init1, dyn_init2))

dyn_reg = np.hstack((A, D, B, -np.eye(n)))
dyn_size = np.size(dyn_reg,1)
dyn_shift = n+m+k

dyn = np.zeros((N*n, TOT))

for i in range(N):
    dyn[4*i:4*i+4, dyn_shift*i:dyn_shift*i + dyn_size  ] = dyn_reg
    
dyn = np.vstack((dyn_init, dyn))

#DEGISEBILIR
x0 = [[1],[1],[1],[1]]
x0 = np.asarray(x0)

eq = x0

for i in range(N):
    eq = np.vstack((eq, -d))
    

# Create an OSQP object
prob = osqp.OSQP()
P = sparse.csr_matrix(P)
dyn = sparse.csr_matrix(dyn) 

# Setup workspace and change alpha parameter
prob.setup(P, q.T, dyn, eq, eq, alpha=1.0, verbose = False)

# Solve problem
res = prob.solve()

sol = res.x












