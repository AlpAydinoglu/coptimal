import osqp
import numpy as np
from scipy import sparse
from scipy.linalg import block_diag
from scipy import linalg


#system and MPC dimensions
n = 4
m = 2
k = 1
N = 15
TOT = N*(n+m+k) + n

#system parameters
g = 9.81
mp = 0.411
mc = 0.978
len_p = 0.6
len_com = 0.4267
d1 = 0.35
d2 = -0.35
ks= 50

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












