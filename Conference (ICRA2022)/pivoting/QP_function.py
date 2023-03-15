import osqp
import numpy as np
from scipy import sparse
from scipy.linalg import block_diag
from scipy import linalg
from lemke_lcp import*
import timeit

def QP_function(x0, delta, omega,N,G, A,B,D,d,E,c,F,H):
    
    #system_parameters
    n = 10
    m = 10
    k = 4
    TOT = N*(n+m+k) + n

       
    
    
    #cost matrices
    #Q = [[5000, 0, 0, 0, 0, 0], [0, 10, 0, 0, 0, 0], [0, 0, 10, 0, 0 ,0], [0, 0, 0, 10, 0, 0], [0, 0, 0, 0, 10, 0], [0, 0, 0, 0, 0, 10]]
    #Q = np.asarray(Q)
    Q = np.eye(n)
    Q[4,4] = 100 #115
    Q[2,2] = 100
    Q[0,0] = 100
    Q[6,6] = 50
    Q[8,8] = 50
    
    Q[5,5] = 11
    Q[3,3] = 9
    Q[1,1] = 11
    
    #Q = Q / 10
    
    S = np.zeros((m,m))
    #R = np.eye(k)
    R = 0.01*np.eye(k)
    #QN = linalg.solve_discrete_are(A, B, Q, R)
    QN = Q

    #setup for quadratic cost
    Csetup_initX = np.zeros((n,n))
    Csetup_initLAM = S
    Csetup_initU = R
    Csetup_init = block_diag(Csetup_initX, Csetup_initLAM, Csetup_initU)
    
    Csetup_reg = block_diag(Q,  S, R)
    
    Csetup_end = QN
    
    C = Csetup_init
    
    qf_init = np.zeros((n+m+k,1))
    x_reg = [[0],[0],[np.sqrt(2)],[0],[np.pi/4],[0],[0.9],[0],[0.9],[0]]
    x_reg = np.asarray(x_reg)
    qf_reg = np.vstack(( x_reg, np.zeros((m+k,1))    ))
    qf = qf_init
    
    for i in range(N-1):
        C = block_diag(C, Csetup_reg)
        qf = np.vstack((qf, qf_reg    ))
    
    C = block_diag(C, Csetup_end)
    qf = np.vstack((qf, x_reg))
    
    #scaling because of OSQP structure
    C = 2*C
    
    qf_add = - qf.T @ C
    
    #add the position it should converge
    
    
    #setup for ADMM cost
    #G = np.eye(n+m+k)
    
    #for cartpole
    #asd = 0.1 * np.eye(n+m+k)
    #asd[n+m+k-1,n+m+k-1] = 0
    Gsetup = G #Burada Gsetup = asd idi
    
    for i in range(N-1):
        Gsetup = block_diag(Gsetup,G)
        
    Gsetup = block_diag(Gsetup, np.zeros((n,n)))
    
    #scaling because of OSQP structure
    Gsetup = 2 * Gsetup
    
    #DEGISECEKLER
    #rho = np.zeros((N*(n+k+m) + n,1))
    #omega = np.zeros((N*(n+k+m) + n,1))
    cc = delta-omega
    #cc[0:4] = np.zeros((4,1))
    
    
    #LINEAR COST (DIVIDE By 2 BECAUSE MULTIPLIED EARLIER)
    q = - cc.T @ Gsetup + qf_add
    
    #QUADRATIC COST
    P = C + Gsetup
    
    #DYNAMIC CONSTRAINTS
    dyn_init1 = np.eye(n)
    dyn_init2 = np.zeros((n, TOT-n))
    
    #for cartpole (\lambda_0 = sth)
    #dyn_init3 = np.zeros((m,n))
    #dyn_init4 = np.eye(m)
    #dyn_init5 = np.zeros((m, TOT-n-m))
    #dyn_init_extra = np.hstack((dyn_init3, dyn_init4, dyn_init5))
    
    dyn_init = np.hstack((dyn_init1, dyn_init2))
    
    
    #for cartpole
    #dyn_init = np.vstack((dyn_init, dyn_init_extra))
    
    dyn_reg = np.hstack((A, D, B, -np.eye(n)))
    dyn_size = np.size(dyn_reg,1)
    dyn_shift = n+m+k
    
    dyn = np.zeros((N*n, TOT))
    
    for i in range(N):
        dyn[n*i:n*i+n, dyn_shift*i:dyn_shift*i + dyn_size  ] = dyn_reg
        
    dyn = np.vstack((dyn_init, dyn))
    
    #DEGISEBILIR
    #x0 = [[1],[1],[1],[1]]
    #x0 = np.asarray(x0)
    
    eq = x0
    
    #for the cartpole (changed)
    #qs = E @ x0 + c
    #sol_lcp = lemkelcp(F,qs)
    #ke = np.reshape(sol_lcp[0], (m,1))
    #ke = np.zeros((m,1))
    #eq = np.vstack((eq, ke))

    
    for i in range(N):
        eq = np.vstack((eq, -d))
        
    
    # #inequality constraints (add inequality for all element_jj)
    # jj=15  #u_3 >= 0
    # element = np.zeros((N,TOT))
    # for i in range(N):
    #     element[i, (jj - 1) + (i)*(n+m+k)] = 1
    # dyn = np.vstack((dyn, element))
    # l = np.vstack((eq, np.zeros((N,1))  ))
    # u = np.vstack((eq, 10000*np.ones((N,1))  )) #INFINITY
    
    # jj=16 #u_4 >= 0
    # element = np.zeros((N,TOT))
    # for i in range(N):
    #     element[i, (jj - 1) + (i)*(n+m+k)] = 1
    # dyn = np.vstack((dyn, element))
    # l = np.vstack((l, np.zeros((N,1))  ))
    # u = np.vstack((u, 10000*np.ones((N,1))  )) #INFINITY
    
    # jj=3 #1 <= x_3 <= 3
    # element = np.zeros((N-1,TOT))
    # for i in range(N):
    #     if i > 0:
    #         element[i-1, (jj - 1) + (i)*(n+m+k)] = 1
    # dyn = np.vstack((dyn, element))
    # l = np.vstack((l, 1*np.ones((N-1,1))  ))
    # u = np.vstack((u, 3*np.ones((N-1,1))  )) #INFINITY
    
    # jj=5 #3 <= x_5 <= 5
    # element = np.zeros((N-1,TOT))
    # for i in range(N):
    #     if i > 0:
    #         element[i-1, (jj - 1) + (i)*(n+m+k)] = 1
    # dyn = np.vstack((dyn, element))
    # l = np.vstack((l, 3*np.ones((N-1,1))  ))
    # u = np.vstack((u, 5*np.ones((N-1,1))  )) #INFINITY
    
    # Create an OSQP object
    prob = osqp.OSQP()
    sP = sparse.csr_matrix(P)
    sdyn = sparse.csr_matrix(dyn) 
    
    # Setup workspace and change alpha parameter
    prob.setup(P = sP, q = q.T, A = sdyn, l = eq, u = eq, verbose = False) #time_limit = 0.1)
    
    starttime = timeit.default_timer()
    
    # Solve problem
    res = prob.solve()
    
    t_diff_qp = timeit.default_timer() - starttime
    
    sol = res.x
    
    return sol, t_diff_qp













