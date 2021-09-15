#Projection function 3

import osqp
import numpy as np
from scipy import sparse
from scipy.linalg import block_diag
from scipy import linalg
from numpy import linalg as LA
import qcqp
from qcqp import utilities as u
import time
import timeit



def QP_projection(cons, cons_v, rho, G):
    #system_parameters
    n = 6
    m = 6
    k = 4
    TOT = (n+m+k)
    TUT = TOT
    dt = 0.1   
    g = 9.81
    mu = 1
    h = dt
    E = [[0, 0, 0, 0, 0, 0], [0, 1, 0, -1, 0, 0], [0, -1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, -1], [0, -1, 0, 0, 0, 1]]
    E = np.asarray(E)
    F = [[0, -1, -1, 0, 0, 0], [1, 2*h, -2*h, 0, h, -h], [1, -2*h, 2*h, 0, -h, h], [0, 0, 0, 0, -1,-1], [0, h, -h, 1, 2*h, -2*h], [0, -h, h, 1, -2*h, 2*h]]
    F = np.asarray(F)
    c = [[0],[-h*g], [h*g], [0], [-h*g], [h*g]]
    c = np.asarray(c)
    H = [[0, 0, mu, 0], [-h, 0, 0, 0], [h, 0, 0, 0], [0, 0, 0, mu], [0, -h, 0, 0], [0, h, 0, 0]]
    H = np.asarray(H)
    
    P = G + rho * np.eye(TOT)
    P = 2 * P
    q = - 2 * cons.T @ G - 2 * cons_v.T
    Mcons1 = np.hstack((E, F, H))
    Mcons2 = np.hstack( ( np.zeros((m,n)) , np.eye(m) , np.zeros((m,k)) ) )
    dyn = np.vstack((Mcons1,Mcons2))
    l = np.zeros((2*m,1))
    u = 1000*np.ones((2*m,1))
    
    # Create an OSQP object
    prob = osqp.OSQP()
    sP = sparse.csr_matrix(P)
    sdyn = sparse.csr_matrix(dyn) 
    
    # Setup workspace and change alpha parameter
    prob.setup(P = sP, q = q.T, A = sdyn, l = l, u = u, verbose = False)
    
    starttime = timeit.default_timer()
    
    # Solve problem
    res = prob.solve()
    
    t_diff_qp = timeit.default_timer() - starttime
    
    sol = res.x
    
    return sol, t_diff_qp


def Projection3_one_step(cons,delta):
    
    #system_parameters
    n = 6
    m = 6
    k = 4
    TOT = (n+m+k)
    TUT = TOT
    dt = 0.1   
    g = 9.81
    mu = 1
    h = dt
    E = [[0, 0, 0, 0, 0, 0], [0, 1, 0, -1, 0, 0], [0, -1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, -1], [0, -1, 0, 0, 0, 1]]
    E = np.asarray(E)
    F = [[0, -1, -1, 0, 0, 0], [1, 2*h, -2*h, 0, h, -h], [1, -2*h, 2*h, 0, -h, h], [0, 0, 0, 0, -1,-1], [0, h, -h, 1, 2*h, -2*h], [0, -h, h, 1, -2*h, 2*h]]
    F = np.asarray(F)
    c = [[0],[-h*g], [h*g], [0], [-h*g], [h*g]]
    c = np.asarray(c)
    H = [[0, 0, mu, 0], [-h, 0, 0, 0], [h, 0, 0, 0], [0, 0, 0, mu], [0, -h, 0, 0], [0, h, 0, 0]]
    H = np.asarray(H)
    
    
    pc = block_diag(1000*np.eye(n), np.eye(m), np.eye(k))
    G = pc
    
    rho_scale = 1.2 #0.1, 1.5 calisiyor
    admm_iter = 20
    rho = 0.1
    
    # P1 = np.zeros((n,TOT))
    # P2 = np.hstack((E, F, H))
    # P3 = np.zeros((k,TOT))
    # P = np.vstack((P1,P2,P3))
    # P = (P + P.T)/2
    # q = np.vstack((np.zeros((n,1)), c, np.zeros((k,1))   ))
    # r = 0
    
    # f = u.QuadraticFunction(sparse.csr_matrix(P), sparse.csc_matrix(q), r, '==')
    
    P1 = np.hstack((F,H))
    P2 = np.zeros((k,TOT-n))
    P = np.vstack((P1,P2))
    P = (P + P.T)/2
    q = np.vstack(( c + E @ cons[0:n], np.zeros((k,1))   ))
    r = 0
    
    f = u.QuadraticFunction(sparse.csr_matrix(P), sparse.csc_matrix(q), r, '==')
    
    #x = u.onecons_qcqp(np.zeros((16,)), f)
    
    
    #lmb = np.reshape(   lmb, (TOT,1))
    
    
    #cons = np.ones((16,1))  
    
    #cons_v = np.zeros((16,1)) 
    
    
    #sol = QP_projection(cons, cons_v, rho, G)
    #print(sol)
    
    #delta = delta
    #delta = np.reshape(delta, (TOT,))
    
    delta = np.zeros((TOT,))
    omega = np.zeros((TOT,))
    #cons = np.ones((16,1))
    
    time_hold = 0

    
    for i in range(admm_iter):
        
        cons_v = omega - delta
        
        #QP step
        z, t_diff_qp  = QP_projection(cons, cons_v, rho, G)
        
        asd2 = cons[0:n]
        asd2 = np.reshape(asd2, (n,))
        z[0:n] = asd2
        
        time_hold = time_hold + t_diff_qp
        
        #Projection 
        inp = omega + z
        
        # start_time = time.time()
                    
        starttime = timeit.default_timer()
        
        inp_new = inp[n:TOT]
        
        delta_new = u.onecons_qcqp( inp_new , f)
        delta[0:n] = z[0:n]
        delta[n:TOT] = delta_new
    
        t_diff_admm = timeit.default_timer() - starttime
        
        if i < admm_iter:
            time_hold = time_hold + t_diff_admm
        
        # asd = time.time() - start_time
        # format_float = "{:.5f}".format(asd)
        # print(format_float)
        
        #UPDATE STEP
        omega = omega + z - delta
        rho = rho * rho_scale
        omega = omega / rho_scale
        
    return z, time_hold


def Projection_function3(z, delta, omega, N):
    
    n = 6
    m = 6
    k = 4
    TOT = (n+m+k)
    
    tt_hold = 0
    
    for i in range(N):
        cons = z[TOT*i:TOT*(i+1)] + omega[TOT*i:TOT*(i+1)]
        sol_vss, time_hold = Projection3_one_step(cons, delta[TOT*i:TOT*(i+1)])
        sol_vss = np.reshape(sol_vss, (TOT,1))
        delta[TOT*i:TOT*(i+1)] = sol_vss
        
        tt_hold = tt_hold + time_hold
    
    t_diff = tt_hold
    
    return delta, t_diff
    
    
    
    
    
    
    
    
    
