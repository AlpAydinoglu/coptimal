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
    n = 4
    m = 2
    k = 1
    N = 10
    TOT = (n+m+k)
    TUT = TOT
    
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
    n = 4
    m = 2
    k = 1
    TOT = (n+m+k)
    TUT = TOT
    
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
    
    
    pc = block_diag(1000*np.eye(n), 1*np.eye(m), 0*np.eye(k))
    G = pc
    
    rho_scale = 0.0001 #yakinda 1.2, 1 uzakta ise 4,4 fena degil
    admm_iter = 2
    rho = 100
    
    P1 = np.zeros((n,TOT))
    P2 = np.hstack((E, F, H))
    P3 = np.zeros((k,TOT))
    P = np.vstack((P1,P2,P3))
    P = (P + P.T)/2
    q = np.vstack((np.zeros((n,1)), c, np.zeros((k,1))   ))
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
    omega[0] = cons[0]
    omega[1]= cons[1]
    omega[2] = cons[2]
    omega[3] = cons[3]
    #omega = np.reshape(delta, (TOT,))
    #cons = np.ones((16,1))
    
    time_hold = 0
    
    for i in range(admm_iter):
        
        cons_v = omega - delta
        
        #QP step
        z, t_diff_qp = QP_projection(cons, cons_v, rho, G)
        
        time_hold = time_hold + t_diff_qp

        
        #Projection 
        inp = omega + z
        
        # start_time = time.time()
        
        starttime = timeit.default_timer()
        
        delta = u.onecons_qcqp( inp , f)
        
        t_diff_admm = timeit.default_timer() - starttime
        
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
    
    n = 4
    m = 2
    k = 1
    TOT = (n+m+k)
    
    tt_hold = 0
    
    for i in range(N):
        cons = z[TOT*i:TOT*(i+1)] + omega[TOT*i:TOT*(i+1)]
        sol_vss, time_hold = Projection3_one_step(cons, delta[TOT*i:TOT*(i+1)])
        sol_vss = np.reshape(sol_vss, (TOT,1))
        delta[TOT*i:TOT*(i+1)] = sol_vss
        asd = 1
        
        tt_hold = tt_hold + time_hold
    
    t_diff = tt_hold
    
    return delta, t_diff
    
    
    
    
    
    
    
    
    
