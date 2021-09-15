import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
from scipy.linalg import block_diag
import timeit
from joblib import Parallel, delayed


def linear_expression(A, b, x, tol=1.e-9):
    """
    Generates a list of Gurobi linear expressions A_i x + b_i (one element per row of A).
    Arguments
    ----------
    A : numpy.ndarray
        Linear term.
    b : numpy.ndarray
        Offest term.
    x : instance of gurobipy.Var
        Variable of the linear expression.
    tol : float
        Maximum absolute value for the elements of A and b to be considered nonzero.
    Returns
    ----------
    exprs : list of gurobipy.LinExpr
        List of linear expressions.
    """

    # linear term (explicitly state that it is a LinExpr since it can be that A[i] = 0)

    return A @ x + b[:,0]

def quadratic_expression(H, x, d, tol=1.e-9):
    """
    Generates a Gurobi quadratic expressions x' H x.
    Arguments
    ----------
    H : numpy.ndarray
        Hessian of the quadratic expression.
    x : instance of gurobipy.Var
        Variable of the linear expression.
    d : constant
    tol : float
        Maximum absolute value for the elements of H to be considered nonzero.
    Returns
    ----------
    expr : gurobipy.LinExpr
        Quadratic expressions.
    """


    return x@H@x - 2*(H @ d)[:,0] @ x

def trial(model):
    
    TOT = 16
    starttime = timeit.default_timer()
    # Optimize model
    model.optimize()
    t_diff = timeit.default_timer() - starttime
    # print(t_diff)
    
    sol_v = model.x     
    sol_vs = sol_v[0:TOT]
    sol_vss = np.asarray(sol_vs)
    sol_vss = np.reshape(sol_vss, (TOT,1))
    
    return sol_vss

def parallel_gurobi(cons, TUT, m, G, E, F, H, n, k, c, M):
        #cons = z[TOT*i:TOT*(i+1)] + omega[TOT*i:TOT*(i+1)]
        
        #starttime = timeit.default_timer()
    
        TOT = TUT    
        
        
    
        # Create a new model
        model = gp.Model()
        model.Params.LogToConsole = 0
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        
        #model.Params.TimeLimit = 1e-1
        
        # Create variables
        delta_v = model.addMVar(TUT, lb=[-gp.GRB.INFINITY]*TUT, vtype=GRB.CONTINUOUS)
        cons_binary = model.addMVar(m, vtype=GRB.BINARY)


        # Set objective
        obj = quadratic_expression(G,delta_v,cons)
        
        model.setObjective(obj, GRB.MINIMIZE)
        
        
        
        
        #Set constraint matrix (Ex + F \lambda + H u + c) and lambda > 0
        Mcons1 = np.hstack((E, F, H))
        Mcons2 = np.hstack( ( np.zeros((m,n)) , np.eye(m) , np.zeros((m,k)) ) )
        # import pdb
        # pdb.set_trace()
        M_stack = np.vstack((Mcons1,Mcons2,-Mcons1,-Mcons2))
        M_binary_stack = np.vstack((np.zeros((2*m,m)), -M*np.eye(m), M*np.eye(m)))
        c_stack = np.hstack((c[:,0], np.zeros(m), -c[:,0] + M*np.ones(m), np.zeros(m)))  
        model.addConstr(M_stack@delta_v + M_binary_stack@cons_binary + c_stack >= 0)


        
        # for j in range(m):
        #     model.addConstr(  Mcons1_exp[j] >= 0)
        #     model.addConstr(Mcons2_exp[j] >= 0)
        #     model.addConstr(  Mcons1_exp[j] <= M*(1 - cons_binary[j]) )
        #     model.addConstr(  Mcons2_exp[j] <= M*( cons_binary[j] ) )
        
        starttime = timeit.default_timer()
        sol_vss = trial(model)
        # print(model.runtime)
        t_diff = timeit.default_timer() - starttime
        # print(t_diff)
        # model.optimize()
        

        
        # sol_v = model.x     
        # sol_vs = sol_v[0:TOT]
        # sol_vss = np.asarray(sol_vs)
        # sol_vss = np.reshape(sol_vss, (TOT,1))
        

        
        return sol_vss


def Projection_function2(z, delta, omega, N, G, A,B,D,d,E,c,F,H):
    
   #system_parameters
    n = 6
    m = 6
    k = 4
    TOT = (n+m+k)
    TUT = TOT
    M = 1000
       
   
    pc = block_diag(1000*np.eye(n), np.eye(m), np.eye(k))
    G = pc
    
    cons = []
    
    for i in range(N):
        cons.append(  z[TOT*i:TOT*(i+1)] + omega[TOT*i:TOT*(i+1)] )
    
    starttime = timeit.default_timer()

    par_sol = Parallel(n_jobs=N, prefer=None)(delayed(  parallel_gurobi )   (tt, TUT, m, G, E, F, H, n, k, c, M)  for tt in cons)
    
    t_diff = timeit.default_timer() - starttime
    # print(t_diff)

    
    for i in range(N):
        delta[TOT*i:TOT*(i+1)] = par_sol[i]

        

        
    return delta, t_diff
