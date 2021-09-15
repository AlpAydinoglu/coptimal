import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import timeit

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
    exprs = [gp.LinExpr(sum([A[i,j]*x[j] for j in range(A.shape[1]) if np.abs(A[i,j]) > tol])) for i in range(A.shape[0])]

    # offset term
    exprs = [expr+b[i] if np.abs(b[i]) > tol else expr for i, expr in enumerate(exprs)]

    return exprs

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

    return sum([ (x[i] - d[i] ) * H[i,j] * (x[j] - d[j]) for i, j in np.ndindex(H.shape) if np.abs(H[i,j]) > tol])

def Projection_function2(z, delta, omega, N, G):
    
    z = np.reshape(z, (np.size(z,0),1))
    n=4
    m=2
    k=1
    M = 1000
    TOT = n+m+k
    TUT = TOT
    #system parameters
    g = 9.81
    mp = 0.411
    mc = 0.978
    len_p = 0.6
    len_com = 0.4267
    d1 = 0.35
    d2 = -0.35
    ks= 50
    E = [[-1, len_p, 0, 0], [1, -len_p, 0, 0 ]]
    E = np.asarray(E)
    F = 1/ks * np.eye(2)
    F = np.asarray(F)
    c = [[d1], [-d2]]
    c = np.asarray(c)
    H = np.zeros((2,1))
    
    px = 1000
    plam = 1
    pu = 0
    pc = [[px, 0, 0, 0, 0, 0, 0], [0, px, 0, 0, 0, 0, 0 ], [0, 0, px, 0, 0, 0, 0 ], [0, 0, 0, px, 0 ,0 ,0], [0, 0, 0, 0, plam, 0, 0], [0, 0, 0, 0, 0, plam, 0  ], [0,0,0,0,0,0,0]]
    pc = np.asarray(pc)
    
    #G = pc*G
    G = pc
    
    t_diff = 0
    
    for i in range(N):
        cons = z[TOT*i:TOT*(i+1)] + omega[TOT*i:TOT*(i+1)]
        
        # Create a new model
        model = gp.Model()
        model.Params.LogToConsole = 0
        model.Params.OutputFlag = 0
        
        # Create variables
        delta_v = model.addVars(TUT, lb=[-gp.GRB.INFINITY]*TUT, vtype=GRB.CONTINUOUS)
        cons_binary = model.addVars(m, vtype=GRB.BINARY)
    
        # Set objective
        obj = quadratic_expression(G,delta_v,cons)
        model.setObjective(obj, GRB.MINIMIZE)
        
        #Set constraint matrix (Ex + F \lambda + H u + c)
        Mcons1 = np.hstack((E, F, H))
        Mcons2 = np.hstack( ( np.zeros((m,n)) , np.eye(m) , np.zeros((m,k)) ) )
        
        Mcons1_exp = linear_expression(Mcons1, c, delta_v)
        Mcons2_exp = linear_expression(Mcons2, np.zeros((m,1)), delta_v)
        
        for j in range(m):
            model.addConstr(  Mcons1_exp[j] >= 0)
            model.addConstr(Mcons2_exp[j] >= 0)
            model.addConstr(  Mcons1_exp[j] <= M*(1 - cons_binary[j]) )
            model.addConstr(  Mcons2_exp[j] <= M*( cons_binary[j] ) )
         
        # Optimize model
        starttime = timeit.default_timer()
        model.optimize()
        t_diff_h = timeit.default_timer() - starttime
        t_diff = t_diff + t_diff_h
        
        sol_v = model.x     
        sol_vs = sol_v[0:TOT]
        sol_vss = np.asarray(sol_vs)
        sol_vss = np.reshape(sol_vss, (TOT,1))
        delta[TOT*i:TOT*(i+1)] = sol_vss
        asd = 1
        
    #for cartpole
    delta[0:n] = np.zeros((n,1))
        
    return delta, t_diff
