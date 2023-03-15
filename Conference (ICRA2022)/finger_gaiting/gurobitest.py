#!/usr/bin/env python3.7

# Copyright 2021, Gurobi Optimization, LLC

# This example formulates and solves the following simple MIP model
# using the matrix API:
#  maximize
#        x +   y + 2 z
#  subject to
#        x + 2 y + 3 z <= 4
#        x +   y       >= 1
#        x, y, z binary

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

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

    return sum([ (x[i]-d[i] ) * H[i,j] * (x[j] - d[j]) for i, j in np.ndindex(H.shape) if np.abs(H[i,j]) > tol])




try:
    
    n = 4
    m = 2
    k = 1
    TUT = n+m+k
    G = np.eye(TUT)
    cons = 5*np.ones((TUT,1))
    M = 1000 #big-M variable
    
    d1 = 0.35
    d2 = -0.35
    ks= 50
    len_p = 0.6
    E = [[-1, len_p, 0, 0], [1, -len_p, 0, 0 ]]
    E = np.asarray(E)
    F = 1/ks * np.eye(2)
    F = np.asarray(F)
    c = [[d1], [-d2]]
    c = np.asarray(c)
    H = np.zeros((2,1))
    
    # Create a new model
    model = gp.Model()
    model.Params.LogToConsole = 0
    model.Params.OutputFlag = 0
    
    # Create variables
    delta = model.addVars(TUT, vtype=GRB.CONTINUOUS)
    cons_binary = model.addVars(m, vtype=GRB.BINARY)

    #obj = sum([( delta[i] - cons[i]) * G[i,j] * (delta[j] - cons[j]) for i, j in np.ndindex(G.shape) if np.abs(G[i,j]) > 1.e-9])
    obj = quadratic_expression(G,delta,cons)
    
    
    # Set objective
    model.setObjective(obj, GRB.MINIMIZE)
    
    #Set constraint matrix (Ex + F \lambda + H u + c)
    Mcons1 = np.hstack((E, F, H))
    Mcons2 = np.hstack( ( np.zeros((m,n)) , np.eye(m) , np.zeros((m,k)) ) )
    
    Mcons1_exp = linear_expression(Mcons1, c, delta)
    Mcons2_exp = linear_expression(Mcons2, np.zeros((m,1)), delta)
    
    for i in range(m):
        model.addConstr(  Mcons1_exp[i] >= 0)
        model.addConstr(Mcons2_exp[i] >= 0)
        model.addConstr(  Mcons1_exp[i] <= M*(1 - cons_binary[i]) )
        model.addConstr(  Mcons2_exp[i] <= M*( cons_binary[i] ) )
    
    # Build (sparse) constraint matrix
    val = np.array([1.0, 2.0, 3.0, -1.0, -1.0])
    row = np.array([0, 0, 0, 1, 1])
    col = np.array([0, 1, 2, 0, 1])

    A = sp.csr_matrix((val, (row, col)), shape=(2, 3))

    # Build rhs vector
    #rhs = np.array([4.0, -1.0])

    # Add constraints
    #model.addConstr(A @ x <= rhs, name="c")

    # Optimize model
    model.optimize()

    #print(x.X)
    print('Obj: %g' % model.objVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')
    
