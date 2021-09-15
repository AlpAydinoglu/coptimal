import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
from scipy.linalg import block_diag

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

#system_parameters
n = 6
m = 6
k = 4
TOT = (n+m+k)
TUT = TOT
dt = 0.1
M = 1000
g = 9.81
mu = 1
h = dt
sample_size = 10000

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

x = np.zeros((TOT, sample_size))
inpt = np.zeros((TOT, sample_size))

for i in range(sample_size):

    if i % 100 == 0:
        print(i)
    
    cons = np.random.uniform(0,5, TOT)
    asd = np.reshape(cons, (TOT,1))
    inpt[:, [i]] = asd
    
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
    
    #Set constraint matrix (Ex + F \lambda + H u + c) and lambda > 0
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
    model.optimize()
    
    sol_v = model.x     
    sol_vs = sol_v[0:TOT]
    sol_vss = np.asarray(sol_vs)
    sol_vss = np.reshape(sol_vss, (TOT,1))
    x[:, [i]] = sol_vss 
    

np.save('datax', x)
np.save('datainpt', inpt)