import torch

import os
import sys

import osqp
#from osqpth.osqpth import OSQP
#from qpth.qp import QPFunction

from utils import quaternion as quat
from utils import dirs 

#from lemkelcp import lemkelcp as lcp

import numpy as np
from pyomo.environ import *
from pyomo.mpec import *

import pdb

opt = SolverFactory('path', executable=dirs.lib_path('pathampl'))

def solve_lcp_path(lcp_mat, lcp_vec):
    def addLCVectorConstraint(model, M, q, vars, I):
        def lc_rule(m, i):
            return complements(sum(M[i-1,j-1] * vars[j] for j in I) + q[i-1] >= 0, vars[i] >= 0)
        model.c = Complementarity(I, rule=lc_rule)

    lcp_mat_numpy = lcp_mat.detach().cpu().numpy()
    lcp_vec_numpy = lcp_vec.squeeze(1).detach().cpu().numpy()

    model = ConcreteModel()
    #model.x1 = Var()  
    #model.f1 = Complementarity(expr=
    #complements(model.x1 >= 0,
    #squ(model.x1-1) >= 0))
    DIM = lcp_vec_numpy.size

    model.I = RangeSet(1,DIM)
    model.z = Var(model.I, domain=NonNegativeReals)
    addLCVectorConstraint(model, lcp_mat_numpy, lcp_vec_numpy, model.z, model.I)
    #for i in range(0,DIM):
    #    model.z[i+1] = 0

    results = opt.solve(model)#, warmstart=True)
    #pdb.set_trace()
    if results.solver.status != SolverStatus.ok:
        print('Error in LCP solver')
    
    sol = np.zeros(DIM)
    for i in range(0,DIM):
        sol[i] = value(model.z[i+1])

    return torch.tensor(sol).unsqueeze(1)

def solve_lcp_lemkelcp(lcp_mat, lcp_vec):
    sol, exit_code, msg = lcp.lemkelcp(lcp_mat.detach().cpu().numpy(), lcp_vec.detach().cpu().numpy(), maxIter = 1000)

    assert exit_code == 0, msg
    return torch.tensor(sol).double().unsqueeze(1)

def solve_lcp_osqp(lcp_mat, lcp_vec):
    def torch_to_cvxopt(mat):
        return matrix(mat.detach().cpu().numpy())

    k = lcp_mat.shape[0]

    M = lcp_mat
    q = lcp_vec.squeeze()

    R = torch.cat((-lcp_mat, -torch.eye(k)), dim=0)
    h = torch.cat((lcp_vec, torch.zeros(k, 1)), dim=0)
    
    sol = solve_qp_osqpth(2*M.unsqueeze(0), q.unsqueeze(0), R.unsqueeze(0), h.unsqueeze(0))
    return sol

def solve_lcp_batch(lcp_mats, lcp_vecs, solver=solve_lcp_lemkelcp):
    batch_n = lcp_mats.shape[0]

    for i in range(batch_n):
        if i == 0:
            sols = solver(lcp_mats[i], lcp_vecs[i]).unsqueeze(0)
        else:
            sol = solver(lcp_mats[i], lcp_vecs[i]).unsqueeze(0) 
            sols = torch.cat((sols, sol), dim=0)

    return sols

def solve_qp_osqpth(A, b, R, h, check_spd=True):
    assert A.shape[0] == b.shape[0] == R.shape[0] == h.shape[0]
    batch_n = A.shape[0]
    
    # Just pass all indices to OSQP
    A_numpy = (torch.ones_like(A[0])).detach().cpu().numpy()
    R_numpy = (torch.ones_like(R[0])).detach().cpu().numpy()
    
    solver = OSQP(A_numpy.nonzero(),
                  A_numpy.shape,
                  R_numpy.nonzero(),
                  R_numpy.shape,
                  max_iter=100000,
                  eps_abs=1e-6,
                  eps_rel=1e-6)
    
    # Handle weird OSQPTH batch sizing
    if batch_n == 1:
        A, b, R, h = A.flatten(), b.flatten(), R.flatten(), h.flatten()
    else:
        A = A.reshape(batch_n, -1)
        b = b.reshape(batch_n, -1)
        R = R.reshape(batch_n, -1)
        h = h.reshape(batch_n, -1)

    sol = solver(A, b, R, torch.ones_like(h) * (-float('inf')), h)
    
    sol = sol.reshape(batch_n, -1)

    return sol 

def solve_qp_qpth(A, b, R, h):
    solver = QPFunction(check_Q_spd=False, eps=1e-6)
    sol = solver(A, b.squeeze(1), R, h.squeeze(2), torch.tensor([]), torch.tensor([]))
    return sol