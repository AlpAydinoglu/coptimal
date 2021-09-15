# flake8: noqa

import os.path as op
import os
ROOT_DIR = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
def root_path(*path: str) -> str:
    return op.join(ROOT_DIR, *path)

EXP_KEY = 'CONTACTNETS_EXPERIMENT'
EXP_DIR = 'experiments'
if os.getenv(EXP_KEY) is not None:
    OUT_DIR = root_path(EXP_DIR, os.getenv(EXP_KEY), 'out')
    RESULTS_DIR = root_path(EXP_DIR, os.getenv(EXP_KEY), 'results')
else:
    OUT_DIR = root_path('out')
    RESULTS_DIR = root_path('results')

DATA_DIR = root_path('data')
EXPERIMENTS_DIR = root_path(EXP_DIR)
LIB_DIR = root_path('lib')
LOGS_DIR = root_path('logs')
PROCESSING_DIR = root_path('contactnets', 'utils', 'processing')

def out_path(*path: str) -> str:
    return op.join(OUT_DIR, *path)

def data_path(*path: str) -> str:
    return op.join(DATA_DIR, *path)

def results_path(*path: str) -> str:
    return op.join(RESULTS_DIR, *path)

def experiments_path(*path: str) -> str:
    return op.join(EXPERIMENTS_DIR, *path)

def lib_path(*path: str) -> str:
    return op.join(LIB_DIR, *path)

def logs_path(*path: str) -> str:
    return op.join(LOGS_DIR, *path)

def processing_path(*path: str):
    return op.join(PROCESSING_DIR, *path)
