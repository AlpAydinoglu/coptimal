import numpy as np

def Evaluate_system(x,dt):

    mu1 = 0.1
    mu2 = 0.1
    mu3 = 0.1
    g = 9.81
    h = 1
    w = 1
    m = 1
    
    rt = np.sqrt(h*h + w*w)
    rt = -rt
    
    sin = np.sin(x[4])
    sin = sin[0]
    cos = np.cos(x[4])
    cos = cos[0]
    
    z = w*sin - h*cos
    
    A = [[1,dt,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,1,dt,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,dt,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,dt,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,dt],[0,0,0,0,0,0,0,0,0,1]]
    A = np.asarray(A)
    
    B = [[0,0,-dt*dt*cos,dt*dt*sin], [0,0,-dt*cos, dt*sin], [0,0,dt*dt*sin, dt*dt*cos],[0,0,dt*sin,dt*cos],[0,0,-dt*dt*x[6], dt*dt*x[8]], [0,0,-dt*x[6], dt*x[8]],[dt*dt,0,0,0],[dt,0,0,0],[0,dt*dt,0,0],[0,dt,0,0]]
    B = np.asarray(B)
    
    D = [[0, dt*dt*sin, -dt*dt*sin, 0, -cos*dt*dt, dt*dt*cos,0,dt*dt,-dt*dt,0],[0, dt*sin, -dt*sin, 0, -cos*dt, dt*cos,0,dt,-dt,0],   [0,dt*dt*cos,-dt*dt*cos,0,dt*dt*sin,-dt*dt*sin,0,0,0,dt*dt], [0,dt*cos,-dt*cos,0,dt*dt*sin,-dt*sin,0,0,0,dt]   ,   [0, -dt*dt*h, dt*dt*h,0,dt*dt*w,-dt*dt*w,0,-dt*dt*cos*w-dt*dt*sin*h,dt*dt*cos*w+dt*dt*sin*h,dt*dt*sin*w-h*cos*dt*dt]  , [0, -dt*h, dt*h,0,dt*w,-dt*w,0,-dt*cos*w-dt*sin*h,dt*cos*w+dt*sin*h,dt*sin*w-h*cos*dt] , [0, -dt*dt, dt*dt,0,0,0,0,0,0,0]    ,   [0, -dt, dt,0,0,0,0,0,0,0]    ,   [0,0,0,0,-dt*dt,dt*dt,0,0,0,0]   ,   [0,0,0,0,-dt,dt,0,0,0,0]]
    D = np.asarray(D)
    
    d = [[0],[0],[-dt*dt*m*g],[-dt*m*g],[0],[0],[0],[0],[0],[0]]
    d = np.asarray(d)
    
    E = [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,-1,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,-1],[0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0],np.negative([0,1,0,0,0,rt,0,0,0,0]),np.negative([0,-1,0,0,0,-rt,0,0,0,0]),[0,0,1,dt,-h*sin+w*cos+z,z*dt,0,0,0,0]]
    E = np.asarray(E)
    
    c = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[-h*cos-w*sin-dt*dt*m*g]]
    c = np.asarray(c)
    
    F = [[0,-1,-1,0,0,0,0,0,0,0],[1,dt,-dt,0,0,0,0,0,0,0],[1,-dt,dt,0,0,0,0,0,0,0],[0,0,0,0,-1,-1,0,0,0,0],[0,0,0,1,dt,-dt,0,0,0,0],[0,0,0,1,-dt,dt,0,0,0,0],[0,0,0,0,0,0,0,-1,-1,mu3],np.negative([0,dt*sin-rt*dt*h,-dt*sin+rt*dt*h,0,-dt*cos+rt*dt*w,dt*cos-dt*rt*w,-1,dt-rt*dt*cos*w-rt*dt*sin*h,-dt+rt*dt*cos*w+rt*dt*sin*h,sin*w*rt*dt - h*cos*rt*dt]),np.negative([0,-dt*sin+rt*dt*h,dt*sin-rt*dt*h,0,dt*cos-rt*dt*w,-dt*cos+rt*dt*w,-1,-dt+rt*dt*cos*w+rt*dt*sin*h,dt-rt*dt*cos*w-rt*dt*sin*h,-sin*w*rt*dt + h*cos*rt*dt]),[0,dt*dt*cos-z*dt*dt*h,-dt*dt*cos+z*dt*dt*h,0,dt*dt*sin+z*dt*dt*w,-dt*dt*sin-dt*dt*w*z,0,-z*dt*dt*cos*w-z*dt*dt*sin*h,z*dt*dt*cos*w+z*dt*dt*sin*h,dt*dt+z*dt*dt*sin*w-z*h*cos*dt*dt]]
    F = np.asarray(F)
    
    x5 = x[6]
    x5 = x5[0]
    
    x7 = x[8]
    x7 = x7[0]
    
    H = [[0,0,mu1,0],[-dt,0,0,0],[dt,0,0,0],[0,0,0,mu2],[0,-dt,0,0],[0,dt,0,0],[0,0,0,0],np.negative([0,0,-dt*cos-rt*dt*x5,dt*sin+dt*rt*x7]),np.negative([0,0,dt*cos+rt*dt*x5,-dt*sin-dt*rt*x7]),[0,0,dt*dt*sin-dt*dt*x5*z,dt*dt*cos+dt*dt*x7*z ]]
    H = np.asarray(H)
    
    return A,B,D,d,E,c,F,H

