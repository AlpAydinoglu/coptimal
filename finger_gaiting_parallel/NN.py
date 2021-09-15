import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
from scipy.linalg import block_diag
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io


x = np.load('datax.npy')
inpt = np.load('datainpt.npy')

#set the seed for repeatability
torch.manual_seed(2)

X = torch.from_numpy(x.T).float()
Y = torch.from_numpy(inpt.T).float()

num_samples = X.shape[0]
num_train_samples = int(np.ceil(num_samples*0.8))
num_test_samples = num_samples - num_train_samples
Xtrain,Xtest = torch.split(X,[num_train_samples,num_test_samples])
Ytrain,Ytest = torch.split(Y,[num_train_samples,num_test_samples])

trainset = torch.utils.data.TensorDataset(torch.Tensor(Xtrain),torch.Tensor(Ytrain))

##neural network structure
sz = 16 #number of neurons on each 
szl = 100
## neural net
net = nn.Sequential(
        nn.Linear(sz,szl,bias=True),
        nn.ReLU(),
        nn.Linear(szl,szl,bias=True),
        nn.ReLU(),
        nn.Linear(szl,sz,bias=True),
        )


## optimization setup
criterion = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=1e-2) 
verbose = False
train_batch_size = 100
trainloader = torch.utils.data.DataLoader(trainset, batch_size= train_batch_size, shuffle=True, num_workers=0)
epoch = 8000
net.train()

# train
for t in range(epoch):
    for i, (X,Y) in enumerate(trainloader):
        batch_size = X.shape[0]
        out = net(X)
        loss = nn.MSELoss()(out,Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if(np.mod(t,100)==0):
        print('epoch: ',t,'MSE loss: ',loss.item())