import copy
from time import sleep
from threading import Thread,Lock
from mpi4py import MPI
import numpy as np
import torch
import torch.nn as nn
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

class models():
    def __init__(self):
        self.model = nn.Sequential(
                nn.Linear(9,64),
                nn.ReLU(),
                nn.Linear(64,64),
                nn.ReLU(),
                nn.Linear(64,10),
                )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        self.bestmodel=copy.deepcopy(self.model)
        self.mu=Lock()

def listenget():
    while True:
        buffer0 = np.empty(16*9,dtype="float32").reshape((16,9))
        movedata = np.empty(16,dtype="int")
        comm.Recv([buffer0,MPI.FLOAT],source= 0, tag=0)
        comm.Recv([movedata,MPI.INT64_T],source= 0, tag=0)
        # print(movedata,flush=True)
        # print("buffer0",buffer0,flush=True)
        mask=movedata==0
        np.nonzero(movedata)
        with torch.no_grad():
            m.mu.acquire()
            senddata0=m.bestmodel(torch.tensor(buffer0[mask])).numpy()
            senddata1=m.bestmodel(torch.tensor(buffer0[~mask])).numpy()
            # print(senddata0[0:4],flush=True)
            # print(senddata[0:4],flush=True)
            m.mu.release()
            senddata = np.empty(16*10,dtype="float32").reshape((16,10))
            senddata[mask]=senddata0
            senddata[~mask]=senddata1
            comm.Send([senddata,MPI.FLOAT],dest= 0, tag=0)
def loss(model,X,y):
    res=model(torch.Tensor(X))
    mse=nn.MSELoss()(2*nn.Sigmoid()(res[:,0])-1, torch.Tensor(y[:,0]))
    penalty=nn.CrossEntropyLoss()(res[:,1:],torch.Tensor(y[:,1:]))
    return mse+penalty


def listenupdate():
    iternum=0
    while True:
        X = np.empty(512*9,dtype="float32").reshape((512,9))
        y = np.empty(512*10,dtype="float32").reshape((512,10))
        comm.Recv([X,MPI.FLOAT],source= 0, tag=1)
        comm.Recv([y,MPI.FLOAT],source= 0, tag=1)
        iternum+=1
        print("X:",X[0:3],flush=True)
        print("y:",y,flush=True)
        l=loss(m.model,X,y)
        m.mu.acquire()
        m.optimizer.zero_grad()
        l.backward()
        m.optimizer.step()
        print(l,flush=True)
        if(iternum%50==0):
            torch.save(m.model.state_dict(), "model/model"+str(iternum)+".pth")
            m.bestmodel=copy.deepcopy(m.model)
        m.mu.release()

m=models()
# create two new threads
t1 = Thread(target=listenget)
t2 = Thread(target=listenupdate)

# start the threads
t1.start()
t2.start()

# wait for the threads to complete
t1.join()
t2.join()

