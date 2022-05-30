from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(comm.Get_size())
buffer1 = np.empty(4,dtype="double").reshape((2,2))
buffer2 = np.empty(4,dtype="double").reshape((2,2))
req0=comm.Irecv([buffer1,MPI.DOUBLE],source= 0, tag=0)
req1=comm.Irecv([buffer2,MPI.DOUBLE],source= 0, tag=1)
# print(MPI.Request.Waitsome([req0,req1]))
while True:
    i=MPI.Request.Waitany([req0,req1])
    print(i,flush=True)
    if(0==i):
        print(buffer1,flush=True)
        req0=comm.Irecv([buffer1,MPI.DOUBLE],source= 0, tag=0)
    else:
        print(buffer2,flush=True)
        req1=comm.Irecv([buffer2,MPI.DOUBLE],source= 0, tag=1)
print(buffer2)