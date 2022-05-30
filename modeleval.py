import numpy as np
import torch
import torch.nn as nn
model = nn.Sequential(
        nn.Linear(9,64),
        nn.ReLU(),
        nn.Linear(64,64),
        nn.ReLU(),
        nn.Linear(64,10),
        )
model.load_state_dict(torch.load("model/model2000.pth"))
model.eval()
res=model(torch.tensor([1,-1,-1,0,1,0,0,0,0],dtype=torch.float32))
v=2*torch.sigmoid(res[0])-1
p=torch.softmax(res[1:],0)
print(v)
print(p.detach().numpy())