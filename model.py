import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

def adjacency_normalize(A, symmetric=True):
    """
    Normalize the adjacency matrix according to the formula of GCN
    """
    # A = A+I
    A = A + torch.eye(A.size(0)).cuda() # When the diagonal of the adjacency matrix is 0
    # Calculate the degree of all nodes
    d = A.sum(1)
    if symmetric:
        #D = D^-1/2
        D = torch.diag(torch.pow(d , -0.5))
        return D.mm(A).mm(D)
    else :
        # D=D^-1
        D =torch.diag(torch.pow(d,-1))
        return D.mm(A)

class GCN(nn.Module):
    '''
    basic GCN model: Z = AXW
    '''
    def __init__(self, dim_in, dim_out):
        super(GCN,self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_in,bias=False)
        self.fc2 = nn.Linear(dim_in, dim_in//2,bias=False)
        self.fc3 = nn.Linear(dim_in//2, dim_out,bias=False)

    def forward(self, A_s, X_s, A_t, X_t):
        '''
        Calculate three-layer GCN
        '''
        X_s = F.relu(self.fc1(A_s.mm(X_s)))
        X_s = F.relu(self.fc2(A_s.mm(X_s)))
        X_t = F.relu(self.fc1(A_t.mm(X_t)))
        X_t = F.relu(self.fc2(A_t.mm(X_t)))
        return self.fc3(A_s.mm(X_s)), self.fc3(A_t.mm(X_t))

## Cross-graph convolution
class CNEM(nn.Module):
    '''
    CNEM model
    '''
    def __init__(self, dim_in, dim_out):
        super(CNEM,self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_in//2,bias=False) # (Front) first GCN layer
        self.fc2 = nn.Linear(dim_in//2, dim_in//2,bias=False) # (Front) second GCN layer
        self.fc3 = nn.Linear(dim_in//2, dim_in//2,bias=False) # Affinity Metric computing layer
        self.fc4 = nn.Linear(dim_in//2, dim_out,bias=False) # Cross-graph GCN layer

    def forward(self, A_s, X_s, A_t, X_t):
        X_s = F.relu(self.fc1(A_s.mm(X_s)))
        X_s = F.relu(self.fc2(A_s.mm(X_s)))
        X_t = F.relu(self.fc1(A_t.mm(X_t)))
        X_t = F.relu(self.fc2(A_t.mm(X_t)))
        S = (self.fc3(X_s).mm(X_t.t())).exp()
        S_r = F.softmax(S,dim =1)
        S_c = F.softmax(S,dim =0)
        S_s = S.mm(S.t())
        S_t = S.t().mm(S)
        X_s_new = (X_s+S_r.mm(X_t))/2
        X_t_new = (X_t+S_c.t().mm(X_s))/2
        S_s = adjacency_normalize(S_s)
        S_t = adjacency_normalize(S_t)
        return self.fc4(S_s.mm(X_s_new)), self.fc4(S_t.mm(X_t_new)), S

def build_model(params):
    '''
    Build model
    '''
    if params.model=='GCN':
        model = GCN(params.init_dim, params.emb_dim)
    else:
        model = CNEM(params.init_dim, params.emb_dim)
    # Initialization cuda
    if params.cuda:
        model.cuda()
    return model

def save_best_model(params, model):
    """
    Save the best model
    """
    if not os.path.exists(params.best_model_path): # If the best model path does not exist, creat it
        os.makedirs(params.best_model_path)
    file_name = params.dataset + str(params.node_num) + '_best_model.pth'
    path = os.path.join(params.best_model_path, file_name)
    print("=====> Saving the best model ...")
    torch.save(model, path)

def reload_best_model(params):
    """
    Reload the best model
    """
    file_name = params.dataset + str(params.node_num) + '_best_model.pth'
    path = os.path.join(params.best_model_path, file_name)
    print("=====> Reloading the best model ...")
    assert os.path.isfile(path) # Check if the best model exists
    model = torch.load(path)
    return model