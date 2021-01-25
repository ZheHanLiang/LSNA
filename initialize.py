import numpy as np
import torch
import time

def initialize_seed(params):
    """
    Initialize random seed
    """
    time_head = time.time() # Record start time
    if params.seed>0: # Generate random number seeds for numpy and torch when a positive seed value is set
        np.random.seed(params.seed) # Random seed of numpy
        torch.manual_seed(params.seed) # Random seed of the CPU part in torch
        if params.cuda: # When use cuda
            torch.cuda.manual_seed(params.seed) # Random seed of the GPU part in torch
    time_tail = time.time() # Record finish time
    print("=====> Seeds have been initialized!\tTime: %.3f"%(time_tail-time_head))

def initialize_feature(params):
    """
    Initialize random node features
    """
    # ## Noisy initial embedding
    # feature_s = (torch.ones(params.node_num, params.init_dim) + torch.randn(params.node_num, params.init_dim) / 10) * params.radius
    # feature_t = (torch.ones(params.node_num, params.init_dim) + torch.randn(params.node_num, params.init_dim) / 10) * params.radius
    ## Initial embedding without noise
    feature_s = (torch.ones(params.node_num, params.init_dim)) * params.radius
    feature_t = (torch.ones(params.node_num, params.init_dim)) * params.radius
    if params.cuda:
        return feature_s.cuda(), feature_t.cuda()
    else:
        return feature_s, feature_t
