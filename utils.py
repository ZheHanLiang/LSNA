import os
import numpy as np
import networkx as nx
import heapq as hq
import torch
import time
import sklearn.metrics.pairwise

def adjacency_normalize(A , symmetric=True):
    """
    Normalize the adjacency matrix according to the formula of GCN
    """
    # A = A+I
    A = A + torch.eye(A.size(0)) # When the diagonal of the adjacency matrix is 0
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

def read_graph(params):
    """
    Read graph
    """
    time_head = time.time() # Record starting time
    source_graph_path = params.data_path + params.dataset + "_source_edges.txt"
    target_graph_path = params.data_path + params.dataset + "_target_edges.txt"
    g_source = nx.read_edgelist(source_graph_path, nodetype=int)
    g_target = nx.read_edgelist(target_graph_path, nodetype=int)
    time_tail = time.time() # Record finish time
    print("=====> Graphs have been read!\tTime: %.3f"%(time_tail-time_head))

    return g_source, g_target

def adjacency_matrix_normalize(params, G_source, G_target):
    """
    Read adjacency matrix of graph
    """
    time_head = time.time() # Record starting time
    A_source = nx.adjacency_matrix(G_source)
    A_target = nx.adjacency_matrix(G_target)
    time_tail = time.time() # Record finish time
    print("=====> Adjacency matrixs have been read!\tTime: %.3f"%(time_tail-time_head))
    time_head = time.time() # Record starting time
    # A_source_norm = adjacency_normalize(torch.FloatTensor(A_source),True)
    # A_target_norm = adjacency_normalize(torch.FloatTensor(A_target),True)
    # # When meet error: 'sparse matrix length is ambiguous' use this code
    A_source_norm = adjacency_normalize(torch.FloatTensor(A_source.todense()),True)
    A_target_norm = adjacency_normalize(torch.FloatTensor(A_target.todense()),True)
    time_tail = time.time() # Record finish time
    print("=====> Adjacency matrixs have been normalized!\tTime: %.3f"%(time_tail-time_head))
    if params.cuda:
        return A_source_norm.cuda(), A_target_norm.cuda()
    else:
        return A_source_norm, A_target_norm

def get_metric_matrix(matrix1, matrix2, method):
    """
    Calculate the metric matrix
    """
    assert method in ['Euclid', 'cosine'], "Unkown operation!" # Assert the reasonability of measurement method
    if method=='cosine':
        metric_matrix = sklearn.metrics.pairwise.cosine_similarity(matrix1, matrix2)
        ## Calculation method without library
        # dot = matrix1.dot(matrix2.transpose())
        # matrix1_norm = np.sqrt(np.multiply(matrix1, matrix1).sum(axis=1))
        # matrix2_norm = np.sqrt(np.multiply(matrix2, matrix2).sum(axis=1))
        # metric_matrix = np.divide(dot, matrix1_norm * matrix2_norm.transpose())
    else:
        metric_matrix = sklearn.metrics.pairwise.euclidean_distances(matrix1, matrix2)
        ## Calculation method without library
        # metric_matrix = np.sqrt(-2*np.dot(matrix1, matrix2.T) + np.sum(np.square(matrix2), axis = 1) + np.transpose([np.sum(np.square(matrix1), axis = 1)]))
    
    if method=='cosine': # In the case of cosine, logarithmic operations are required to ensure that the lower the value for the better performance
        metric_matrix = np.exp(-metric_matrix)

    return metric_matrix

def get_two_graph_scores(embeddings1, embeddings2, operation, method):
    """
    Calculate the alignment accuracy of nodes between two graphs under different topK/calculate the optimal alignment pair
    """
    assert embeddings1.shape==embeddings2.shape, "embeddings1.shape!=embeddings2.shape" # Assert whether the number of embedding nodes is equal
    assert operation in ['evaluate', 'match'], "Unkown operation!" # Assert the operation is evaluation or alignment
    top1_count = 0
    top5_count = 0
    top10_count = 0
    pairs = []
    metric_matrix = get_metric_matrix(embeddings1, embeddings2, method)
    time_head = time.time() # Record starting time
    for i in range(len(embeddings1)):
        sort = np.argsort(metric_matrix[i])
        if operation=='evaluate': # When 'evaluate', analyze the 'top'
            if sort[0] == i: # Best match situation
                top1_count += 1
                top5_count += 1
                top10_count += 1
            else:
                for num in range(10):
                    if num <5 and sort[num] == i:
                        top5_count += 1
                        top10_count += 1
                    elif sort[num] == i:
                        top10_count += 1
        else: # When 'match', record the best alignment pairs
            pairs.append(sort[0])
        if i % 1000 == 0:
            time_tail = time.time() # Record finish time
            print("=====> Have matched %d nodes\tTime: %.3f"%(i, time_tail-time_head))
            if operation=='evaluate':
                print("=====> Accuracy number: top-1 %d, top-5 %d, top-10 %d"%(top1_count, top5_count, top10_count))
            time_head = time.time() # Record starting time
    if operation=='evaluate':
        return top1_count/len(embeddings1), top5_count/len(embeddings1), top10_count/len(embeddings1)
    else:
        assert len(pairs)==len(embeddings1), "Length of pairs is error!" # Assert the length of the optimal alignment pair is correct
        return pairs

def load_embeddings(params, source):
    """
    Load the initial feature matrix obtained by encoding profile
    """
    time_head = time.time() # Record starting time
    feature = np.ones(shape=(params.node_num, params.init_dim)) # Initialize the embedding matrix
    emb_dir = params.emb_dir + params.dataset
    emb_dir += "_source.emb" if source else "_target.emb" # Set the embedding file path
    with open(emb_dir, 'r', encoding='utf-8') as f: # Read embedding file
        for i, line in enumerate(f.readlines()): # Read line by line
            if i == 0:
                result = line.strip().split(' ')
                # Assert embedding information
                assert len(result)==2, "Information format of embeddings is error!"
                assert int(result[0])==params.node_num, "Amount of embeddings is error!"
                assert int(result[1])==params.init_dim, "Dimension of embeddings is error!"
            else:
                node, vector = line.strip().split(' ', 1) # Divide a single line of text into nodes and embedding vectors
                vector = np.fromstring(vector, sep=' ') # Convert text vector to numpy vector
                assert len(vector)==params.init_dim, "Length of embeddings is error!" # Assert the embedding length is correct
                feature[int(node)] = vector # Store the resulting vector in the corresponding row of the embedding matrix
    embeddings = torch.from_numpy(feature).float() # Transform numpy to tensor
    embeddings = embeddings.cuda() if params.cuda else embeddings # Transform cuda
    time_tail = time.time() # Record finish time
    if source:
        print("Source embeddings have been loaded!\tTime: %.3f"%(time_tail-time_head))
    else:
        print("Target embeddings have been loaded!\tTime: %.3f"%(time_tail-time_head))
    assert embeddings.size() == (params.node_num, params.init_dim), "Size of embeddings is error!" # Assert embedding size is correct
    # Initialize cuda
    if params.cuda:
        embeddings.cuda()

    return embeddings
