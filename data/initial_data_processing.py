##########################################################################
# @File: initial_data_processing.py
# @Author: Zhehan Liang
# @Date: 6/8/2020
# @Intro: Adjust the original data, process the number of nodes, the minimum degree of the deleted edge node,
# the data format, etc., and delete a certain number of non-duplicated edges respectively,
# and finally save them as two new graphs of the source domain and the target domain, and the saving format is .txt
# tips: Non-cross network, e.g. flickr-flickr
# @Data source: https://www.aminer.cn/cosnet
##########################################################################

import numpy as np
import os
import time
import argparse

def parse_args():
    # Parameter setting
    parser = argparse.ArgumentParser(description="Initial data processing.")
    parser.add_argument('--dataset', '-d', type=str, default='flickr', help='dataset name.')
    parser.add_argument('--total_num', '-n', type=int, default=10000, help='Number of nodes.')
    parser.add_argument('--start_node', '-s', type=int, default=0, help='Start node.')
    parser.add_argument('--degree', '-g', type=int, default=3, help='The nodes which will discard edge with this degree at least.')
    parser.add_argument('--discard_edge_rate', '-r', type=float, default=0.01, help='The discard edge rate of the generated subgraph.')
    return parser.parse_args()

def data_processing(args):
    ## Parameter setting
    dataset = args.dataset
    input_dir = "./graph_data/" # Original data file path
    output_dir = "./graph_edge/" # New data file path
    total_num = args.total_num # Total number of nodes after discarding
    start_node = args.start_node # The number of the first node to select, the default starts from 0
    degree = args.degree # The minimum degree of discarded node
    nodes_range = range(start_node, start_node+total_num) # Limit node search range
    discard_edge_rate = args.discard_edge_rate # The proportion of discarded edges in a single graph

    ## Path setting
    input_data_dir = input_dir + "{0}/{0}.edges".format(dataset) # Original data path
    output_new_data_dir = output_dir + "{0}-{0}_{1}_new.edges".format(dataset, total_num) # File path of graph with specific number of nodes
    output_source_dir = output_dir + "{0}-{0}_source_edges.txt".format(dataset) # File path of source domain graph
    output_target_dir = output_dir + "{0}-{0}_target_edges.txt".format(dataset) # File path of target domain graph

    ## Read the edges in the original graph data and save it (after saving, you can avoid running this part twice)
    time_head = time.time()
    edges = ""
    f = open(input_data_dir, 'r', encoding='utf-8')
    for line in f.readlines(): # Read in each row of data in turn
        nodes = line.strip().split(' ') # nodes = [node0, node1]
        if int(nodes[0]) in nodes_range and int(nodes[1]) in nodes_range and nodes[0]!=nodes[1]: # Only record data within the specific range of the nodes
            edges += "{0} {1}\n".format(nodes[0], nodes[1])
        if int(nodes[0])>=total_num: break # According to the characteristics of the original data format, the search can be ended early
    f.close()
    if not os.path.exists(output_dir): # Check whether the output path exists, if not, just create it
        os.makedirs(output_dir)
    f = open(output_new_data_dir, 'w', encoding='utf-8')
    f.write(edges)
    f.close()
    time_tail = time.time()
    print("%d nodes had been saved!\n1.Time:%.3f"%(total_num, time_tail-time_head))

    ## Calculate the data of the new graph
    time_head = time.time()
    present_node = start_node # The current node identifier, starting from the 'start_node'
    index_edge = 0 # Edge index
    num_degree = 0 # Record the degree of the current node
    num_high_degree_edge = 0 # Record the number of edges in the sub-graph composed of nodes that meet the degree requirement, that is, the number of edges can be discarded
    node_set = set() # Record the node that was read (to facilitate the subsequent supplement of the missing island node in the dicarding process)
    nodes_beyond_degree = set() # Record nodes that meet degree requirement
    can_discard_list = [] # Record the index of edges that can be discarded

    ## Traverse the file for the first time
    ## 1. find the nodes whose degree meets the requirements
    ## 2. record all the read nodes and the total number of edges in the original graph
    f = open(output_new_data_dir, 'r', encoding='utf-8')
    for line in f.readlines():
        nodes = line.strip().split(' ') # Read in each row of data in turn
        if int(nodes[0])==present_node: # Still the same node
            num_degree += 1 # The degree of the current node plus 1
        else: # When recording to the next node
            if num_degree>=degree: # If the degree of the previous node reaches the requirement
                nodes_beyond_degree.add(present_node) # Record this node
            num_degree = 1 # Initialize the degree of the new node to 1
            present_node = int(nodes[0]) # Update node identifier
        node_set.add(int(nodes[0])) # Add the read node
        node_set.add(int(nodes[1])) # Add the read node
        index_edge += 1 # Update the index of the current edge
    f.close()
    num_edge = index_edge # Record the total number of edges in the original graph
    index_edge = 0 # The index of edge

    ## Traverse the file for the second time, and determine the number and index of the edges
    ## that can be discarded according to the nodes whose degree meets the requirements
    f = open(output_new_data_dir, 'r', encoding='utf-8')
    for line in f.readlines():
        nodes = line.strip().split(' ') # Read in each row of data in turn
        if nodes[0]!=nodes[1] and int(nodes[0]) in nodes_beyond_degree and int(nodes[1]) in nodes_beyond_degree: # Self-connections are not included edges that can be discarded
            can_discard_list.append(index_edge)
            num_high_degree_edge += 1
        index_edge += 1 # Update the index of the current edge
    f.close()
    time_tail = time.time()
    print("The num of nodes is %d, the max node is %d" %(len(node_set), max(node_set)))
    print("The num of edges is %d" % num_edge)
    print("The num of high-degree nodes is %d" % (len(nodes_beyond_degree)))
    print("The num of high-degree edges is %d" % (num_high_degree_edge))
    num_edge_discard = int(num_edge * discard_edge_rate) # Calculate whether there are more edges that can be discarded than need to be discarded
    print("The num of edges need to discard is %d" % num_edge_discard)
    assert num_high_degree_edge>2*num_edge_discard, "Edges aren't enough to discard!" # Assert that there are enough edges to discard(two times because there are two subgraphs)
    print("The num of edges after discarding is %d" % (num_edge-num_edge_discard))
    print("Edges are enough to discard!")
    print("2.Time:%.3f\n################################################"%(time_tail-time_head))

    ## Randomly select edges to be discarded in two graphs
    time_head = time.time()
    edge_set_1 = set()
    edge_set_2 = set()
    while len(edge_set_1)<num_edge_discard:
        rd = np.random.randint(0, len(can_discard_list))
        edge_set_1.add(can_discard_list[rd])
        if len(edge_set_1)%10000==0: # Show in every 10000 steps
            print("Have select %d edges to discard in G1"%(len(edge_set_1)))
    while len(edge_set_2)<num_edge_discard:
        rd = np.random.randint(0, len(can_discard_list))
        if can_discard_list[rd] not in edge_set_1: # The discarded edges cannot be the same
            edge_set_2.add(can_discard_list[rd])
            if len(edge_set_2)%10000==0:
                print("Have select %d edges to discard in G2"%(len(edge_set_2)))
    time_tail = time.time()
    print("3.Time:%.3f\n################################################"%(time_tail-time_head))

    ## Construct the source domain and target domain subgraphs, 
    ## where the coexistence edge calculation is performed to verify whether the construction result is correct
    time_head = time.time()
    index_edge = 0 # The index of the current edge
    num_com = 0 # The number of edges that remained in both two subgraphs
    flag = 0 # Coexistence identifier, to determine whether the current edge remains in both subgraphs
    edges_source = "" # The edge to remain in the source domain
    edges_target = "" # The edge to remain in the target domain
    node_set_s = set() # The node to remain in the source domain
    node_set_t = set() # The node to remain in the target domain

    f = open(output_new_data_dir, 'r', encoding='utf-8')
    for line in f.readlines():
        if index_edge in edge_set_1 and index_edge in edge_set_2: # Check if there are edges in both sets to be discarded
            print("ERROR: Have the edge that both to discard in G1 ande G2!")
        else:
            nodes = line.strip().split(' ')
            if index_edge not in edge_set_1:
                # edges_source += "{0} {1} {'weight': 1.0}\n".format(nodes[0], nodes[1])
                edges_source += "{0} {1} \n".format(nodes[0], nodes[1])
                node_set_s.add(int(nodes[0]))
                node_set_s.add(int(nodes[1]))
                flag += 1
            if index_edge not in edge_set_2:
                # edges_target += "{0} {1} {'weight': 1.0}\n".format(nodes[0], nodes[1])
                edges_target += "{0} {1} \n".format(nodes[0], nodes[1])
                node_set_t.add(int(nodes[0]))
                node_set_t.add(int(nodes[1]))
                flag += 1
        if flag==2: # When flag=2, it means that the current edge remains in both two subgraphs
            num_com += 1
        flag = 0 # Reset flag
        index_edge += 1 # Switch to the next edge
        if index_edge%100000==0: # Show in every 10000 edges
            print("%s edges have finished input" %index_edge)
    rate = 1.0*num_com/num_edge # Calculate the coexistence rate
    print("The rate of common edges is %f" % rate)
    f.close()
    time_tail = time.time()
    print("4.Time:%.3f\n################################################"%(time_tail-time_head))

    ## If there are isolated islands without accurate records, find them and make up the self-connection
    time_head = time.time()
    complete_set = set()
    for i in range(total_num): complete_set.add(i)
    # Find out the nodes that are not connected
    missing_list_s = list(complete_set-node_set_s)
    missing_list_t = list(complete_set-node_set_t)
    if len(missing_list_s)>0: # If there are outliers in the source domain
        for node in missing_list_s:
            # edges_source += "{0} {0} {'weight': 1.0}\n".format(str(node))
            edges_source += "{0} {0} \n".format(str(node)) # Add the self-connection of missing node
    if len(missing_list_t)>0: # If there are outliers in the target domain
        for node in missing_list_t:
            # edges_target += "{0} {0} {'weight': 1.0}\n".format(str(node))
            edges_target += "{0} {0} \n".format(str(node)) # Add the self-connection of missing node
    time_tail = time.time()
    print("5.Time:%.3f\n################################################"%(time_tail-time_head))

    ## Save source and target subgraphs
    f_s = open(output_source_dir, 'w', encoding='utf-8')
    f_t = open(output_target_dir, 'w', encoding='utf-8')
    f_s.write(edges_source)
    f_s.close()
    f_t.write(edges_target)
    f_t.close()
    print("Source graph and target graph have been saved!\n################################################")

if __name__ == "__main__":
    args = parse_args()
    data_processing(args)