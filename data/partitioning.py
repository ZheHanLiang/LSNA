# -*- coding:utf-8 _*-
import os
import csv
import time
import copy
import pickle
import argparse
import numpy as np
from community_louvain_adaption import best_partition
import networkx as nx
from collections import defaultdict
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram, NeighborhoodHash
wl_kernel = WeisfeilerLehman(normalize=True, base_graph_kernel=NeighborhoodHash)

def parse_args():
    # Parameter setting
    parser = argparse.ArgumentParser(description="Partitioning")
    parser.add_argument('--dataset', '-d', type=str, default='FT', help='FT=Facebook-Twitter, WD=Weibo-Douban.')
    parser.add_argument('--input_dir', '-i', type=str, default='./dataset/', help='Path of input files.')
    parser.add_argument('--output_dir', '-o', type=str, default='./dataset/align/', help='Path of output files.')
    return parser.parse_args()

def partitioning(args):
    if args.dataset=="FT":
        network_common_dir = args.input_dir + 'alignment_Facebook-Twitter.csv'
        network_names = ['facebook', 'twitter']
    elif args.dataset=="WD":
        network_common_dir = args.input_dir + 'alignment_Weibo-Douban.csv'
        network_names = ['weibo', 'douban']
    else:
        print("ERROR: please input the correct dataset name (FT or WD)!")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    common_dict = {}
    with open(network_common_dir, 'r') as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            common_dict[int(row[0])] = int(row[1])
    print("====>anchor number:",len(common_dict))

    for index, network_name in enumerate(network_names):
        print("====>{0}.{1}\tbegin!".format(index+1, network_name))
        G = nx.Graph()
        network_dir = args.input_dir + '{}_network.csv'.format(network_name)
        start = time.time()
        with open(network_dir, 'r') as f:
            reader = csv.reader(f)
            for i,row in enumerate(reader):
                if i==0:
                    continue
                node1 = int(row[0])
                node2 = int(row[1])
                G.add_edge(node1,node2)
        print("====>{0}.{1}\tedge number:".format(index+1, network_name), nx.number_of_edges(G), '\t', "node number:", nx.number_of_nodes(G), '\t', "Time:", time.time()-start)
        if index==0:
            G_s = G
        else:
            G_t = G
        start = time.time()
        part, iteration = best_partition(G,resolution=1,random_state=1)
        community_number = max(part.values()) + 1
        print("====>{0}.{1}\tcommunity number:".format(index+1, network_name), community_number, '\t', "iteration number:", iteration, '\t', "Time:", time.time()-start)
        start = time.time()
        com2nodes = defaultdict(list)
        for node,com in part.items():
            com2nodes[com].append(node)
        if index==0:
            com2nodes_s = com2nodes
        else:
            com2nodes_t = com2nodes
        length = []
        for com, node_list in com2nodes.items():
            length.append(len(node_list))
        print("====>{0}.{1}\tbig graph ratio:".format(index+1, network_name), '%.3f'%(len([i for i in length if i>500])*1.0/len(length)), '\t', "Num:", len(com2nodes), '\t', "Max:", max(length), '\t', "Time:", time.time()-start)
    ## alignment
    start = time.time()
    com_list_s = []
    com_G_list_s = []
    com_list_t = []
    com_G_list_t = []
    for com in com2nodes_s:
        com_list_s.append(com)
        G_sub = G_s.subgraph(com2nodes_s[com])
        node_num = nx.number_of_nodes(G_sub)
        G_adj = nx.adjacency_matrix(G_sub)
        G_node_labels = {i:0 for i in range(node_num)}
        com_G_list_s.append(Graph(initialization_object=G_adj, node_labels=G_node_labels))
    for com in com2nodes_t:
        com_list_t.append(com)
        G_sub = G_t.subgraph(com2nodes_t[com])
        node_num = nx.number_of_nodes(G_sub)
        G_adj = nx.adjacency_matrix(G_sub)
        G_node_labels = {i:0 for i in range(node_num)}
        com_G_list_t.append(Graph(initialization_object=G_adj, node_labels=G_node_labels))
    wl_kernel.fit_transform(com_G_list_s)
    similarity = wl_kernel.transform(com_G_list_t).T
    align = {}
    for i in range(similarity.shape[0]):
        row = similarity[i].tolist()
        max_index = row.index(max(row))
        align[i] = max_index
    for index, key in enumerate(align):
        str_s = ""
        str_t = ""
        G_sub_s = G_s.subgraph(com2nodes_s[com_list_s[key]])
        G_sub_t = G_t.subgraph(com2nodes_t[com_list_t[align[key]]])
        for edge in nx.edges(G_sub_s):
            str_s += '{0} {1} \n'.format(edge[0], edge[1])
        for edge in nx.edges(G_sub_t):
            str_t += '{0} {1} \n'.format(edge[0], edge[1])
        f_s = open(args.output_dir+"facebook_{0}".format(index), 'w', encoding='utf-8')
        f_t = open(args.output_dir+"twitter_{0}".format(index), 'w', encoding='utf-8')
        f_s.write(str_s)
        f_s.close()
        f_t.write(str_t)
        f_t.close()
    print("====>all the subnetworks have been saved! Time:", time.time()-start)

if __name__ == "__main__":
    args = parse_args()
    partitioning(args)