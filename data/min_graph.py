from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import subprocess
import networkx as nx
import sys
sys.path.append('../')
from utils import *


def get_weighted_graph(adj, graph_size, num_max_2, mst):
    '''
    assign weights to the graphs
    adj: numpy matrix, adjacnecy matrix
    graph_size: int, number of nodes 
    num_max_2: maximum number allowed
    mst: whether is used for minimum spanning tree
    
    '''
    if mst:
        rnd_or_sm = np.random.uniform()
        if rnd_or_sm>0.8:
            adj = np.multiply(adj, np.random.choice(range(1, num_max_2),size=(graph_size,graph_size)))
        else: ######## 20% for hard examples
            diff = np.random.choice(range(1, 6))
            d_min = 1+diff
            upper = np.random.choice(range(d_min, num_max_2))
            adj = np.multiply(adj, np.random.choice(range(upper-diff, upper+1),size=(graph_size,graph_size)))
        i_lower = np.tril_indices(graph_size, -1)
        adj[i_lower] = adj.T[i_lower]
    else: ###### shortest path #########
        rnd_or_sm = np.random.uniform()
        weight_max = np.ceil(num_max_2/graph_size).astype(int)
        if rnd_or_sm>0.5:
            adj = np.multiply(adj, np.random.choice(range(1, weight_max),size=(graph_size,graph_size)))
        else: ##### 50% for hard examples
            diff = np.random.choice(range(1, 6))
            d_min = 1+diff
            if d_min<weight_max:
                upper = np.random.choice(range(d_min, weight_max))
                adj = np.multiply(adj, np.random.choice(range(upper-diff, upper+1),size=(graph_size,graph_size)))
            else:
                adj = np.multiply(adj, np.random.choice(range(1, weight_max),size=(graph_size,graph_size)))
        i_lower = np.tril_indices(graph_size, -1)
        adj[i_lower] = adj.T[i_lower]
    return adj


def get_traces(adj, src, graph_size, inf, mst):
    '''
    Get the traces from corresponding graph algorithms
    '''
    if mst:
        g = Graph_mst(graph_size, inf)
        g.graph = adj
        return g.primmst(src, traces=True)
    else:
        g= Graph_paths(graph_size, inf)
        g.graph = adj
        return g.dijkstra(src, traces=True)


def get_end_exmp(data_sel):
    '''
    examples with end_token, for training end_token properly
    '''
    sorted_data = np.sort(data_sel)
    sorted_data_ind = np.argsort(data_sel)
    sort_data_mask = np.zeros((data_sel.shape[-1]+1, data_sel.shape[-1]), dtype=np.float32)
    for i in range(1, data_sel.shape[-1]+1):
        sort_data_mask[i, sorted_data_ind[i-1]] = 1
    return np.tile(data_sel,[data_sel.shape[-1], 1]), sort_data_mask, sorted_data[:,np.newaxis]
    
def get_random_graph(num_max_2, graph_size, graph_type, mst):
    '''
    num_max_2: maximum number allowed
    graph_size: number of nodes
    graph_type: 'DR', d-regular, 'NWS':  Newman–Watts–Strogatz, 'BA': Barabási–Albert,  'ER': Erdős-Rényi
    mst: whether it is used for minimum spanning tree task
       
    '''
    
    if graph_type == 'DR':
        if graph_size % 2: ### odd
            deg = np.random.choice(range(2, graph_size, 2))
        else:
            deg = np.random.choice(range(2, graph_size))
        graph_ele = nx.generators.random_graphs.random_regular_graph(deg, graph_size)
        adj = np.asarray(nx.adjacency_matrix(graph_ele).todense())
        adj = get_weighted_graph(adj, graph_size, num_max_2, mst)
    elif graph_type == 'NWS':
        k = np.random.choice(range(2, min(6,graph_size)))
        p = np.random.uniform()
        graph_ele = nx.generators.random_graphs.newman_watts_strogatz_graph(graph_size, k, p)
        adj = np.asarray(nx.adjacency_matrix(graph_ele).todense())
        np.fill_diagonal(adj, 0)
        adj = get_weighted_graph(adj, graph_size, num_max_2, mst)
    elif graph_type == 'BA':
        m = np.random.choice(range(2, min(6,graph_size)))
        graph_ele = nx.generators.random_graphs.barabasi_albert_graph(graph_size, m)
        adj = np.asarray(nx.adjacency_matrix(graph_ele).todense())
        np.fill_diagonal(adj, 0)
        adj = get_weighted_graph(adj, graph_size, num_max_2, mst)
    else:
        p = np.random.uniform()
        graph_ele = nx.generators.random_graphs.erdos_renyi_graph(graph_size, p)
        adj = np.asarray(nx.adjacency_matrix(graph_ele).todense())
        np.fill_diagonal(adj, 0)
        adj = get_weighted_graph(adj, graph_size, num_max_2, mst)
    return adj    
        

        



def min_graph(current_dir, reload_from_dir_2, reload_dir_2, num_max_2, data_size, graph_size, end_token, inf, graph_type, name, mst=True, store_data=True, combined=True): ##### Erdős-Rényi, Newman–Watts–Strogatz, d-regular, Barabási–Albert
    '''
    current_dir: string, current checkpoint and data will be saved in this directory
    reload_from_dir_2: bool, whether to reload
    reload_dir_2: string, reload directory
    num_max_2: int, maximum number allowed
    data_size: int, number of graph samples
    graph_size: number of nodes
    end_token: int, a number used to represent infinity
    name: name to reload or save the numpy matrix
    mst: whether it is for minimum spanning tree task
    store_data: whether to store the data generated
    combined: whether to combine the data into tensor slices
    '''
    if not reload_from_dir_2:
        data_exmp = np.zeros((data_size, graph_size, graph_size), dtype=np.int64)
        data_mask = np.zeros((data_size, graph_size+1, graph_size),dtype=np.float32)
        data_true = np.zeros((data_size, graph_size, 1), dtype=np.int64)
        p_end_token = graph_size/num_max_2
        end_t_exmp = []
        end_t_mask = []
        end_t_true = []
        if graph_type == 'DR':
            for i in range(data_size):
                if graph_size % 2: ### odd
                    deg = np.random.choice(range(2, graph_size, 2))
                else:
                    deg = np.random.choice(range(2, graph_size))
                graph_ele = nx.generators.random_graphs.random_regular_graph(deg, graph_size)
                adj = np.asarray(nx.adjacency_matrix(graph_ele).todense())
                ###### assign weight ########
                adj = get_weighted_graph(adj, graph_size, num_max_2, mst)
                # print(adj)
                src = np.random.choice(range(graph_size))
                # print(src)
                data_seq, data_msk, data_true_seq = get_traces(adj, src, graph_size, inf, mst)
                data_exmp[i,:,:] = data_seq
                data_mask[i,:,:] = data_msk
                data_true[i,:,:] = data_true_seq
                end_show = np.random.binomial(1,p_end_token)
                # end_show = True
                if end_show:
                    data_sel = np.concatenate((data_seq[-1,:-1],[end_token]))
                    data_s, data_s_msk, data_s_truth = get_end_exmp(data_sel)
                    end_t_exmp.append(data_s)
                    end_t_mask.append(data_s_msk)
                    end_t_true.append(data_s_truth)
            # print(data_exmp)
            # print(data_mask)
            # print(data_true)
            data_sel = np.concatenate((data_seq[-1,:-1],[end_token]))
            data_s, data_s_msk, data_s_truth = get_end_exmp(data_sel)
            end_t_exmp.append(data_s)
            end_t_mask.append(data_s_msk)
            end_t_true.append(data_s_truth)
            end_t_exmp = np.stack(end_t_exmp)
            end_t_mask = np.stack(end_t_mask)
            end_t_true = np.stack(end_t_true)
            data_exmp = np.concatenate((data_exmp, end_t_exmp))
            data_mask = np.concatenate((data_mask, end_t_mask))
            data_true = np.concatenate((data_true, end_t_true))
            # print(data_exmp)
            # print(data_mask)
            # print(data_true)
                                        
        elif graph_type == 'NWS':
            for i in range(data_size):
                k = np.random.choice(range(2, min(6,graph_size)))
                p = np.random.uniform()
                graph_ele = nx.generators.random_graphs.newman_watts_strogatz_graph(graph_size, k, p)
                adj = np.asarray(nx.adjacency_matrix(graph_ele).todense())
                np.fill_diagonal(adj, 0)
                adj = get_weighted_graph(adj, graph_size, num_max_2, mst)
                # print(adj)
                src = np.random.choice(range(graph_size))
                data_seq, data_msk, data_true_seq = get_traces(adj, src, graph_size, inf, mst)
                data_exmp[i,:,:] = data_seq
                data_mask[i,:,:] = data_msk
                data_true[i,:,:] = data_true_seq
                end_show = np.random.binomial(1,p_end_token)
                # end_show = True
                if end_show:
                    data_sel = np.concatenate((data_seq[-1,:-1],[end_token]))
                    data_s, data_s_msk, data_s_truth = get_end_exmp(data_sel)
                    end_t_exmp.append(data_s)
                    end_t_mask.append(data_s_msk)
                    end_t_true.append(data_s_truth)
            data_sel = np.concatenate((data_seq[-1,:-1],[end_token]))
            data_s, data_s_msk, data_s_truth = get_end_exmp(data_sel)
            end_t_exmp.append(data_s)
            end_t_mask.append(data_s_msk)
            end_t_true.append(data_s_truth)
            end_t_exmp = np.stack(end_t_exmp)
            end_t_mask = np.stack(end_t_mask)
            end_t_true = np.stack(end_t_true)
            data_exmp = np.concatenate((data_exmp, end_t_exmp))
            data_mask = np.concatenate((data_mask, end_t_mask))
            data_true = np.concatenate((data_true, end_t_true))
            
        elif graph_type == 'BA':
            for i in range(data_size):
                m = np.random.choice(range(2, min(6,graph_size)))
                graph_ele = nx.generators.random_graphs.barabasi_albert_graph(graph_size, m)
                adj = np.asarray(nx.adjacency_matrix(graph_ele).todense())
                np.fill_diagonal(adj, 0)
                adj = get_weighted_graph(adj, graph_size, num_max_2, mst)
                # print(adj)
                src = np.random.choice(range(graph_size))
                data_seq, data_msk, data_true_seq = get_traces(adj, src, graph_size, inf, mst)
                data_exmp[i,:,:] = data_seq
                data_mask[i,:,:] = data_msk
                data_true[i,:,:] = data_true_seq
                end_show = np.random.binomial(1,p_end_token)
                # end_show = True
                if end_show:
                    data_sel = np.concatenate((data_seq[-1,:-1],[end_token]))
                    data_s, data_s_msk, data_s_truth = get_end_exmp(data_sel)
                    end_t_exmp.append(data_s)
                    end_t_mask.append(data_s_msk)
                    end_t_true.append(data_s_truth)
            data_sel = np.concatenate((data_seq[-1,:-1],[end_token]))
            data_s, data_s_msk, data_s_truth = get_end_exmp(data_sel)
            end_t_exmp.append(data_s)
            end_t_mask.append(data_s_msk)
            end_t_true.append(data_s_truth)
            end_t_exmp = np.stack(end_t_exmp)
            end_t_mask = np.stack(end_t_mask)
            end_t_true = np.stack(end_t_true)
            data_exmp = np.concatenate((data_exmp, end_t_exmp))
            data_mask = np.concatenate((data_mask, end_t_mask))
            data_true = np.concatenate((data_true, end_t_true))
                
        elif graph_type == 'ER':
            for i in range(data_size):
                p = np.random.uniform()
                graph_ele = nx.generators.random_graphs.erdos_renyi_graph(graph_size, p)
                adj = np.asarray(nx.adjacency_matrix(graph_ele).todense())
                np.fill_diagonal(adj, 0)
                adj = get_weighted_graph(adj, graph_size, num_max_2, mst)
                # print(adj)
                src = np.random.choice(range(graph_size))
                data_seq, data_msk, data_true_seq = get_traces(adj, src, graph_size, inf, mst)
                data_exmp[i,:,:] = data_seq
                data_mask[i,:,:] = data_msk
                data_true[i,:,:] = data_true_seq
                end_show = np.random.binomial(1,p_end_token)
                # end_show = True
                if end_show:
                    data_sel = np.concatenate((data_seq[-1,:-1],[end_token]))
                    data_s, data_s_msk, data_s_truth = get_end_exmp(data_sel)
                    end_t_exmp.append(data_s)
                    end_t_mask.append(data_s_msk)
                    end_t_true.append(data_s_truth)
            data_sel = np.concatenate((data_seq[-1,:-1],[end_token]))
            data_s, data_s_msk, data_s_truth = get_end_exmp(data_sel)
            end_t_exmp.append(data_s)
            end_t_mask.append(data_s_msk)
            end_t_true.append(data_s_truth)
            end_t_exmp = np.stack(end_t_exmp)
            end_t_mask = np.stack(end_t_mask)
            end_t_true = np.stack(end_t_true)
            data_exmp = np.concatenate((data_exmp, end_t_exmp))
            data_mask = np.concatenate((data_mask, end_t_mask))
            data_true = np.concatenate((data_true, end_t_true))
                
        else:
            num_gtype = 4
            data_size_floor = np.floor(data_size/num_gtype).astype(int)
            data_size_list = [data_size_floor]*(num_gtype-1)
            data_size_list.append(data_size-data_size_floor*(num_gtype-1))
            gtype = ['DR', 'NWS', 'BA', 'ER']
            data_exmp = []
            data_mask = []
            data_true = []
            for i in range(num_gtype):
                d_exmp, d_msk, d_t = min_graph(current_dir, reload_from_dir_2, reload_dir_2, num_max_2, data_size_list[i], graph_size, end_token, inf, gtype[i], name, mst, False, False)
                data_exmp.append(d_exmp)
                data_mask.append(d_msk)
                data_true.append(d_t)
            data_exmp = np.concatenate(data_exmp)
            data_mask = np.concatenate(data_mask)
            data_true = np.concatenate(data_true)
            # print(data_exmp)
            # print(data_mask)
            # print(data_true)
        if store_data:
            np.save("{}{}_data_exmp.npy".format(current_dir, name), data_exmp)
            np.save("{}{}_data_mask.npy".format(current_dir, name), data_mask)
            np.save("{}{}_data_true.npy".format(current_dir, name), data_true)  
    else:
        data_exmp = np.load("{}{}_data_exmp.npy".format(reload_dir_2, name))
        data_mask = np.load("{}{}_data_mask.npy".format(reload_dir_2, name))
        data_true = np.load("{}{}_data_true.npy".format(reload_dir_2, name))
       
    
    if combined:
        # print("data_exmp:")
        # print(data_exmp[0,:,:])
        # print("data_mask:")
        # print(data_mask[0,:,:])
        # print("data_true:")
        # print(data_true[0,:,:])
        
        dataset = tf.data.Dataset.from_tensor_slices((data_exmp, data_mask, data_true))    
        return dataset, data_exmp.shape[0]
    else:
        return data_exmp, data_mask, data_true