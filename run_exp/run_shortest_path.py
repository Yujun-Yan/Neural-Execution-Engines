from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.append('../data/')
sys.path.append('../model/')
sys.path.append('../')
from model import *
from utils import *
from min_graph import get_random_graph

def run_dist(current_dir, reload_dir_1, reload_dir_2, checkpoint_path_1, checkpoint_path_2_data, checkpoint_path_2_msk):
    binary_size = 8

    Test_size = 100  # examples per size of graphs

    num_layers = 6
    dff = 128

    dropout_rate = 0.1
    res_ratio = 1.5

    num_max = 2 ** binary_size

    inf = 2 ** binary_size
    end_token = 2 ** (binary_size+1)  ####### end_token only used in sorting
    start_token_dec = 0

    EPOCHS = 1

    make_sym = True
    
    d_model_1 = 24
    target_vocab_size_add = binary_size + 1
    out_num_1 = True
    out_pos_1 = False
    assert(out_num_1 or out_pos_1)
    USE_positioning_1 = True
    pos_1 = 3
    
    d_model_2 = 16
    target_vocab_size_sort = binary_size + 2
    filter_size = 3
    num_filters = 16
    out_num_2 = True
    out_pos_2 = True
    assert(out_num_2 or out_pos_2)
    USE_positioning_2 = False
    pos_2 = 3
    
    with open("{}parameters.txt".format(current_dir), 'w') as fi:
        fi.write("binary_size: {}\nTest_size: {}\nnum_layers: {}\ndff: {}\ndropout_rate: {}\nres_ratio: {}\nd_model_1: {}\nd_model_2: {}\nfilter_size: {}\nnum_filters: {}\ntarget_vocab_size_add: {}\ntarget_vocab_size_sort: {}\nmake_sym: {}\nEPOCHS: {}\nout_num_2: {}\nout_pos_2: {}\nout_num_1: {}\nout_pos_1: {}\nUSE_positioning_2: {}\nUSE_positioning_1: {}\nnum_max: {}\nreload_dir_2: {}\nreload_dir_1: {}".format(
            binary_size, Test_size, num_layers, dff, dropout_rate, res_ratio, d_model_1, d_model_2, filter_size, num_filters, target_vocab_size_add, target_vocab_size_sort, make_sym, EPOCHS, out_num_2, out_pos_2, out_num_1, out_pos_1, USE_positioning_2, USE_positioning_1, num_max, reload_dir_2, reload_dir_1))
    
    transformer_1 = Transformer(num_layers, d_model_1, binary_size+1, dff, pos_1, target_vocab_size_add, make_sym, USE_positioning_1, out_num_1, out_pos_1, res_ratio, dropout_rate)
    transformer_2 = Transformer(num_layers, d_model_2, binary_size+2, dff, pos_2, target_vocab_size_sort, make_sym, USE_positioning_2, out_num_2, out_pos_2, res_ratio, dropout_rate)
    msk_transform_2 = mask_transform(num_filters, filter_size, dropout_rate)
    learning_rate_1 = CustomSchedule(d_model_1)

    optimizer_1 = tf.keras.optimizers.Adam(learning_rate_1, beta_1=0.9, beta_2=0.98, 
                                        epsilon=1e-9)
    learning_rate_2_data = CustomSchedule(d_model_2)
    learning_rate_2_msk = CustomSchedule(num_filters)

    optimizer_2_data = tf.keras.optimizers.Adam(learning_rate_2_data, beta_1=0.9, beta_2=0.98, 
                                        epsilon=1e-9)
    optimizer_2_msk = tf.keras.optimizers.Adam(learning_rate_2_msk, beta_1=0.9, beta_2=0.98, 
                                        epsilon=1e-9)
    
    ckpt_1 = tf.train.Checkpoint(transformer_1=transformer_1,
                           optimizer_1=optimizer_1)

    ckpt_manager_1 = tf.train.CheckpointManager(ckpt_1, checkpoint_path_1, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager_1.latest_checkpoint:
        ckpt_1.restore(ckpt_manager_1.latest_checkpoint)
        print ('Model_1 checkpoint restored!!')
    
    
    ckpt_2_data = tf.train.Checkpoint(transformer_2=transformer_2,
                           optimizer_2_data=optimizer_2_data)
    ckpt_2_msk = tf.train.Checkpoint(msk_transform_2=msk_transform_2,
                            optimizer_2_msk=optimizer_2_msk)
    ckpt_manager_2_data = tf.train.CheckpointManager(ckpt_2_data, checkpoint_path_2_data, max_to_keep=5)
    ckpt_manager_2_msk = tf.train.CheckpointManager(ckpt_2_msk, checkpoint_path_2_msk, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager_2_data.latest_checkpoint:
        ckpt_2_data.restore(ckpt_manager_2_data.latest_checkpoint)
        print ('Model_2_data checkpoint restored!!')
    if ckpt_manager_2_msk.latest_checkpoint:
        ckpt_2_msk.restore(ckpt_manager_2_msk.latest_checkpoint)
        print ('Model_2_msk checkpoint restored!!')
    
    
    def shortest_path(graph, seed_node, current_dist):
        ############## graph: adj(i,j), weight from i to j, (batch, num_node, num_node)
        ############## seed_node: one-hot vector, (batch, num_node)
        ############## current_dist: current dist. to the seed_node (batch,num_node)
        shortest_dist = []
        node_list = []
        seed_node = seed_node[:,:,tf.newaxis]                ########(batch, num_node, 1)
        current_dist = current_dist[:,:,tf.newaxis]          ########(batch, num_node, 1)
        seq_l = seed_node.shape[1]
        batch_size = seed_node.shape[0]
        enc_padding_mask = tf.cast(tf.transpose(seed_node, [0, 2, 1]), tf.float32)    ########(batch, 1, num_node)
        enc_padding_mask = tf.concat([enc_padding_mask, tf.zeros((batch_size, 1, 1), dtype=tf.float32)], -1) ########(batch, 1, num_node+1)
        enc_padding_mask = enc_padding_mask[:, tf.newaxis, :, :]  ######(batch, 1, 1, num_node+1)
        dec_padding_mask = enc_padding_mask
        combined_mask = None
        while (1):
            selected_adj_list = tf.matmul(graph,seed_node, transpose_a=True)   #####(batch, num_node, 1)
            selected_dist = tf.matmul(seed_node, current_dist, transpose_a=True) #####(batch, 1, 1)
    #         print(selected_adj_list)
    #         print(selected_dist)
            x_add = tf.concat([selected_adj_list, tf.tile(selected_dist, [1, seq_l, 1])], axis=-1)
            # print(x_add)
            dec_inp = tf.ones((batch_size, seq_l, 1), dtype=tf.int64)*start_token_dec
            predictions, _ = transformer_1(x_add, dec_inp, False, None, None, None) ########## compute possible paths
            y_add = back2int(tf.cast(tf.greater(predictions, 0), tf.int64)) ######(batch, num_node, 1)
    #         print(y_add)

            x_comp = tf.concat([y_add, current_dist], axis=-1)
            # print(x_comp)
            predictions, _, _ = transformer_2(x_comp, dec_inp, False, None, None, None) ######### update current distance lists
            current_dist = back2int(tf.cast(tf.greater(predictions, 0), tf.int64)) ########(batch, num_node, 1)
            # print(current_dist)
            x_min = tf.transpose(current_dist, [0, 2, 1])########(batch, 1, num_node)
            x_min = tf.concat([x_min, tf.ones((batch_size, 1, 1), dtype=tf.int64)*end_token], -1)
            
    #         print(x_min)
    #         print(dec_inp[:,:1,:])
            predictions, _, predicted_pos = transformer_2(x_min, dec_inp[:,:1,:], False, enc_padding_mask, combined_mask, dec_padding_mask) ######### find the next node and corresponding shortest distance
            ########### replace argmax (uncertain behavior) ############
            predicted_pos_max = tf.reduce_max(predicted_pos, axis=-1, keepdims=True)
            predicted_pos_max = tf.equal(predicted_pos, predicted_pos_max)
            predicted_pos_ind = tf.reshape(predicted_pos_max, [-1, predicted_pos_max.shape[-1]])
            predicted_pos_ind = tf.where(predicted_pos_ind)
            pos_id = tf.cast(tf.math.segment_min(predicted_pos_ind[:,1],predicted_pos_ind[:,0]), tf.int64)
            pos_id = tf.reshape(pos_id, [predicted_pos_max.shape[0], predicted_pos_max.shape[1]]) ###(batch, 1)
            seed_node = tf.one_hot(pos_id, seq_l, dtype=tf.int64)
            seed_node = tf.transpose(seed_node, [0,2,1])  ###(batch, num_node, 1)
            pred = tf.squeeze(predictions, -2)
            pred = tf.cast(tf.greater(pred, 0), tf.int64)
            out_binary = back2int(pred) ########(batch, 1)
            if tf.reduce_any(tf.equal(out_binary, end_token)):
                break
            shortest_dist.append(out_binary)
            node_list.append(pos_id)  ####(batch, 1) 
            chg_msk = tf.one_hot(pos_id, seq_l+1)[:,:,:,tf.newaxis]
            init_msk = tf.transpose(enc_padding_mask,[0,1,3,2])
            x = tf.concat([init_msk,chg_msk],axis=-1)
            x = tf.reshape(x,[-1, x.shape[-2], x.shape[-1]])
            x = 2*x-1
            predict_msk = msk_transform_2(x, False)
            predict_msk = tf.cast(tf.greater(predict_msk, 0), tf.float32)
            enc_padding_mask = predict_msk[:, tf.newaxis, tf.newaxis, :]
            dec_padding_mask = enc_padding_mask
        shortest_dist = tf.concat(shortest_dist, -1)
        node_list = tf.concat(node_list, -1)
        return shortest_dist, node_list


    # In[19]:


    graph = np.array([[0, 4, 0, 0, 0, 0, 0, 8, 0], [4, 0, 8, 0, 0, 0, 0, 11, 0], [0, 8, 0, 7, 0, 4, 0, 0, 2], [0, 0, 7, 0, 9, 14, 0, 0, 0], [0, 0, 0, 9, 0, 10, 0, 0, 0], [0, 0, 4, 14, 10, 0, 2, 0, 0], [0, 0, 0, 0, 0, 2, 0, 1, 6], [8, 11, 0, 0, 0, 0, 1, 0, 7], [0, 0, 2, 0, 0, 0, 6, 7, 0]])
    graph[graph==0]= inf
    graph = np.int64(graph)
    graph = graph[tf.newaxis, :, :]
    seed_node = np.int64(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]))
    current_dist = np.ones_like(seed_node)*inf
    current_dist[0] = 0
    seed_node = seed_node[tf.newaxis, :]
    current_dist = current_dist[tf.newaxis, :]
   
    shortest_dist, node_list = shortest_path(graph, seed_node, current_dist)
    print(node_list)
    print(shortest_dist)


    graph = np.array([[0, 4, 0, 0, 0, 0, 0, 8, 0], [4, 0, 8, 0, 0, 0, 0, 11, 0], [0, 8, 0, 7, 0, 4, 0, 0, 2], [0, 0, 7, 0, 9, 14, 0, 0, 0], [0, 0, 0, 9, 0, 10, 0, 0, 0], [0, 0, 4, 14, 10, 0, 2, 0, 0], [0, 0, 0, 0, 0, 2, 0, 1, 6], [8, 11, 0, 0, 0, 0, 1, 0, 7], [0, 0, 2, 0, 0, 0, 6, 7, 0]])
    g = Graph_paths(graph.shape[-1],inf)
    g.graph = graph
    dist_list = g.dijkstra(0)
    print(node_list)
    dist = [dist_list[node] for node in node_list[0]]
    print(dist)


    # In[23]:


    test_acc = tf.keras.metrics.Accuracy(name='test_acc')

    test_seq = [25, 50, 75, 100]

    for s in test_seq:
        test_acc.reset_states()
        print("Length: {}".format(s))
        graph_list = []
        seed_node_list = []
        current_dist_list = []
        dist_list_list = []
        graph_type = ['DR', 'NWS', 'BA', 'ER']
        for it in range(Test_size):
            graph = get_random_graph(num_max, s, graph_type[it%4], False)
            g = Graph_paths(s,inf)
            g.graph = graph
            source = np.random.randint(s)
            seed_node = np.zeros((s), dtype=np.int32)
            seed_node[source] = 1
            dist_list = g.dijkstra(source)
            current_dist = np.ones_like(seed_node)*inf
            current_dist[source] = 0
            graph[graph==0] = inf
            graph_list.append(graph)
            seed_node_list.append(seed_node)
            current_dist_list.append(current_dist)
            dist_list_list.append(dist_list)
        graph_list = tf.cast(tf.stack(graph_list, axis=0), tf.int64)
        seed_node_list = tf.cast(tf.stack(seed_node_list, axis=0), tf.int64)
        current_dist_list = tf.cast(tf.stack(current_dist_list, axis=0), tf.int64)
        
        shortest_dist, node_list = shortest_path(graph_list, seed_node_list, current_dist_list)
        dist = []
        for i in range(len(dist_list_list)):
            dist.append(tf.gather(dist_list_list[i], node_list[i,:]))
        dist = tf.stack(dist, axis=0)
        test_acc.update_state(dist, shortest_dist)
        print("Acc: {}".format(test_acc.result()))
                                   

