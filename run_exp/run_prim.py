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

####### only needs min function ################
def run_prim(current_dir, reload_dir_data, reload_dir_msk, checkpoint_path_data_1, checkpoint_path_data_2, checkpoint_path_msk):
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
    
    d_model = 16
    target_vocab_size_sort = binary_size + 2
    filter_size = 3
    num_filters = 16
    out_num_1 = True
    out_pos_1 = True
    out_num_2 = True
    out_pos_2 = True
    assert(out_num_1 or out_pos_1)
    assert(out_num_2 or out_pos_2)
    USE_positioning = False
    pos = 3
    
    with open("{}parameters.txt".format(current_dir), 'w') as fi:
        fi.write("binary_size: {}\nTest_size: {}\nnum_layers: {}\ndff: {}\ndropout_rate: {}\nres_ratio: {}\nd_model: {}\nfilter_size: {}\nnum_filters: {}\ntarget_vocab_size_sort: {}\nmake_sym: {}\nEPOCHS: {}\nout_num_2: {}\nout_pos_2: {}\nout_num_1: {}\nout_pos_1: {}\nUSE_positioning: {}\nnum_max: {}\nreload_dir_data: {}\nreload_dir_msk: {}".format(
            binary_size, Test_size, num_layers, dff, dropout_rate, res_ratio, d_model, filter_size, num_filters, target_vocab_size_sort, make_sym, EPOCHS, out_num_2, out_pos_2, out_num_1, out_pos_1, USE_positioning, num_max, reload_dir_data, reload_dir_msk))
    
    transformer_1 = Transformer(num_layers, d_model, binary_size+2, dff, pos, target_vocab_size_sort, make_sym, USE_positioning, out_num_1, out_pos_1, res_ratio, dropout_rate) ##### update node_val, masked min
    transformer_2 = Transformer(num_layers, d_model, binary_size+2, dff, pos, target_vocab_size_sort, make_sym, USE_positioning, out_num_2, out_pos_2, res_ratio, dropout_rate) ### select the new node to include to mst
    msk_transform = mask_transform(num_filters, filter_size, dropout_rate)
    learning_rate_1 = CustomSchedule(d_model)
    learning_rate_2_data = CustomSchedule(d_model)
    learning_rate_2_msk = CustomSchedule(num_filters)
    
    optimizer_1 = tf.keras.optimizers.Adam(learning_rate_1, beta_1=0.9, beta_2=0.98, 
                                            epsilon=1e-9)
    optimizer_2_data = tf.keras.optimizers.Adam(learning_rate_2_data, beta_1=0.9, beta_2=0.98, 
                                        epsilon=1e-9)
    optimizer_2_msk = tf.keras.optimizers.Adam(learning_rate_2_msk, beta_1=0.9, beta_2=0.98, 
                                        epsilon=1e-9)
    
    ckpt_1 = tf.train.Checkpoint(transformer_2=transformer_1,
                           optimizer_2_data=optimizer_1)

    ckpt_2_data = tf.train.Checkpoint(transformer_2=transformer_2,
                           optimizer_2_data=optimizer_2_data)
    ckpt_2_msk = tf.train.Checkpoint(msk_transform_2=msk_transform,
                            optimizer_2_msk=optimizer_2_msk)
    
    ckpt_manager_1 = tf.train.CheckpointManager(ckpt_1, checkpoint_path_data_1, max_to_keep=5)
    ckpt_manager_2_data = tf.train.CheckpointManager(ckpt_2_data, checkpoint_path_data_2, max_to_keep=5)
    ckpt_manager_2_msk = tf.train.CheckpointManager(ckpt_2_msk, checkpoint_path_msk, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    
    
        # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager_1.latest_checkpoint:
        ckpt_1.restore(ckpt_manager_1.latest_checkpoint)
        print ('Model_1 checkpoint restored!!')
    if ckpt_manager_2_data.latest_checkpoint:
        ckpt_2_data.restore(ckpt_manager_2_data.latest_checkpoint)
        print ('Model_2_data checkpoint restored!!')
    if ckpt_manager_2_msk.latest_checkpoint:
        ckpt_2_msk.restore(ckpt_manager_2_msk.latest_checkpoint)
        print ('Model_2_msk checkpoint restored!!')
    
    
    def mst(graph, seed_node, node_val):
        ############## graph: adj(i,j), weight from i to j, (batch, num_node, num_node)
        ############## seed_node: one-hot vector, (batch, num_node)
        ############## node_val: weight-related value (batch,num_node)
        ############## mst_set: nodes included are set to 1 (batch, num_node)
        
        mst_set = seed_node[:,:,tf.newaxis]  ###### (batch, num_node, 1)
        seed_node = seed_node[:,:,tf.newaxis] ###### (batch, num_node, 1)
        node_val = node_val[:,:,tf.newaxis]  ###### (batch, num_node, 1)
        
        weight_list = []
        next_node = []
        
        seq_l = seed_node.shape[1]
        batch_size = seed_node.shape[0]
        
        
        enc_sel = tf.concat([tf.cast(tf.transpose(mst_set,[0,2,1]), tf.float32), tf.zeros([batch_size, 1, 1])],-1) ##### (batch_size, 1, num_node+1)
        enc_sel = enc_sel[:,:,tf.newaxis,:]
        dec_sel = enc_sel
        combined_sel = None
        
        dec_inp = tf.ones((batch_size, seq_l, 1), dtype=tf.int64)*start_token_dec
        dec_sel_inp = tf.ones((batch_size, 1, 1), dtype=tf.int64)*start_token_dec
        # print(graph)
        while (1):
            enc_msk_val = tf.cast(tf.concat([tf.zeros_like(mst_set), mst_set], -1), tf.float32) ##### (batch, num_node, 2)
            enc_msk_val = enc_msk_val[:,:,tf.newaxis,:] ######(batch, num_node, 1, 2)
            dec_msk_val = enc_msk_val
            combined_msk_val = None
            
            selected_adj_list = tf.matmul(graph, seed_node, transpose_a=True)   #####(batch, num_node, 1)
            x_node_val = tf.concat([node_val, selected_adj_list], -1) ##### (batch, num_node, 2)
            
           
            node_val_pred, _, _ = transformer_1(x_node_val, dec_inp, False, enc_msk_val, combined_msk_val, dec_msk_val) ######## update node_val
            node_val = back2int(tf.cast(tf.greater(node_val_pred, 0), tf.int64)) ######(batch, num_node, 1)
            
           
            x_sel = tf.transpose(node_val, [0, 2, 1]) #### (batch, 1, num_node)
            x_sel = tf.concat([x_sel, tf.ones((batch_size, 1, 1), dtype=tf.int64)*end_token], -1)
            
           
            predictions, _, predicted_pos = transformer_2(x_sel, dec_sel_inp, False, enc_sel, combined_sel, dec_sel) ########## find the next node and corresponding weights
            
            weight_sel = tf.squeeze(back2int(tf.cast(tf.greater(predictions, 0), tf.int64)),-2) #### (batch, 1)
            
            if tf.reduce_any(tf.equal(weight_sel, end_token)):
                break
            weight_list.append(weight_sel)
            ########### replace argmax (uncertain behavior) ############
            # print(predicted_pos)
            predicted_pos_max = tf.reduce_max(predicted_pos, axis=-1, keepdims=True)
            predicted_pos_max = tf.equal(predicted_pos, predicted_pos_max)
            predicted_pos_ind = tf.reshape(predicted_pos_max, [-1, predicted_pos_max.shape[-1]])
            predicted_pos_ind = tf.where(predicted_pos_ind)
            pos_id = tf.cast(tf.math.segment_min(predicted_pos_ind[:,1],predicted_pos_ind[:,0]), tf.int64)
            pos_id = tf.reshape(pos_id, [predicted_pos_max.shape[0], predicted_pos_max.shape[1]]) ###(batch, 1)
            ###################
            next_node.append(pos_id)

            seed_node = tf.one_hot(pos_id, seq_l, dtype=tf.int64)
            seed_node = tf.transpose(seed_node, [0,2,1])  ###(batch, num_node, 1)
            
            init_msk = tf.transpose(enc_sel,[0,1,3,2]) ##### (batch_size, 1, num_node+1, 1)
            chg_msk = tf.one_hot(pos_id, seq_l+1)[:,:,:,tf.newaxis]
            x = tf.concat([init_msk,chg_msk],axis=-1)
            x = tf.reshape(x,[-1, x.shape[-2], x.shape[-1]])
            x = 2*x-1
            predict_msk = msk_transform(x, False)
            predict_msk = tf.cast(tf.greater(predict_msk, 0), tf.float32) ### (batch, num_node+1)
            enc_sel = tf.cast(predict_msk[:,tf.newaxis,tf.newaxis,:], tf.float32)
            dec_sel = enc_sel
            mst_set = tf.cast(predict_msk[:,:-1,tf.newaxis], tf.float32)
            
            
        weight_list = tf.concat(weight_list, -1)
        next_node = tf.concat(next_node, -1)
        return weight_list, next_node


    # In[19]:

    graph = np.array([[ 0, 222, 78, 0, 0, 0, 0, 0, 0, 0], [222, 0, 0, 0, 229, 0, 0, 0, 0, 0], [78, 0, 0, 174, 0, 0, 0, 0, 0, 0], [0, 0, 174, 0, 125, 0, 0, 0, 0, 0],[0, 229, 0, 125, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 191, 31, 0, 0], [0, 0, 0, 0, 0, 191, 0, 0, 24, 0], [0, 0, 0, 0, 0, 31, 0, 0, 0, 30], [0, 0, 0, 0, 0, 0, 24, 0, 0, 23],[0, 0, 0, 0, 0, 0, 0, 30, 23, 0]])
    print(graph)
    graph[graph==0]= inf
    graph = np.int64(graph)
    graph = graph[tf.newaxis, :, :]
    seed_node = np.int64(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
    node_val = np.ones_like(seed_node)*inf
    node_val[3] = 0
    seed_node = seed_node[tf.newaxis, :]
    node_val = node_val[tf.newaxis, :]
    weight_list, next_node = mst(graph, seed_node, node_val)
    weight_list = tf.squeeze(weight_list, 0)
    next_node = tf.squeeze(next_node, 0)
    b_flg = False
    for ind in range(weight_list.shape[-1]):
        if weight_list[ind] == inf:
            b_flg = True
            break
    if b_flg:
        next_node = next_node[:ind] 
        weight_list = weight_list[:ind]
    print(next_node)
    print(weight_list)


    test_acc = tf.keras.metrics.Accuracy(name='test_acc')

    test_seq = [25, 50, 75, 100]


    for s in test_seq:
        test_acc.reset_states()
        print("Length: {}".format(s))
        graph_list_ori = []
        seed_node_list = []
        node_val_list = []
        next_weight_list = []
        next_node_list = []
        it = 0
        graph_type = ['DR', 'NWS', 'BA', 'ER']
        while(it<Test_size):
            graph = get_random_graph(num_max, s, graph_type[it%4], True)
            g = Graph_mst(s, inf)
            g.graph = graph
            source = np.random.randint(s)
            seed_node = np.zeros((s), dtype=np.int32)
            seed_node[source] = 1
            weight, next_node= g.primmst(source)
            if len(weight)==0:
                continue
            else:
                it += 1
            node_val = np.ones_like(seed_node)*inf
            node_val[source] = 0
            graph[graph==0] = inf
            graph_list_ori.append(graph)
            seed_node_list.append(seed_node)
            node_val_list.append(node_val)
            next_weight_list.append(weight)
            next_node_list.append(next_node)

        graph_list = tf.cast(tf.stack(graph_list_ori, axis=0), tf.int64)
        seed_node_list = tf.cast(tf.stack(seed_node_list, axis=0), tf.int64)
        node_val_list = tf.cast(tf.stack(node_val_list, axis=0), tf.int64)
        
        weight_mt, next_node_mt = mst(graph_list, seed_node_list, node_val_list)
        for i in range(len(next_weight_list)):
            seq_true = next_weight_list[i]
            
            seq_pred = weight_mt[i,:]
            b_flg = False
            for ind in range(seq_pred.shape[-1]):
                if seq_pred[ind]==inf:
                    b_flg = True
                    break
            if b_flg:
                seq_pred = seq_pred[:ind]
            
            test_acc.update_state(tf.reduce_sum(seq_true), tf.reduce_sum(seq_pred))
        print("Acc: {}".format(test_acc.result()))
                                   

