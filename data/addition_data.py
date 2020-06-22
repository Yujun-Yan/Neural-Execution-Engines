from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import subprocess

def add_data(current_dir, reload_from_dir_1, reload_dir_1, num_max_1, Train_size, Val_size, end_token, binary_size):
    '''
    current_dir: string, current checkpoint and data will be saved in this directory
    reload_from_dir_1: bool, whether to reload
    reload_dir_1: string, reload directory
    num_max_1: int, maximum number allowed
    Train_size: int, number of training data
    Val_size: int, number of validation data
    end_token: int, a number used to represent infinity
    binary_size: int, number of bits to represnt an integer
    
    '''
    if not reload_from_dir_1:
        ##### order all possible pairs to ensure no pairs in the test will be seen in the training
        inf_com = [num_max_1+1]
        possible_combinations = np.concatenate([np.arange(num_max_1, 0, -2, dtype='int64'), inf_com], -1)  ##### possible_combinations[i] counts number of (i,j), j>=i
        end_ind = np.cumsum(possible_combinations) ### starting indices for each group
        Train_val_ind = np.random.choice(range(end_ind[-1]),(Train_size + Val_size), replace=False)
        Train_ind = Train_val_ind[:Train_size]
        Val_ind = Train_val_ind[Train_size:]
        
        def from_ind_to_seq(num):
            ele_1 = np.int64(next(x for x, val in enumerate(end_ind) if val >= num+1))
            ele_2 = num_max_1-ele_1-(end_ind[ele_1]-num)
            if ele_1 == 2**(binary_size-1):
                ele_2 = num_max_1 + 1 - (end_ind[ele_1]-num)
                ele_1 = end_token
            return np.array([ele_1, ele_2])
        Train_exmp = np.zeros((Train_size, 2), dtype='int64')
        Val_exmp = np.zeros((Val_size, 2), dtype='int64')
        for i, num in enumerate(Train_ind):
            Train_exmp[i,:] = from_ind_to_seq(num)
        Train_exmp = np.concatenate([Train_exmp, np.fliplr(Train_exmp)])
        for i, num in enumerate(Val_ind):
            Val_exmp[i,:] = from_ind_to_seq(num)
        Val_exmp = np.concatenate([Val_exmp, np.fliplr(Val_exmp)])
        Train_add_out = np.sum(Train_exmp, -1)
        Train_add_out[Train_add_out>end_token] = end_token
        Val_add_out = np.sum(Val_exmp, -1)
        Val_add_out[Val_add_out>end_token] = end_token
        Train_seed = 1*np.ones((Train_exmp.shape[0]), dtype='int64')
        Val_seed = 1*np.ones((Val_exmp.shape[0]), dtype='int64')
        np.save("{}Train_exmp".format(current_dir), Train_exmp)
        np.save("{}Train_seed".format(current_dir), Train_seed)
        np.save("{}Train_add_out".format(current_dir), Train_add_out)
        np.save("{}Val_exmp".format(current_dir), Val_exmp)
        np.save("{}Val_seed".format(current_dir), Val_seed)
        np.save("{}Val_add_out".format(current_dir), Val_add_out)
        
    else:
        Train_exmp = np.load("{}Train_exmp.npy".format(reload_dir_1))
        Train_seed = np.load("{}Train_seed.npy".format(reload_dir_1))
        Train_add_out = np.load("{}Train_add_out.npy".format(reload_dir_1))
        Val_exmp = np.load("{}Val_exmp.npy".format(reload_dir_1))
        Val_seed = np.load("{}Val_seed.npy".format(reload_dir_1))
        Val_add_out = np.load("{}Val_add_out.npy".format(reload_dir_1))
    print("Train example:")
    print(Train_exmp[:3,:])
    print("Train add out:")
    print(Train_add_out[:3])
    print("Train seed:")
    print(Train_seed[:3])
    print("Val example:")
    print(Val_exmp[:3,:])
    print("Val add out:")
    print(Val_add_out[:3])
    print("Val seed:")
    print(Val_seed[:3])
    Train_dataset_1 = tf.data.Dataset.from_tensor_slices((Train_exmp, Train_seed, Train_add_out))
    Val_dataset_1 = tf.data.Dataset.from_tensor_slices((Val_exmp, Val_seed, Val_add_out))
    return Train_dataset_1, Val_dataset_1, Train_exmp.shape[0]