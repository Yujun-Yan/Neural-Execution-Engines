from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import subprocess

def mul_data(current_dir, reload_from_dir_1, reload_dir_1, max_num, max_prod, Train_size, Val_size, binary_size):
    '''
    current_dir: string, current checkpoint and data will be saved in this directory
    reload_from_dir_1: bool, whether to reload
    reload_dir_1: string, reload directory
    max_num: int, maximum number allowed
    max_prod: int, maximum product
    Train_size: int, number of training data
    Val_size: int, number of validation data
    binary_size: int, number of bits to represnt an integer
    '''
    if not reload_from_dir_1:
        sm = np.arange(1, max_num+1)
        zero_com = [2 ** binary_size] # 0-max_prod
        possible_combinations = np.int64(np.concatenate([zero_com, np.floor(max_prod/sm)-sm+1]))
        end_ind = np.cumsum(possible_combinations)
        Train_val_ind = np.random.choice(range(end_ind[-1]),(Train_size + Val_size), replace=False)
        Train_ind = Train_val_ind[:Train_size]
        Val_ind = Train_val_ind[Train_size:]
        
        def from_ind_to_seq(num):
            ele_1 = np.int64(next(x for x, val in enumerate(end_ind) if val >= num+1))
            ele_2 = num-np.concatenate([[0], end_ind])[ele_1]+ele_1
            return np.array([ele_1, ele_2])
        Train_exmp = np.zeros((Train_size, 2), dtype='int64')
        Val_exmp = np.zeros((Val_size, 2), dtype='int64')
        for i, num in enumerate(Train_ind):
            Train_exmp[i,:] = from_ind_to_seq(num)
        Train_exmp = np.concatenate([Train_exmp, np.fliplr(Train_exmp)])
        for i, num in enumerate(Val_ind):
            Val_exmp[i,:] = from_ind_to_seq(num)
        Val_exmp = np.concatenate([Val_exmp, np.fliplr(Val_exmp)])
        Train_mul_out = np.prod(Train_exmp, -1)
        Val_mul_out = np.prod(Val_exmp, -1)
        Train_seed = 1*np.ones((Train_exmp.shape[0]), dtype='int64')
        Val_seed = 1*np.ones((Val_exmp.shape[0]), dtype='int64')
        np.save("{}Train_exmp".format(current_dir), Train_exmp)
        np.save("{}Train_seed".format(current_dir), Train_seed)
        np.save("{}Train_mul_out".format(current_dir), Train_mul_out)
        np.save("{}Val_exmp".format(current_dir), Val_exmp)
        np.save("{}Val_seed".format(current_dir), Val_seed)
        np.save("{}Val_mul_out".format(current_dir), Val_mul_out)
        
    else:
        Train_exmp = np.load("{}Train_exmp.npy".format(reload_dir_1))
        Train_seed = np.load("{}Train_seed.npy".format(reload_dir_1))
        Train_mul_out = np.load("{}Train_mul_out.npy".format(reload_dir_1))
        Val_exmp = np.load("{}Val_exmp.npy".format(reload_dir_1))
        Val_seed = np.load("{}Val_seed.npy".format(reload_dir_1))
        Val_mul_out = np.load("{}Val_mul_out.npy".format(reload_dir_1))

    print("Train example:")
    print(Train_exmp[:3,:])
    print("Train mul out:")
    print(Train_mul_out[:3])
    print("Train seed:")
    print(Train_seed[:3])
    print("Val example:")
    print(Val_exmp[:3,:])
    print("Val mul out:")
    print(Val_mul_out[:3])
    print("Val seed:")
    print(Val_seed[:3])
    Train_dataset_1 = tf.data.Dataset.from_tensor_slices((Train_exmp, Train_seed, Train_mul_out))
    Val_dataset_1 = tf.data.Dataset.from_tensor_slices((Val_exmp, Val_seed, Val_mul_out))
    return Train_dataset_1, Val_dataset_1, Train_exmp.shape[0]