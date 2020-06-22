from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import subprocess

def add_holdout_data(current_dir, reload_from_dir_1, reload_dir_1, num_max_1, hld, Train_size, Val_size, end_token):
    '''
    current_dir: string, current checkpoint and data will be saved in this directory
    reload_from_dir_1: bool, whether to reload
    reload_dir_1: string, reload directory
    num_max_1: int, maximum number allowed
    hld: int, number of holdout
    Train_size: int, number of training data
    Val_size: int, number of validation data
    end_token: int, a number used to represent infinity
    
    '''
    if not reload_from_dir_1:
        holdout_number = np.random.choice(range(num_max_1), hld, replace=False)   # list of numbers that are held out
        train_number = np.setdiff1d(range(num_max_1+1), holdout_number) ### include inf
        
        def create_examples(num_set_1, num_set_2, count):
            i = 0
            add_list = []
            data_list = []
            while i<count:
                first_half = np.random.choice(num_set_1, (1000)) # first operand
                second_half = np.random.choice(num_set_2, (1000)) # second operand
                inf_id = np.logical_or((first_half==end_token), (second_half==end_token)) ########## infinity location
                addition = first_half + second_half
                addition[inf_id] = end_token    #### insure num+inf=inf
                keep_id = addition<num_max_1
                keep_id = np.logical_or(inf_id, keep_id)
                first_half = first_half[keep_id]
                second_half = second_half[keep_id]
                addition = addition[keep_id]
                data_list.append(np.stack([first_half,second_half], axis=1))
                add_list.append(addition)
                i += first_half.shape[0]
            data = np.concatenate(data_list, axis=0)
            addition = np.concatenate(add_list, axis=0)
            return data[:count,:], addition[:count]
                
        Train_exmp, Train_add_out = create_examples(train_number, train_number, Train_size)
        Val_exmp, Val_add_out = create_examples(range(num_max_1+1), holdout_number, Val_size)
        np.apply_along_axis(np.random.shuffle, 1, Train_exmp)
        np.apply_along_axis(np.random.shuffle, 1, Val_exmp)
        
        
        Train_exmp = np.int64(Train_exmp)
        Val_exmp = np.int64(Val_exmp)
        Train_add_out = np.int64(Train_add_out)
        Val_add_out = np.int64(Val_add_out)
        
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