from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import subprocess

def sel_sort_data_gen(current_dir, reload_from_dir_2, reload_dir_2, num_max_2, Train_SIZE, Val_size, train_sub_size, val_sub_size, rep_num, seq_len_p2, state_size_2, end_token, inf):
    '''
    current_dir: string, current checkpoint and data will be saved in this directory
    reload_from_dir_2: bool, whether to reload
    reload_dir_2: string, reload directory
    num_max_2: int, maximum number allowed
    Train_SIZE: int, number of training data
    train_sub_size: int, number of difficult samples in the training data
    Val_size: int, number of validation data
    val_sub_size: int ,number of difficult samples in the validation data
    rep_num: int, repeat number, repeat hard examples but permutated differently
    state_size_2: int, how many intermediate states when doing merge sort
    seq_len_p2: int, length of the whole sequence
    end_token: int, number to represent end_token
    inf: int, number to represent infinity
    '''
    if not reload_from_dir_2:
        probs = np.array([1.0]*num_max_2+[2.0]*2)
        probs = probs/np.sum(probs)
        Train_exmp = np.random.choice(list(range(num_max_2))+[end_token]+[inf], size=(Train_SIZE, seq_len_p2), p=probs)
        Val_exmp = np.random.choice(list(range(num_max_2))+[end_token]+[inf], (Val_size, seq_len_p2), p=probs)
        train_special = []
        val_special = []
        for i in range(4):
            step = i + 1
            start_tr_val = np.random.choice(range(num_max_2-(seq_len_p2-1)*step),(train_sub_size + val_sub_size, 1), replace=False)
            tr_val = np.tile(start_tr_val, [1, seq_len_p2])
            tr_val += np.arange(0, seq_len_p2*step, step)
            train_special.append(tr_val[:train_sub_size, :])
            val_special.append(tr_val[train_sub_size:, :])
        train_special = np.concatenate(train_special, axis=0)
        train_special = np.tile(train_special, (rep_num, 1))
        val_special = np.concatenate(val_special, axis=0)
        val_special = np.tile(val_special, (rep_num, 1))
        np.apply_along_axis(np.random.shuffle, 1, train_special)
        np.apply_along_axis(np.random.shuffle, 1, val_special)
        Train_exmp = np.concatenate([Train_exmp, train_special], axis=0)
        Val_exmp = np.concatenate([Val_exmp, val_special], axis=0)
        Sorted_train = np.sort(Train_exmp)
        Sorted_train_ind = np.argsort(Train_exmp)
        train_mask = np.zeros((Train_exmp.shape[0], state_size_2, seq_len_p2), dtype=np.float32)
        for i in range(1, state_size_2):
            train_mask[np.arange(Train_exmp.shape[0]), i, Sorted_train_ind[:,i-1]] = 1  
        Sorted_val = np.sort(Val_exmp)
        Sorted_val_ind = np.argsort(Val_exmp)
        val_mask = np.zeros((Val_exmp.shape[0], state_size_2, seq_len_p2), dtype=np.float32)
        for i in range(1, state_size_2):
            val_mask[np.arange(Val_exmp.shape[0]), i, Sorted_val_ind[:,i-1]] = 1
        np.save("{}Train_exmp".format(current_dir), Train_exmp)
        np.save("{}train_mask".format(current_dir), train_mask)
        np.save("{}Sorted_train".format(current_dir), Sorted_train)
        np.save("{}Val_exmp".format(current_dir), Val_exmp)
        np.save("{}val_mask".format(current_dir), val_mask)
        np.save("{}Sorted_val".format(current_dir), Sorted_val)
    else:
        Train_exmp = np.load("{}Train_exmp.npy".format(reload_dir_2))
        Val_exmp = np.load("{}Val_exmp.npy".format(reload_dir_2))
        train_mask = np.load("{}train_mask.npy".format(reload_dir_2))
        val_mask = np.load("{}val_mask.npy".format(reload_dir_2))
        Sorted_train = np.load("{}Sorted_train.npy".format(reload_dir_2))
        Sorted_val = np.load("{}Sorted_val.npy".format(reload_dir_2))

    print("Train_example:")
    print(Train_exmp[:3,:])
    print("train_mask:")
    print(train_mask[:3,:])
    print("Sorted_train:")
    print(Sorted_train[:3])
    print("Val_example:")
    print(Val_exmp[:3,:])
    print("val_mask:")
    print(val_mask[:3,:])
    print("Sorted_val:")
    print(Sorted_val[:3])

    Train_dataset_2 = tf.data.Dataset.from_tensor_slices((Train_exmp, train_mask, Sorted_train))    
    Val_dataset_2 = tf.data.Dataset.from_tensor_slices((Val_exmp, val_mask, Sorted_val))
    
    return Train_dataset_2, Val_dataset_2, Train_exmp.shape[0] 