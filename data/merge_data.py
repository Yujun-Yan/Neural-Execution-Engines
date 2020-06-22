from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import subprocess
def join(seq):
    '''
    seq = [seq1 end seq2 end]
    this funtion will return the merged results of two sorted arrays and also the pointer manipulation traces
    '''
    seql = len(seq)
    first_l = np.int64(np.floor(seql/2))
    seq_0 = seq[:first_l]
    seq_1 = seq[first_l:]
    seq_0 = np.concatenate([seq_0, [np.Infinity]])
    seq_1 = np.concatenate([seq_1, [np.Infinity]])
    ptr_0 = 0
    ptr_1 = 0
    i = 0
    output_seq = []
    mask_ind = np.zeros((seql+1, 2))
    mask_ind[i,:] = [ptr_0, ptr_1]
    pos_list = []
    while not ((ptr_0==first_l) & (ptr_1==seql-first_l)):
        if seq_0[ptr_0] <= seq_1[ptr_1]:
            output_seq.append(seq_0[ptr_0])
            pos_list.append(ptr_0)
            ptr_0 += 1
        else:
            output_seq.append(seq_1[ptr_1])
            pos_list.append(ptr_1 + first_l +1)
            ptr_1 += 1
        i += 1
        mask_ind[i,:] = [ptr_0, ptr_1]
    return output_seq, pos_list, mask_ind
def merge_data_gen(current_dir, reload_from_dir, reload_dir, num_max, Train_SIZE, Train_small_size, Val_size, Val_small_size, state_size, seq_len_1, seq_len):
    '''
    current_dir: string, current checkpoint and data will be saved in this directory
    reload_from_dir: bool, whether to reload
    reload_dir: string, reload directory
    num_max: int, maximum number allowed
    Train_size: int, number of training data
    Train_small_size: int, number of difficult samples in the training data
    Val_size: int, number of validation data
    Val_small_size: int ,number of difficult samples in the validation data
    state_size: int, how many intermediate states when doing merge sort
    seq_len_1: int, length of first sequence
    seq_len: int, length of the whole sequence
    
    '''
    
    
    
    if not reload_from_dir:
        def create_seq(seq_len_1, dct):
            turn0 = False
            seq0 = []
            seq1 = []
            mx = np.random.choice(list(dct.keys()), 1)[0]
            seq0.append(mx)
            while (len(seq0)<seq_len_1) & (len(seq1)<seq_len_1):
                nxt = np.random.choice(dct[mx], 1)[0]
                if turn0:
                    seq0.append(nxt)
                else:
                    seq1.append(nxt)
                if nxt>mx:
                    mx = nxt
                    turn0 = not turn0
            if (len(seq0)==seq_len_1) & (len(seq1)==seq_len_1):
                return np.array(seq0+seq1)
            elif (len(seq0)==seq_len_1):
                return np.concatenate([seq0, seq1, np.random.choice(dct[mx], seq_len_1-len(seq1))])
            else:
                return np.concatenate([seq0, np.random.choice(dct[mx], seq_len_1-len(seq0)), seq1])
        
        def add_to_dict(ele_1, ele_2, dct):
            if ele_1 not in dct:
                dct[ele_1] = [ele_2]
            else:
                dct[ele_1].append(ele_2)
                
            if ele_2 not in dct:
                dct[ele_2] = [ele_1]
            else:
                dct[ele_2].append(ele_1)
        
        def from_ind_to_seq(num, dct, dct_small):
            ele_1 = np.int64(next(x for x, val in enumerate(end_ind) if val >= num+1))
            ele_2 = num_max-(end_ind[ele_1]-num)
            add_to_dict(ele_1, ele_2, dct)
            if np.absolute(ele_1-ele_2)<= 6:
                add_to_dict(ele_1, ele_2, dct_small)
        
        
        possible_combinations = np.arange(num_max, 0, -1, dtype='int64')
        end_ind = np.cumsum(possible_combinations)
        Train_val_ind = np.random.permutation(end_ind[-1])
        split_ind = np.int64(end_ind[-1]*0.7)
        Train_ind = Train_val_ind[:split_ind]
        Val_ind = Train_val_ind[split_ind:]
        train_dict = dict()
        train_small = dict()
        val_dict = dict()
        val_small = dict()
        
        for tid in Train_ind:
            from_ind_to_seq(tid, train_dict, train_small)
        for vid in Val_ind:
            from_ind_to_seq(vid, val_dict, val_small)
        
        Train_exmp = np.zeros((Train_SIZE, seq_len), dtype='int64')
        Train_out = np.zeros_like(Train_exmp)
        pos_train = np.zeros_like(Train_exmp)
        mask_train = np.zeros((Train_exmp.shape[0], state_size, 2), dtype='int64')
        
        Train_exmp_small = np.zeros((Train_small_size, seq_len), dtype='int64')
        Train_out_small = np.zeros_like(Train_exmp_small)
        pos_train_small = np.zeros_like(Train_exmp_small)
        mask_train_small = np.zeros((Train_exmp_small.shape[0], state_size, 2), dtype='int64')
        
        
        Val_exmp = np.zeros((Val_size, seq_len), dtype='int64')
        Val_out = np.zeros_like(Val_exmp, dtype='int64')
        pos_val = np.zeros_like(Val_exmp, dtype='int64')
        mask_val = np.zeros((Val_exmp.shape[0], state_size, 2), dtype='int64')
        
        Val_exmp_small = np.zeros((Val_small_size, seq_len), dtype='int64')
        Val_out_small = np.zeros_like(Val_exmp_small)
        pos_val_small = np.zeros_like(Val_exmp_small)
        mask_val_small = np.zeros((Val_exmp_small.shape[0], state_size, 2), dtype='int64')
        
        for i in range(Train_SIZE):
            seq = create_seq(seq_len_1, train_dict)
            out_seq, pos_list, mask_ind = join(seq)
            Train_exmp[i,:] = seq
            Train_out[i,:] = out_seq
            pos_train[i,:] = pos_list
            mask_train[i,:,:] = mask_ind
        
        for i in range(Train_small_size):
            seq = create_seq(seq_len_1, train_small)
            out_seq, pos_list, mask_ind = join(seq)
            Train_exmp_small[i,:] = seq
            Train_out_small[i,:] = out_seq
            pos_train_small[i,:] = pos_list
            mask_train_small[i,:,:] = mask_ind
        
        
        for i in range(Val_size):
            seq = create_seq(seq_len_1, val_dict)
            out_seq, pos_list, mask_ind = join(seq)
            Val_exmp[i,:] = seq
            Val_out[i,:] = out_seq
            pos_val[i,:] = pos_list
            mask_val[i,:,:] = mask_ind
        
        for i in range(Val_small_size):
            seq = create_seq(seq_len_1, val_small)
            out_seq, pos_list, mask_ind = join(seq)
            Val_exmp_small[i,:] = seq
            Val_out_small[i,:] = out_seq
            pos_val_small[i,:] = pos_list
            mask_val_small[i,:,:] = mask_ind
        
        Train_exmp = np.concatenate([Train_exmp_small, Train_exmp], axis=0)
        Train_out = np.concatenate([Train_out_small, Train_out], axis=0)
        pos_train = np.concatenate([pos_train_small, pos_train], axis=0)
        mask_train = np.concatenate([mask_train_small, mask_train], axis=0)
        
        Val_exmp = np.concatenate([Val_exmp_small, Val_exmp], axis=0)
        Val_out = np.concatenate([Val_out_small, Val_out], axis=0)
        pos_val = np.concatenate([pos_val_small, pos_val], axis=0)
        mask_val = np.concatenate([mask_val_small, mask_val], axis=0)
        
        np.save("{}Train_exmp".format(current_dir), Train_exmp)
        np.save("{}Train_out".format(current_dir), Train_out)
        np.save("{}pos_train".format(current_dir), pos_train)
        np.save("{}mask_train".format(current_dir), mask_train)
        
        np.save("{}Val_exmp".format(current_dir), Val_exmp)
        np.save("{}Val_out".format(current_dir), Val_out)
        np.save("{}pos_val".format(current_dir), pos_val)
        np.save("{}mask_val".format(current_dir), mask_val)
        
    
    else:
        Train_exmp = np.load("{}Train_exmp.npy".format(reload_dir))
        Train_out = np.load("{}Train_out.npy".format(reload_dir))
        pos_train = np.load("{}pos_train.npy".format(reload_dir))
        mask_train = np.load("{}mask_train.npy".format(reload_dir))
        
        Val_exmp = np.load("{}Val_exmp.npy".format(reload_dir))
        Val_out = np.load("{}Val_out.npy".format(reload_dir))
        mask_val = np.load("{}mask_val.npy".format(reload_dir))
        pos_val = np.load("{}pos_val.npy".format(reload_dir))
        

    print("Train example:")
    print(Train_exmp[:3,:])
    print("Mask train:")
    print(mask_train[:3,:,:])
    print("Train out:")
    print(Train_out[:3,:])
    print("Pos train:")
    print(pos_train[:3,:])
    print("Val example:")
    print(Val_exmp[:3,:])
    print("Mask val:")
    print(mask_val[:3,:,:])
    print("Val out:")
    print(Val_out[:3,:])
    print("Pos val")
    print(pos_val[:3,:])    

    Train_dataset = tf.data.Dataset.from_tensor_slices((Train_exmp, mask_train, Train_out, pos_train))    
    Val_dataset = tf.data.Dataset.from_tensor_slices((Val_exmp, mask_val, Val_out, pos_val))
    return Train_dataset, Val_dataset, Train_exmp.shape[0]
