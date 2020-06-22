from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.append('../data/')
sys.path.append('../model/')
sys.path.append('../')
from merge_data import *
from model import *
from utils import *

def run_merge(current_dir, reload_from_dir, reload_dir, checkpoint_path_data, checkpoint_path_msk):
    binary_size = 8
    d_model = 16

    seq_len_1 = 8
    seq_len = seq_len_1*2

    Train_SIZE = 5000 
    Train_small_size = 500
    BATCH_SIZE = 64
    Val_size = 500
    Val_small_size = 50
    num_filters = 16
    filter_size = 3


    num_max = 2**binary_size
    end_token = 2 ** binary_size
    pad_token = 2 ** binary_size + 1

    state_size = seq_len+1
    num_layers = 6
    dff = 128

    target_vocab_size = binary_size + 1
    dropout_rate = 0.1
    
    if reload_from_dir:
        EPOCHS = 1
    else:
        EPOCHS = 100
    res_ratio = 1.5

    out_num = True
    out_pos = True
    assert(out_num or out_pos)
    pos_enc = 3

    discount = 0.005
    make_sym = True
    USE_positioning = False
    with open("{}parameters.txt".format(current_dir), 'w') as fi:
        fi.write("binary_size: {}\nd_model: {}\nseq_len_1: {}\nseq_len: {}\nTrain_SIZE: {}\nTrain_small_size: {}\nBATCH_SIZE: {}\nVal_size: {}\nVal_small_size: {}\nnum_filters: {}\nfilter_size: {}\nnum_max: {}\nnum_layers: {}\ndff: {}\ntarget_vocab_size: {}\ndropout_rate: {}\nmake_sym: {}\nEPOCHS: {}\nres_ratio: {}\nout_num: {}\nout_pos: {}\nUSE_positioning: {}\npos_enc: {}\ndiscount: {}".format(binary_size, d_model, seq_len_1, seq_len, Train_SIZE, Train_small_size, BATCH_SIZE, Val_size, Val_small_size, num_filters, filter_size, num_max, num_layers, dff, target_vocab_size, dropout_rate, make_sym, EPOCHS, res_ratio, out_num, out_pos, USE_positioning, pos_enc, discount) + reload_from_dir * "\nreload_dir: {}".format(reload_dir))
    
    Train_dataset, Val_dataset,  Texmp_size = merge_data_gen(current_dir, reload_from_dir, reload_dir, num_max, Train_SIZE, Train_small_size, Val_size, Val_small_size, state_size, seq_len_1, seq_len)
    def encode(tr, mask, srt, pos_list):
        seq_l = np.int64(np.floor(len(tr)/2))
        tr = np.hstack((tr[:seq_l], end_token, tr[seq_l:], end_token))
        srt = np.hstack((srt, end_token))
        pos_list = np.hstack((pos_list, seq_l))
        return tr, mask, srt, pos_list
    
    def tf_encode(tr, mask, srt, pos_list):
        return tf.py_function(encode, [tr, mask, srt, pos_list], [tf.int64, tf.int64, tf.int64, tf.int64])

    def encode_wo_mask(tr, srt):
        seq_l = np.int64(np.floor(len(tr)/2))
        tr = np.hstack((tr[:seq_l], end_token, tr[seq_l:], end_token))
        srt = np.hstack((srt, end_token))
        return tr, srt

    def tf_encode_wo_mask(tr, srt):
        return tf.py_function(encode_wo_mask,[tr, srt], [tf.int64, tf.int64])

    train_dataset = Train_dataset.map(tf_encode)
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(Texmp_size).batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    exmp = next(iter(train_dataset))
    print(exmp[0][0,:])
    print(exmp[1][0,:,:])
    print(exmp[2][0,:])
    print(exmp[3][0,:])

    val_dataset = Val_dataset.map(tf_encode)
    val_dataset = val_dataset.batch(BATCH_SIZE)

    exmp = next(iter(val_dataset))
    print(exmp[0][0,:])
    print(exmp[1][0,:,:])
    print(exmp[2][0,:])
    print(exmp[3][0,:])
    
    transformer = Transformer(num_layers, d_model, binary_size+1, dff, pos_enc, target_vocab_size, make_sym, USE_positioning, out_num, out_pos, res_ratio, dropout_rate)
    msk_transform = mask_transform(num_filters, filter_size, dropout_rate)
    
    learning_rate_data = CustomSchedule(d_model)
    learning_rate_msk = CustomSchedule(num_filters)

    optimizer_data = tf.keras.optimizers.Adam(learning_rate_data, beta_1=0.9, beta_2=0.98, 
                                        epsilon=1e-9)

    optimizer_msk = tf.keras.optimizers.Adam(learning_rate_msk, beta_1=0.9, beta_2=0.98, 
                                        epsilon=1e-9)
    
    ckpt_data = tf.train.Checkpoint(transformer=transformer,
                           optimizer_data=optimizer_data)
    ckpt_msk = tf.train.Checkpoint(msk_transform=msk_transform,
                           optimizer_msk=optimizer_msk)


    ckpt_manager_data = tf.train.CheckpointManager(ckpt_data, checkpoint_path_data, max_to_keep=5)
    ckpt_manager_msk = tf.train.CheckpointManager(ckpt_msk, checkpoint_path_msk, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager_data.latest_checkpoint:
        ckpt_data.restore(ckpt_manager_data.latest_checkpoint)
        print ('Model_data checkpoint restored!!')

    if ckpt_manager_msk.latest_checkpoint:
        ckpt_msk.restore(ckpt_manager_msk.latest_checkpoint)
        print ('Model_msk checkpoint restored!!')
    
    train_data_loss = tf.keras.metrics.Mean(name='train_data_loss')
    train_data_content_loss = tf.keras.metrics.Mean(name='train_data_content_loss')
    train_data_pos_loss = tf.keras.metrics.Mean(name='train_data_pos_loss')
    train_msk_loss = tf.keras.metrics.Mean(name='train_msk_loss')
    train_data_accuracy = tf.keras.metrics.Accuracy(name='train_data_accuracy')
    train_msk_accuracy = tf.keras.metrics.Accuracy(name='train_msk_accuracy')
    d_loss = tf.keras.metrics.Mean(name='d_loss')
    d_accuracy = tf.keras.metrics.Accuracy(name='d_accuracy')
    
    def create_masks(mask, seq_1, seq_2):
        mask_first = mask[:,:,0]
        mask_second = mask[:,:,1]
        batch_size = mask.shape[0]
        state_size = mask.shape[1]
        emb_1 = tf.one_hot(mask_first, seq_1+1, axis=-1)
        emb_2 = tf.one_hot(mask_second, seq_2+1, axis=-1)
        enc_padding_mask = tf.concat([emb_1, emb_2], -1)
        
        combined_mask = None
        enc_padding_mask = 1-enc_padding_mask
        enc_padding_mask = enc_padding_mask[:,:,tf.newaxis,:]
        dec_padding_mask = enc_padding_mask
        return enc_padding_mask, combined_mask, dec_padding_mask
    
    summary_writer = tf.summary.create_file_writer(current_dir + 'logs')
    @tf.function
    def train_step_data(inp, mask, tar, pos, seq_1, seq_2):
        batch_size = mask.shape[0]
        state_size = mask.shape[-2]
        enc_inp = tf.tile(inp[:, tf.newaxis, :], [1, state_size, 1])
        dec_inp = tf.zeros((batch_size, state_size, 1), dtype=tf.int64)
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(mask, seq_1, seq_2)
        chg_mask = tf.one_hot(pos, seq_1+seq_2+2)
        with tf.GradientTape() as tape:
            predictions,_,predicted_pos = transformer(enc_inp, dec_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
            predictions = tf.squeeze(predictions, -2)
            weights = tf.concat([tf.ones((tar.shape[0], seq_1+seq_2)), tf.ones((tar.shape[0], 1))*discount],-1)
            loss_content = loss_function(binary_encoding(tar, binary_size+1), predictions, weights)
            loss_position = loss_pos(chg_mask, predicted_pos, weights)
            loss = loss_content + loss_position
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer_data.apply_gradients(zip(gradients, transformer.trainable_variables))
        train_data_loss(loss)
        train_data_content_loss(loss_content)
        train_data_pos_loss(loss_position)    
        tf.summary.scalar("data_loss", train_data_loss.result(), step=optimizer_data.iterations)
        tf.summary.scalar("data_content_loss", train_data_content_loss.result(), step=optimizer_data.iterations)
        tf.summary.scalar("data_pos_loss", train_data_pos_loss.result(), step=optimizer_data.iterations)
        pred_binary = tf.cast(tf.greater(predictions, 0), tf.int64)
        pred_binary = back2int(pred_binary)
        train_data_accuracy(tar, pred_binary)
        return tf.transpose(enc_padding_mask,[0,1,3,2]), chg_mask[:,:,:,tf.newaxis] 
    
    @tf.function
    def train_step_msk(init_msk, chg_msk):
        x = tf.concat([init_msk, chg_msk], axis=-1)
        ######## exclude the last one ###############
        x = x[:,:-1,:,:]
        tar_msk = init_msk[:,1:,:,0]
        x = tf.reshape(x,[-1, x.shape[-2], x.shape[-1]])
        x = 2*x-1
        with tf.GradientTape() as tape:
            predict_msk = msk_transform(x, True) #### batch_size*seq_len, seq_len
            predict_msk = tf.reshape(predict_msk, (tar_msk.shape[0], tar_msk.shape[-2], tar_msk.shape[-1]))
            loss_msk = loss_function(tar_msk, predict_msk)
        gradients = tape.gradient(loss_msk, msk_transform.trainable_variables)
        optimizer_msk.apply_gradients(zip(gradients, msk_transform.trainable_variables))
        train_msk_loss(loss_msk)
        tf.summary.scalar("msk_loss", train_msk_loss.result(), step=optimizer_msk.iterations)
        predict_msk_binary = tf.cast(tf.greater(predict_msk, 0), tf.float32)
        err = tf.reduce_sum(tar_msk-predict_msk_binary, axis=-1)
        train_msk_accuracy(err, tf.zeros_like(err))
    
    
    def eval_val(state_size, dataset, seq_1, seq_2, name='Validation', include_pos_loss=True):
        d_loss.reset_states()
        d_accuracy.reset_states()
        for element in dataset:
            if include_pos_loss:
                inp, mask, tar, _ = element
                msk_tar, _, _ = create_masks(mask, seq_1, seq_2)
            else:
                inp, tar = element
            enc_inp = inp[:, tf.newaxis, :]
            batch_size = enc_inp.shape[0]
            dec_inp = tf.zeros((batch_size, 1, 1), dtype=tf.int64)
            
            enc_padding_mask = tf.concat([tf.ones((batch_size, 1, 1)), tf.zeros((batch_size, 1, seq_1)), tf.ones((batch_size, 1, 1)), tf.zeros((batch_size, 1, seq_2))], -1)
            enc_padding_mask = 1-enc_padding_mask[:,:,tf.newaxis,:]
            dec_padding_mask = enc_padding_mask
            combined_mask = None
            
            if include_pos_loss:
                mask_list = []
            out_list = []
            for i in range(state_size):
                predictions,_,predicted_pos = transformer(enc_inp, dec_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)
                out_list.append(tf.squeeze(predictions, -2))
                ########### replace argmax (uncertain) ############
                predicted_pos_max = tf.reduce_max(predicted_pos, axis=-1, keepdims=True)
                predicted_pos_max = tf.equal(predicted_pos, predicted_pos_max)
                predicted_pos_ind = tf.reshape(predicted_pos_max, [-1, predicted_pos_max.shape[-1]])
                predicted_pos_ind = tf.where(predicted_pos_ind)
                pos_id = tf.cast(tf.math.segment_min(predicted_pos_ind[:,1],predicted_pos_ind[:,0]), tf.int64)
                pos_id = tf.reshape(pos_id, [predicted_pos_max.shape[0], predicted_pos_max.shape[1]])
                
                chg_msk = tf.one_hot(pos_id, seq_1+seq_2+2)[:,:,:,tf.newaxis]
                init_msk = tf.transpose(enc_padding_mask,[0,1,3,2])
                x = tf.concat([init_msk,chg_msk],axis=-1)
                x = tf.reshape(x,[-1, x.shape[-2], x.shape[-1]])
                x = 2*x-1
                predict_msk = msk_transform(x, False)
                if include_pos_loss:
                    mask_list.append(predict_msk[:,tf.newaxis,:])
                predict_msk = tf.cast(tf.greater(predict_msk, 0), tf.float32)
                enc_padding_mask = predict_msk[:, tf.newaxis, tf.newaxis, :]
                dec_padding_mask = enc_padding_mask
            if include_pos_loss:
                mask_est = tf.concat(mask_list[:-1], -2)
                loss_position = loss_function(msk_tar[:,1:,0,:], mask_est)
            else:
                loss_position = 0
            out_est = tf.concat(out_list, -2)
            loss_content = loss_function(binary_encoding(tar, binary_size+1), out_est)
            loss = loss_position + loss_content
            d_loss(loss)
            
            out_binary = tf.cast(tf.greater(out_est, 0), tf.int64)
            out_binary = back2int(out_binary)
    #         if not tf.equal(tf.reduce_sum(tar-out_binary),0):
    #             print("inp")
    #             print(inp)
    #             print("tar")
    #             print(tar)
    #             print("pre_out")
    #             print(out_binary)
            d_accuracy(tar, out_binary)
        print('{}_Loss {:.4f} {}_Accuracy {:.4f}'.format(name, d_loss.result(), name, d_accuracy.result()))
        return d_accuracy.result()
    
    for epoch in range(EPOCHS):
        start = time.time()
        train_data_loss.reset_states()
        train_data_content_loss.reset_states()
        train_data_pos_loss.reset_states()
        train_msk_loss.reset_states()
        train_data_accuracy.reset_states()
        train_msk_accuracy.reset_states()
        with summary_writer.as_default():
            for (batch, (inp, mask, tar, pos)) in enumerate(train_dataset):
                init_msk, chg_msk = train_step_data(inp, mask, tar, pos, seq_len_1, seq_len_1)
                train_step_msk(init_msk, chg_msk)
                if batch % 500 == 0:
                    print('Epoch {} Batch {}:\nTraining_Data_Loss {:.4f} Training_Data_Accuracy {:.4f}\nTraining_Msk_Loss {:.4f} Training_Msk_Accuracy {:.4f}'.format(epoch+1, batch, train_data_loss.result(), train_data_accuracy.result(), train_msk_loss.result(), train_msk_accuracy.result()))
                    eval_val(state_size, val_dataset, seq_len_1, seq_len_1)
            if (epoch + 1) % 5 == 0:
                ckpt_save_path_data = ckpt_manager_data.save()
                ckpt_save_path_msk = ckpt_manager_msk.save()
                print("Saving checkpoint for epoch {} at {} and {}".format(epoch + 1, ckpt_save_path_data, ckpt_save_path_msk))
            print('Epoch {} Batch {}:\nTraining_Data_Loss {:.4f} Training_Data_Accuracy {:.4f}\nTraining_Msk_Loss {:.4f} Training_Msk_Accuracy {:.4f}'.format(epoch+1, batch, train_data_loss.result(), train_data_accuracy.result(), train_msk_loss.result(), train_msk_accuracy.result()))
            eval_val(state_size, val_dataset, seq_len_1, seq_len_1)
            print('Time taken for 1 epoch: {} secs\n'.format(time.time()-start))
    
    
    max_seq_len = 100
    max_seq_step = 60
    test_size_random = 60
    test_size_small_steps = 10
    acc = []
    for i in range(9, max_seq_len+1):
        test_random = np.random.choice(range(num_max), (test_size_random, i))
        test_special = []
        if tf.less_equal(i, max_seq_step):
            for step in range(1,5):
                start_tst = np.random.choice(range(num_max-(i-1)*step), (test_size_small_steps, 1), replace=False)
                test = np.tile(start_tst, [1, i])
                test += np.arange(0, i*step, step)
                test_special.append(test)
        else:
            for step in range(1,5):
                start_tst = np.random.choice(range(num_max-(max_seq_step-1)*step), (test_size_small_steps, 1), replace=False)
                test = np.tile(start_tst, [1, max_seq_step])
                test += np.arange(0, max_seq_step*step, step)
                padded_random = np.random.choice(range(num_max), (test_size_small_steps, i-max_seq_step))
                test = np.concatenate((test, padded_random), axis=-1)
                test_special.append(test)
        test_special = np.concatenate(test_special, axis=0)
        np.apply_along_axis(np.random.shuffle, 1, test_special)
        test = np.concatenate([test_random, test_special], axis=0)
        
        test_out = np.zeros_like(test, dtype='int64')
        for j, seq in enumerate(test):
            out_seq, _, _ = join(seq)
            test_out[j,:] = out_seq
        
        test_dataset = tf.data.Dataset.from_tensor_slices((test, test_out))
        test_dataset = test_dataset.map(tf_encode_wo_mask)
        test_dataset = test_dataset.batch(BATCH_SIZE)
        seq_1 = np.int64(np.floor(i/2))
        seq_2 = i-seq_1
        acc.append(eval_val(i+1, test_dataset, seq_1, seq_2, name="Test_seq{}".format(i), include_pos_loss=False)) 
        
        
    plt.figure()
    plt.plot(range(9, 9+len(acc)), acc)
    plt.xlabel('seq_len')
    plt.ylabel('merge_func_acc')
    filename = "merge_func_acc_plot.png"
    plt.savefig(current_dir+filename)
    
    
    def evaluate(inp_sentence, seq_1):
        seq_2 = inp_sentence.shape[-1]-seq_1
        
        inp_sentence = np.hstack((inp_sentence[:seq_1], end_token, inp_sentence[seq_1:], end_token))
        inp_sentence = inp_sentence.astype('int64')
        inp_sentence = tf.convert_to_tensor(inp_sentence)
        
        enc_inp = inp_sentence[tf.newaxis, tf.newaxis, :]
        dec_inp = tf.zeros((1,1,1), dtype=tf.int64)
        enc_padding_mask = tf.concat([tf.ones((1,1,1)), tf.zeros((1, 1, seq_1)), tf.ones((1,1,1)), tf.zeros((1,1,seq_2))],-1)
        enc_padding_mask = 1-enc_padding_mask[:,:,tf.newaxis,:]
        dec_padding_mask = enc_padding_mask
        combined_mask = None
        
        out_list = []
        attention_weights = []
        for i in range(seq_1+seq_2+1): ###### predicted_pos: (1,1,seq_1+seq_2+2)
            predictions, _, predicted_pos = transformer(enc_inp, dec_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)
            attention_weights.append(tf.squeeze(predicted_pos,0))  # (1,seq_1+seq_2+2)
            ############### replace argmax ################################
            out_list.append(tf.squeeze(predictions))
            predicted_pos = tf.squeeze(predicted_pos, 1)
            att_max = tf.reduce_max(predicted_pos, axis=-1, keepdims=True)
            att_max_ind = tf.where(tf.equal(predicted_pos, att_max))
            pos_id = tf.cast(tf.math.segment_min(att_max_ind[:,1],att_max_ind[:,0]), tf.int64)
            pos_id = pos_id[:,tf.newaxis]
            chg_msk = tf.one_hot(pos_id, seq_1+seq_2+2)[:,:,:,tf.newaxis]
            init_msk = tf.transpose(enc_padding_mask,[0,1,3,2])
            x = tf.concat([init_msk,chg_msk],axis=-1)
            x = tf.reshape(x,[-1, x.shape[-2], x.shape[-1]])
            x = 2*x-1
            predict_msk = msk_transform(x, False)
            predict_msk = tf.cast(tf.greater(predict_msk, 0), tf.float32)
            enc_padding_mask = predict_msk[:, tf.newaxis, tf.newaxis, :]
            dec_padding_mask = enc_padding_mask
        
        out_est = tf.stack(out_list, axis=0)
        out_binary = tf.cast(tf.greater(out_est, 0), tf.int64)
        out_binary = back2int(out_binary)   
        return out_binary, tf.concat(attention_weights, -2)
    
    seq_1 = 2
    seq_2 = 2
    random_num = np.random.choice(range(num_max), (seq_1+seq_2))
    print("random num test:")
    print("original_sequence: {}".format(random_num))
    output, _ = evaluate(random_num, seq_1)
    print("predicted_sequence: {}".format(output))
    
    
    def merge_sort(dataset, name='Test'):
        d_loss.reset_states()
        d_accuracy.reset_states()
        for (batch, (inp, tar)) in enumerate(dataset):
            batch_size = inp.shape[0]
            seq_len = inp.shape[-1]
            L = tf.cast(tf.math.ceil(tf.math.log(tf.cast(seq_len, tf.float32))/tf.math.log(2.0)), tf.int64)
            pad_num = tf.pow(2,L)-seq_len
            inp_new = tf.concat([inp[:,:,tf.newaxis], tf.ones([batch_size, seq_len, 1], dtype=tf.int64)*end_token], -1)
            inp_new = tf.concat([inp_new, pad_token*tf.ones([batch_size, pad_num, 2], dtype=tf.int64)], -2)
            for i in range(1, L+1):
                j = tf.pow(2,i)
                inp_res = tf.reshape(inp_new, [-1, j+2])
                batch_size_new = inp_res.shape[0]
                
                enc_inp = inp_res[:, tf.newaxis,:]
                dec_inp = tf.zeros((batch_size_new, 1, 1), dtype=tf.int64)
                
                seq_1 = tf.pow(2,i-1)
                seq_2 = seq_1
                enc_mask = create_padding_mask(inp_res)
                enc_padding_mask = tf.concat([tf.ones((batch_size_new, 1, 1)), tf.zeros((batch_size_new, 1, seq_1)), tf.ones((batch_size_new, 1, 1)), tf.zeros((batch_size_new, 1, seq_2))], -1)
                enc_padding_mask = 1-enc_padding_mask[:,:,tf.newaxis,:]
                enc_padding_mask = tf.maximum(enc_mask, enc_padding_mask)
                
                dec_padding_mask = enc_padding_mask
                combined_mask = None
                
                out_list =[]
                for _ in range(j+1):
                    predictions, _, predicted_pos = transformer(enc_inp, dec_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)
                    out_list.append(tf.squeeze(predictions, -2))
                    ########## compute argmax #################
                    predicted_pos_max = tf.reduce_max(predicted_pos, axis=-1, keepdims=True)
                    predicted_pos_max = tf.equal(predicted_pos, predicted_pos_max)
                    predicted_pos_ind = tf.reshape(predicted_pos_max, [-1, predicted_pos_max.shape[-1]])
                    predicted_pos_ind = tf.where(predicted_pos_ind)
                    pos_id = tf.cast(tf.math.segment_min(predicted_pos_ind[:,1],predicted_pos_ind[:,0]), tf.int64)
                    pos_id = tf.reshape(pos_id, [predicted_pos_max.shape[0], predicted_pos_max.shape[1]])
                    ############################################
                    chg_msk = tf.one_hot(pos_id, seq_1+seq_2+2)[:,:,:,tf.newaxis]
                    init_msk = tf.transpose(enc_padding_mask,[0,1,3,2])
                    x = tf.concat([init_msk,chg_msk],axis=-1)
                    x = tf.reshape(x,[-1, x.shape[-2], x.shape[-1]])
                    x = 2*x-1
                    predict_msk = msk_transform(x, False)
                    predict_msk = tf.cast(tf.greater(predict_msk, 0), tf.float32)
                    enc_padding_mask = predict_msk[:, tf.newaxis, tf.newaxis, :]
                    dec_padding_mask = enc_padding_mask
                inp_new = tf.concat(out_list, -2)
                inp_new = tf.cast(tf.greater(inp_new, 0), tf.int64)
                inp_new = back2int(inp_new)   
                if tf.equal(i, L):
                    d_accuracy(tar, inp_new[:,:seq_len+1])
            print('{}_Accuracy {:.4f}'.format(name, d_accuracy.result()))
            return d_accuracy.result()

    def encode_sort(tr, srt):
        srt = np.hstack((srt, end_token))
        return tr, srt

    def tf_encode_sort(tr, srt):
        return tf.py_function(encode_sort, [tr, srt], [tf.int64, tf.int64])

    max_seq_len = 100
    max_seq_step = 60
    test_size_random = 60
    test_size_small_steps = 10
    acc = []
    for i in range(2, max_seq_len+1):
        test_random = np.random.choice(range(num_max), (test_size_random, i))
        test_special = []
        if tf.less_equal(i, max_seq_step):
            for step in range(1,5):
                start_tst = np.random.choice(range(num_max-(i-1)*step), (test_size_small_steps, 1), replace=False)
                test = np.tile(start_tst, [1, i])
                test += np.arange(0, i*step, step)
                test_special.append(test)
        else:
            for step in range(1,5):
                start_tst = np.random.choice(range(num_max-(max_seq_step-1)*step), (test_size_small_steps, 1), replace=False)
                test = np.tile(start_tst, [1, max_seq_step])
                test += np.arange(0, max_seq_step*step, step)
                padded_random = np.random.choice(range(num_max), (test_size_small_steps, i-max_seq_step))
                test = np.concatenate((test, padded_random), axis=-1)
                test_special.append(test)
        test_special = np.concatenate(test_special, axis=0)
        np.apply_along_axis(np.random.shuffle, 1, test_special)
        test = np.concatenate([test_random, test_special], axis=0)
        
        Sorted_test = np.sort(test)
        test_dataset = tf.data.Dataset.from_tensor_slices((test, Sorted_test))
        test_dataset = test_dataset.map(tf_encode_sort)
        test_dataset = test_dataset.batch(BATCH_SIZE)
        acc.append(merge_sort(test_dataset, name="Test_seq{}".format(i)))   
        


    # In[ ]:


    def merge_sort_test_exmp(inp):
        print("Original sequence:")
        print(inp)
        print("Target sequence:")
        print(np.sort(inp))
        seq_len = inp.shape[-1]
        L = tf.cast(tf.math.ceil(tf.math.log(tf.cast(seq_len, tf.float32))/tf.math.log(2.0)), tf.int64)
        pad_num = tf.pow(2,L)-seq_len
        inp_new = tf.concat([inp[tf.newaxis,:,tf.newaxis], tf.ones([1, seq_len, 1], dtype=tf.int64)*end_token], -1)
        inp_new = tf.concat([inp_new, pad_token*tf.ones([inp_new.shape[0], pad_num, 2], dtype=tf.int64)], -2)
        for i in range(1, L+1):
            print("step {}:".format(i))
            j = tf.pow(2,i)
            inp_res = tf.reshape(inp_new, [-1, j+2])
            batch_size_new = inp_res.shape[0]
            
            enc_inp = inp_res[:, tf.newaxis,:]
            dec_inp = tf.zeros((batch_size_new, 1, 1), dtype=tf.int64)

            seq_1 = tf.pow(2,i-1)
            seq_2 = seq_1
            enc_mask = create_padding_mask(inp_res)
            enc_padding_mask = tf.concat([tf.ones((batch_size_new, 1, 1)), tf.zeros((batch_size_new, 1, seq_1)), tf.ones((batch_size_new, 1, 1)), tf.zeros((batch_size_new, 1, seq_2))], -1)
            enc_padding_mask = 1-enc_padding_mask[:,:,tf.newaxis,:]
            enc_padding_mask = tf.maximum(enc_mask, enc_padding_mask)
            
            dec_padding_mask = enc_padding_mask
            combined_mask = None
                
            out_list =[]
            for _ in range(j+1):
                predictions, _, predicted_pos = transformer(enc_inp, dec_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)
                out_list.append(tf.squeeze(predictions, -2))
                ########## compute argmax #################
                predicted_pos_max = tf.reduce_max(predicted_pos, axis=-1, keepdims=True)
                predicted_pos_max = tf.equal(predicted_pos, predicted_pos_max)
                predicted_pos_ind = tf.reshape(predicted_pos_max, [-1, predicted_pos_max.shape[-1]])
                predicted_pos_ind = tf.where(predicted_pos_ind)
                pos_id = tf.cast(tf.math.segment_min(predicted_pos_ind[:,1],predicted_pos_ind[:,0]), tf.int64)
                pos_id = tf.reshape(pos_id, [predicted_pos_max.shape[0], predicted_pos_max.shape[1]])
                ############################################
                chg_msk = tf.one_hot(pos_id, seq_1+seq_2+2)[:,:,:,tf.newaxis]
                init_msk = tf.transpose(enc_padding_mask,[0,1,3,2])
                x = tf.concat([init_msk,chg_msk],axis=-1)
                x = tf.reshape(x,[-1, x.shape[-2], x.shape[-1]])
                x = 2*x-1
                predict_msk = msk_transform(x, False)
                predict_msk = tf.cast(tf.greater(predict_msk, 0), tf.float32)
                enc_padding_mask = predict_msk[:, tf.newaxis, tf.newaxis, :]
                dec_padding_mask = enc_padding_mask
            inp_new = tf.concat(out_list, -2)
            inp_new = tf.cast(tf.greater(inp_new, 0), tf.int64)
            inp_new = back2int(inp_new)
            inp_new_re = tf.reshape(inp_new, (-1,))
            print(tf.squeeze(tf.gather(inp_new_re, tf.where(tf.not_equal(inp_new_re, end_token))))[:seq_len])


    # In[ ]:


    test_length = 100
    step = 2
    random_num = np.random.choice(range(num_max), (test_length,))
    print("random num test:")
    merge_sort_test_exmp(random_num)
    start_tst = np.random.choice(range(num_max-(test_length-1)*step))
    test = np.tile(start_tst, [test_length])
    test += np.arange(0, test_length*step, step)
    np.apply_along_axis(np.random.shuffle, 0, test)
    print("granularity test (step: {}):".format(step))
    merge_sort_test_exmp(test)
    