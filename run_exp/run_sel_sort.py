from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.append('../data/')
sys.path.append('../model/')
sys.path.append('../')
from sel_sort_data import sel_sort_data_gen
from model import *
from utils import *

def run_sel(current_dir, reload_from_dir_2, reload_dir_2, checkpoint_path_2_data, checkpoint_path_2_msk):
    ''''
    current_dir: string, current checkpoint and data will be saved in this directory
    reload_from_dir_2: bool, whether to reload
    reload_dir_2: string, reload directory
    checkpoint_path_2_data: string, checkpoint path for the data
    checkpoint_path_2_msk: string, checkpoint path for the mask
    
    '''
    binary_size = 8  #### data representation using binary_size bit binary
    d_model = 16     #### size of embedding for each bit representation
    filter_size = 3  #### filter size of 1D convnet
    num_filters = 16  ##### number of filters

    Train_SIZE = 20000 #### random sequences
    BATCH_SIZE = 64  ###### batch size
    Val_size = 2000  ##### number of validation example

    num_layers = 6  #### number of layers in encoder/decoder
    dff = 128  ##### hidden units in MLP

    train_sub_size = 50 #### number of small step sequences for training data per group
    val_sub_size = 10  #### number of small steps sequences for validation data per group
    rep_num = 20 ###### repeatition number

    target_vocab_size = binary_size + 2   #### end_token, inf
    dropout_rate = 0.1  #### dropout rate
    res_ratio = 1.5  #### coefficient in residual connection

    inf = 2 ** binary_size  ##### number representing infinity
    end_token = 2 ** (binary_size+1)  ###### number representing end token
    start_token_dec = 0 ####### number representing start token
    
    if reload_from_dir_2:
        EPOCHS_2 = 1
    else:
        EPOCHS_2 = 100   ###### epochs to train
    seq_len_p2 = 8
    state_size_2 = seq_len_p2 + 1  ###### sequence length

    out_num_2 = True  ###### whether to output number
    out_pos_2 = True  ##### whether to output position
    assert(out_num_2 or out_pos_2)
    USE_positioning_2 = False  ##### whether using positional encoding
    pos_2 = seq_len_p2 + 1
    num_max_2 = 2 ** binary_size  ##### double check
    make_sym = True ##### whether making the attention symmetric
    
    ###### save paraters to file ###########
    with open("{}parameters.txt".format(current_dir), 'w') as fi:
        fi.write("binary_size: {}\nd_model: {}\nfilter_size: {}\nnum_filters: {}\nTrain_SIZE: {}\nBATCH_SIZE: {}\nVal_size: {}\nnum_layers: {}\ndff: {}\ntrain_sub_size: {}\nval_sub_size: {}\nrep_num: {}\ntarget_vocab_size: {}\ndropout_rate: {}\nmake_sym: {}\nEPOCHS_2: {}\nseq_len_p2: {}\nout_num_2: {}\nout_pos_2: {}\nUSE_positioning_2: {}\nnum_max_2: {}".format(
            binary_size, d_model, filter_size, num_filters, Train_SIZE, BATCH_SIZE, Val_size, num_layers, dff, train_sub_size, val_sub_size, rep_num, target_vocab_size, dropout_rate, make_sym, EPOCHS_2, seq_len_p2, out_num_2, out_pos_2, USE_positioning_2, num_max_2) + reload_from_dir_2 * "\nreload_dir_2: {}".format(reload_dir_2))
        
    Train_dataset_2, Val_dataset_2, Texmp_size = sel_sort_data_gen(current_dir, reload_from_dir_2, reload_dir_2, num_max_2, Train_SIZE, Val_size, train_sub_size, val_sub_size, rep_num, seq_len_p2, state_size_2, end_token, inf)
    
    def encode_t2(tr, mask, srt):
        return tr, mask, srt
    def tf_encode_t2(tr, mask, srt):
        return tf.py_function(encode_t2, [tr, mask, srt], [tf.int64, tf.float32, tf.int64])

    def encode_wo_mask(tr, srt):
        tr = np.hstack((tr, [end_token]))
        srt = np.hstack((srt, [end_token]))
        return tr, srt
    def tf_encode_wo_mask(tr, srt):
        return tf.py_function(encode_wo_mask, [tr, srt], [tf.int64, tf.int64])

    train_dataset_2 = Train_dataset_2.map(tf_encode_t2)
    train_dataset_2 = train_dataset_2.cache()
    train_dataset_2 = train_dataset_2.shuffle(Texmp_size).batch(BATCH_SIZE)
    train_dataset_2 = train_dataset_2.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset_2 = Val_dataset_2.map(tf_encode_t2)
    val_dataset_2 = val_dataset_2.batch(BATCH_SIZE)

    exmp = next(iter(train_dataset_2))
    print(exmp[0][0,:])
    print(exmp[1][0])
    print(exmp[2][0,:])

    exmp = next(iter(val_dataset_2))
    print(exmp[0][0,:])
    print(exmp[1][0])
    print(exmp[2][0,:])
    
    transformer_2 = Transformer(num_layers, d_model, binary_size+2, dff, pos_2, target_vocab_size, make_sym, USE_positioning_2, out_num_2, out_pos_2, res_ratio, dropout_rate)
    msk_transform_2 = mask_transform(num_filters, filter_size, dropout_rate)
    
    
    ###### using a custom learning rate with warm-up
    learning_rate_2_data = CustomSchedule(d_model)
    learning_rate_2_msk = CustomSchedule(num_filters)
    
    ####### set up optimizers ###########
    optimizer_2_data = tf.keras.optimizers.Adam(learning_rate_2_data, beta_1=0.9, beta_2=0.98, 
                                        epsilon=1e-9)

    optimizer_2_msk = tf.keras.optimizers.Adam(learning_rate_2_msk, beta_1=0.9, beta_2=0.98, 
                                        epsilon=1e-9)
    
    
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
        
    def create_masks(mask, seq_len_p2):
        '''
        create masks of different states for training
        '''
        
        state_size = seq_len_p2 + 1
        stacked_mask = tf.cumsum(mask, axis=-2)
        enc_padding_mask = stacked_mask[:,:-1,:][:,:,tf.newaxis,:]
        dec_padding_mask = enc_padding_mask
        combined_mask = None
        return enc_padding_mask, combined_mask, dec_padding_mask

    def prep_in_out(inp, tar, batch_size, seq_len_p2):
        '''
        prepare the input/output pairs for training
        '''
        state_size = seq_len_p2 + 1
        enc_inp = tf.tile(inp[:, tf.newaxis, :], [1, state_size-1, 1])
        tar_inp = tf.ones([batch_size, state_size-1, 1], dtype=tf.int64)*start_token_dec
        tar_real = tar[:,:,tf.newaxis]
        return enc_inp, tar_inp, tar_real        


    # In[19]:


    train_data_loss = tf.keras.metrics.Mean(name='train_data_loss')  ####### total training loss
    train_data_content_loss = tf.keras.metrics.Mean(name='train_data_content_loss') ######## number recovery loss during training
    train_data_pos_loss = tf.keras.metrics.Mean(name='train_data_pos_loss') ##### pointer loss during training
    train_msk_loss = tf.keras.metrics.Mean(name='train_msk_loss') ###### mask loss during training
    train_data_accuracy = tf.keras.metrics.Accuracy(name='train_data_accuracy')
    train_msk_accuracy = tf.keras.metrics.Accuracy(name='train_msk_accuracy')
    d_loss = tf.keras.metrics.Mean(name='d_loss')
    d_accuracy = tf.keras.metrics.Accuracy(name='d_accuracy')
    
    
    summary_writer_2 = tf.summary.create_file_writer(current_dir + 'logs')
    @tf.function
    def train_step_2_data(inp, mask, tar):
        seq_len_p2 = mask.shape[-1]
        batch_size = mask.shape[0]
        enc_inp, tar_inp, tar_real = prep_in_out(inp, tar, batch_size, seq_len_p2) ##### get input/output pairs
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(mask, seq_len_p2) ##### get masks for all states
        
        with tf.GradientTape() as tape:
            predictions, _, predicted_mask = transformer_2(enc_inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss_content = loss_function(binary_encoding(tar_real, binary_size+2), predictions)
            loss_position = loss_pos(mask[:, 1:, :], predicted_mask[:, :seq_len_p2, :])
            loss = loss_content + loss_position
        gradients = tape.gradient(loss, transformer_2.trainable_variables)
        optimizer_2_data.apply_gradients(zip(gradients, transformer_2.trainable_variables))
        train_data_loss(loss)
        train_data_content_loss(loss_content)
        train_data_pos_loss(loss_position)
        tf.summary.scalar("data_loss", train_data_loss.result(), step=optimizer_2_data.iterations)
        tf.summary.scalar("data_content_loss", train_data_content_loss.result(), step=optimizer_2_data.iterations)
        tf.summary.scalar("data_pos_loss", train_data_pos_loss.result(), step=optimizer_2_data.iterations)
        pred_binary = tf.cast(tf.greater(predictions, 0), tf.int64)
        pred_binary = back2int(pred_binary)
        train_data_accuracy(tar_real, pred_binary)

    @tf.function
    def train_step_2_msk(mask):
        msk_cum = tf.cumsum(mask, axis=-2)
        init_msk = msk_cum[:,:-1,:,tf.newaxis] ############### init_msk is the mask used in the encoder
        msk_chg = mask[:,1:,:,tf.newaxis]  ######### pointer output from the modified transformer
        tar_msk = msk_cum[:,1:,:]
        ######### concatenate the initial mask with the pointer as the input #####################
        x = tf.concat([init_msk, msk_chg], axis=-1) ### batch_size, seq_len, seq_len, 2
        x = tf.reshape(x, (-1, x.shape[-2], x.shape[-1]))
        x = 2*x-1 ###### normalization, rescale to -1 and 1
        with tf.GradientTape() as tape:
            predict_msk = msk_transform_2(x, True) #### batch_size*seq_len, seq_len
            predict_msk = tf.reshape(predict_msk, (tar_msk.shape[0], tar_msk.shape[-2], tar_msk.shape[-1]))
            loss_msk = loss_function(tar_msk, predict_msk)
        gradients = tape.gradient(loss_msk, msk_transform_2.trainable_variables)
        optimizer_2_msk.apply_gradients(zip(gradients, msk_transform_2.trainable_variables))
        train_msk_loss(loss_msk)
        tf.summary.scalar("msk_loss", train_msk_loss.result(), step=optimizer_2_msk.iterations)
        predict_msk_binary = tf.cast(tf.greater(predict_msk, 0), tf.float32)
        err = tf.reduce_sum(tar_msk-predict_msk_binary, axis=-1)
        train_msk_accuracy(err, tf.zeros_like(err))
        

    # In[38]:


    def eval_val_2(dataset, name='Validation', include_pos_loss=True):
        '''
        The evaluation will use the next mask from the output of mask transform network
        '''
        d_loss.reset_states()
        d_accuracy.reset_states()
        for element in dataset:
            if include_pos_loss:
                inp, mask, tar = element
                mask_cum = tf.cumsum(mask, axis=-2)
            else:
                inp, tar = element
            tar_real = tar           ###### ground truth
            batch_size = inp.shape[0]
            seq_len_p2 = inp.shape[-1]
            state_size = seq_len_p2 + 1
            enc_inp = inp[:, tf.newaxis, :]
            enc_padding_mask = tf.zeros([batch_size, 1, 1, seq_len_p2]) 
            dec_padding_mask = enc_padding_mask ########### initial mask for encoder
            
            mask_list = []
            out_list = []
            for i in range(state_size-1):
                tar_inp = tf.ones([batch_size, 1, 1], tf.int64)*start_token_dec
                combined_mask = None
                ######### find the next smallest number ##############
                predictions, _, predicted_mask = transformer_2(enc_inp, tar_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)
                predictions = predictions[:, :, -1:, :]
                out_list.append(tf.squeeze(predictions, axis=-2))
                last_mask = predicted_mask[:,-1,:]
                last_mask = tf.one_hot(tf.argmax(last_mask, axis=-1), seq_len_p2)[:,:,tf.newaxis] ########## pointer output from the modified transformer
                
                init_mask = tf.squeeze(enc_padding_mask,[1,2])
                init_mask = init_mask[:,:,tf.newaxis]
                x = tf.concat([init_mask,last_mask], axis=-1) ######### concatenate initial mask with the pointer
                predict_msk = msk_transform_2(2*x-1, False) ####### rescale to [-1,1] and send to mask transform network
                mask_list.append(predict_msk[:,tf.newaxis,:])
                predict_msk = tf.cast(tf.greater(predict_msk, 0), tf.float32)
                enc_padding_mask = predict_msk[:, tf.newaxis, tf.newaxis, :]
                dec_padding_mask = enc_padding_mask
            if include_pos_loss:
                mask_est = tf.concat(mask_list, -2)
                loss_position = loss_function(mask_cum[:,1:,:], mask_est)
            else:
                loss_position = 0
            out_est = tf.concat(out_list, -2)
            loss_content = loss_function(binary_encoding(tar_real, binary_size+2), out_est)
            loss = loss_position + loss_content
            d_loss(loss)
            
            out_binary = tf.cast(tf.greater(out_est, 0), tf.int64)
            out_binary = back2int(out_binary)
            d_accuracy(tar_real, out_binary)
        print('{}_Loss {:.4f} {}_Accuracy {:.4f}'.format(name, d_loss.result(), name, d_accuracy.result()))
        return d_accuracy.result()


    # In[23]:


    for epoch in range(EPOCHS_2):
        start = time.time()
        train_data_loss.reset_states()
        train_data_content_loss.reset_states()
        train_data_pos_loss.reset_states()
        train_msk_loss.reset_states()
        train_data_accuracy.reset_states()
        train_msk_accuracy.reset_states()
        with summary_writer_2.as_default():
            for (batch, (inp, mask, tar)) in enumerate(train_dataset_2):
                train_step_2_data(inp, mask, tar)
                train_step_2_msk(mask)
                if batch % 500 == 0:
                    print('Epoch {} Batch {}:\nTraining_Data_Loss {:.4f} Training_Data_Accuracy {:.4f}\nTraining_Msk_Loss {:.4f} Training_Msk_Accuracy {:.4f}'.format(epoch+1, batch, train_data_loss.result(), train_data_accuracy.result(), train_msk_loss.result(), train_msk_accuracy.result()))
                    eval_val_2(val_dataset_2)
            if (epoch + 1) % 5 == 0:
                ckpt_save_path_data = ckpt_manager_2_data.save()
                ckpt_save_path_msk = ckpt_manager_2_msk.save()
                print("Saving checkpoint for epoch {} at {} and {}".format(epoch + 1, ckpt_save_path_data, ckpt_save_path_msk))
            print('Epoch {} Batch {}:\nTraining_Data_Loss {:.4f} Training_Data_Accuracy {:.4f}\nTraining_Msk_Loss {:.4f} Training_Msk_Accuracy {:.4f}'.format(epoch+1, batch, train_data_loss.result(), train_data_accuracy.result(), train_msk_loss.result(), train_msk_accuracy.result()))
            eval_val_2(val_dataset_2)
            print('Time taken for 1 epoch: {} secs\n'.format(time.time()-start))
            


    # In[39]:
    print("Saving embeddings")
    EMB = transformer_2.emb(tf.eye(binary_size+2))
    np.save(current_dir+"sel_sort_emb.npy",EMB)


    max_seq_len = 100
    max_seq_step = 60
    test_size_random = 60
    test_size_small_steps = 10
    acc = []
    for i in range(9, max_seq_len+1):
        test_random = np.random.choice(range(num_max_2), (test_size_random, i))
        test_special = []
        if tf.less_equal(i, max_seq_step):
            for step in range(1,5):
                start_tst = np.random.choice(range(num_max_2-(i-1)*step), (test_size_small_steps, 1), replace=False)
                test = np.tile(start_tst, [1, i])
                test += np.arange(0, i*step, step)
                test_special.append(test)
        else:
            for step in range(1,5):
                start_tst = np.random.choice(range(num_max_2-(max_seq_step-1)*step), (test_size_small_steps, 1), replace=False)
                test = np.tile(start_tst, [1, max_seq_step])
                test += np.arange(0, max_seq_step*step, step)
                padded_random = np.random.choice(range(num_max_2), (test_size_small_steps, i-max_seq_step))
                test = np.concatenate((test, padded_random), axis=-1)
                test_special.append(test)
        test_special = np.concatenate(test_special, axis=0)
        np.apply_along_axis(np.random.shuffle, 1, test_special)
        test = np.concatenate([test_random, test_special], axis=0)
        Sorted_test = np.sort(test)
        test_dataset = tf.data.Dataset.from_tensor_slices((test, Sorted_test))
        test_dataset = test_dataset.map(tf_encode_wo_mask)
        test_dataset = test_dataset.batch(BATCH_SIZE)
        acc.append(eval_val_2(test_dataset, name="Test_seq{}".format(i), include_pos_loss=False))


    # In[42]:


    np.save("{}acc".format(current_dir), acc)
    plt.figure()
    plt.plot(range(9, len(acc)+9), acc)
    plt.xlabel('seq_len')
    plt.ylabel('acc')
    filename='acc_plot.png'
    plt.savefig(current_dir+filename)


    # In[48]:


    def evaluate(inp):
        '''
        Evaluating single sequence, for debug/exploration use, similar to eval_val_2
        '''
        print("Original seq: ")
        print(inp)
        print("Sorted seq:")
        print(np.sort(inp))
        batch_size = inp.shape[0]
        inp = np.hstack([inp, np.ones((batch_size, 1), dtype=np.int64)*end_token])
        seq_len_p2 = inp.shape[-1]
        state_size = seq_len_p2 + 1
        
        enc_inp = inp[:, tf.newaxis, :]
        enc_padding_mask = tf.zeros([batch_size, 1, 1, seq_len_p2])
        dec_padding_mask = enc_padding_mask
        
        out_list = []
        mask_list = []
        for i in range(state_size-1):
            tar_inp = tf.ones([batch_size, 1, 1], tf.int64)*start_token_dec
            combined_mask = None
            predictions, _, predicted_mask = transformer_2(enc_inp, tar_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)
            predictions = predictions[:,:,-1:,:]
            pred_binary = tf.cast(tf.greater(predictions, 0), tf.int64)
            pred_binary = back2int(pred_binary)
            out_list.append(tf.squeeze(pred_binary, -1))
            last_mask = predicted_mask[:,-1,:]
            mask_list.append(last_mask[:,tf.newaxis,:])
            last_mask = tf.one_hot(tf.argmax(last_mask, axis=-1), seq_len_p2)[:,:,tf.newaxis]
            init_mask = tf.squeeze(enc_padding_mask,[1,2])
            init_mask = init_mask[:,:,tf.newaxis]
            x = tf.concat([init_mask,last_mask], axis=-1)
            predict_msk = msk_transform_2(2*x-1, False)
            predict_msk = tf.cast(tf.greater(predict_msk, 0), tf.float32)
            enc_padding_mask = predict_msk[:,tf.newaxis,tf.newaxis,:]
            dec_padding_mask = enc_padding_mask
        
        out_est = tf.concat(out_list, -1)
        print("Predict:")
        print(out_est)
        return mask_list


    # In[49]:


    exmp_seq = 8
    exmp_num = 2
    #### generate random numbers #############
    rand_seq = np.random.choice(range(num_max_2), (exmp_num, exmp_seq))
    exmp_special = []
    for i in range(4):
        step = i+1
        start_exmp = np.random.choice(range(num_max_2-(exmp_seq-1)*step), (exmp_num, 1), replace=False)
        test_exmp = np.tile(start_exmp, [1, exmp_seq])
        test_exmp += np.arange(0, exmp_seq*step, step)
        exmp_special.append(test_exmp)
    exmp_special = np.concatenate(exmp_special, axis=0)
    np.apply_along_axis(np.random.shuffle, 1, exmp_special)
    rand_seq = np.int64(rand_seq)
    exmp_special = np.int64(exmp_special)
    print("random example:")
    evaluate(rand_seq)
    print("special example:")
    evaluate(exmp_special)


    # In[53]:

    ####### get results up to 100 lengths long
    exmp_seq = 100
    exmp_num = 1
    rand_seq = np.int64(np.random.choice(range(num_max_2), (exmp_num, exmp_seq)))
    mask_list = evaluate(rand_seq)
    mask_list = tf.squeeze(tf.concat(mask_list, -2))
    np.save("{}mask_{}".format(current_dir, exmp_seq), mask_list)
    plt.figure()
    plt.imshow(mask_list)
    filename='attention_mat_{}.png'.format(exmp_seq)
    plt.savefig(current_dir+filename)


    # In[54]:


    exmp_seq = 50
    exmp_num = 1
    rand_seq = np.int64(np.random.choice(range(num_max_2), (exmp_num, exmp_seq)))
    mask_list = evaluate(rand_seq)
    mask_list = tf.squeeze(tf.concat(mask_list, -2))
    np.save("{}mask_{}".format(current_dir, exmp_seq), mask_list)
    plt.figure()
    plt.imshow(mask_list)
    filename='attention_mat_{}.png'.format(exmp_seq)
    plt.savefig(current_dir+filename)