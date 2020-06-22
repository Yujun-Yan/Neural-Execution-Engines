from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.append('../data/')
sys.path.append('../model/')
sys.path.append('../')
from min_graph import *
from model import *
from utils import *

def run_min_graph(current_dir, reload_from_dir_2, reload_dir_2, checkpoint_path_2_data, checkpoint_path_2_msk, mst):
    binary_size = 8  #### data representation using binary_size bit binary
    d_model = 16     #### size of embedding for each bit representation
    filter_size = 3
    num_filters = 16

    Train_SIZE = 20000 #### random sequences
    BATCH_SIZE = 64
    Val_size = 2000

    num_layers = 6
    dff = 128

    target_vocab_size = binary_size + 2   #### end_token, inf
    dropout_rate = 0.1
    res_ratio = 1.5

    inf = 2 ** binary_size
    end_token = 2 ** (binary_size+1)
    start_token_dec = 0
    
    if reload_from_dir_2:
        EPOCHS_2 = 1
    else:
        EPOCHS_2 = 150
    seq_len_p2 = 8
    state_size_2 = seq_len_p2 + 1

    out_num_2 = True
    out_pos_2 = True
    assert(out_num_2 or out_pos_2)
    USE_positioning_2 = False
    pos_2 = seq_len_p2 + 1
    num_max_2 = 2 ** binary_size  ##### double check
    make_sym = True
    
    with open("{}parameters.txt".format(current_dir), 'w') as fi:
        fi.write("binary_size: {}\nd_model: {}\nfilter_size: {}\nnum_filters: {}\nTrain_SIZE: {}\nBATCH_SIZE: {}\nVal_size: {}\nnum_layers: {}\ndff: {}\ntarget_vocab_size: {}\ndropout_rate: {}\nmake_sym: {}\nEPOCHS_2: {}\nseq_len_p2: {}\nout_num_2: {}\nout_pos_2: {}\nUSE_positioning_2: {}\nnum_max_2: {}".format(
            binary_size, d_model, filter_size, num_filters, Train_SIZE, BATCH_SIZE, Val_size, num_layers, dff, target_vocab_size, dropout_rate, make_sym, EPOCHS_2, seq_len_p2, out_num_2, out_pos_2, USE_positioning_2, num_max_2) + reload_from_dir_2 * "\nreload_dir_2: {}".format(reload_dir_2))
    
    graph_type_train = 'ER'
    graph_type_val_er = 'ER'
    graph_type_val_mix = 'mix'
    Train_dataset_2, Texmp_size = min_graph(current_dir, reload_from_dir_2, reload_dir_2, num_max_2, Train_SIZE, seq_len_p2, end_token, inf, graph_type_train, 'Train', mst)  
    Val_dataset_2_ER, _ = min_graph(current_dir, reload_from_dir_2, reload_dir_2, num_max_2, Val_size, seq_len_p2, end_token, inf, graph_type_val_er, 'Val_ER', mst)  
    Val_dataset_2_mix,_ = min_graph(current_dir, reload_from_dir_2, reload_dir_2, num_max_2, Val_size, seq_len_p2, end_token, inf, graph_type_val_mix, 'Val_mix', mst)
    
    def encode_t2(tr, mask, srt):
        return tr, mask, srt
    def tf_encode_t2(tr, mask, srt):
        return tf.py_function(encode_t2, [tr, mask, srt], [tf.int64, tf.float32, tf.int64])

    def encode_wo_mask(tr, srt):
        return tr, srt
    def tf_encode_wo_mask(tr, mask, srt):
        return tf.py_function(encode_wo_mask, [tr, srt], [tf.int64, tf.int64])

    train_dataset_2 = Train_dataset_2.map(tf_encode_t2)
    train_dataset_2 = train_dataset_2.cache()
    train_dataset_2 = train_dataset_2.shuffle(Texmp_size).batch(BATCH_SIZE)
    train_dataset_2 = train_dataset_2.prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset_2_er = Val_dataset_2_ER.map(tf_encode_t2)
    val_dataset_2_er = val_dataset_2_er.batch(BATCH_SIZE)
    
    val_dataset_2_mix = Val_dataset_2_mix.map(tf_encode_t2)
    val_dataset_2_mix = val_dataset_2_mix.batch(BATCH_SIZE)

    exmp = next(iter(train_dataset_2))
    print(exmp[0][0,:])
    print(exmp[1][0])
    print(exmp[2][0,:])

    exmp = next(iter(val_dataset_2_er))
    print(exmp[0][0,:])
    print(exmp[1][0])
    print(exmp[2][0,:])
    
    exmp = next(iter(val_dataset_2_mix))
    print(exmp[0][0,:])
    print(exmp[1][0])
    print(exmp[2][0,:])
    
    transformer_2 = Transformer(num_layers, d_model, binary_size+2, dff, pos_2, target_vocab_size, make_sym, USE_positioning_2, out_num_2, out_pos_2, res_ratio, dropout_rate)
    msk_transform_2 = mask_transform(num_filters, filter_size, dropout_rate)
    
    learning_rate_2_data = CustomSchedule(d_model)
    learning_rate_2_msk = CustomSchedule(num_filters)

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
        state_size = seq_len_p2 + 1
        stacked_mask = tf.cumsum(mask, axis=-2)
        enc_padding_mask = stacked_mask[:,:-1,:][:,:,tf.newaxis,:]
        dec_padding_mask = enc_padding_mask
        combined_mask = None
        return enc_padding_mask, combined_mask, dec_padding_mask

    def prep_in_out(inp, tar, batch_size, seq_len_p2):
        state_size = seq_len_p2 + 1
        enc_inp = inp
        tar_inp = tf.ones([batch_size, state_size-1, 1], dtype=tf.int64)*start_token_dec
        tar_real = tar
        return enc_inp, tar_inp, tar_real        


    # In[19]:


    train_data_loss = tf.keras.metrics.Mean(name='train_data_loss')
    train_data_content_loss = tf.keras.metrics.Mean(name='train_data_content_loss')
    train_data_pos_loss = tf.keras.metrics.Mean(name='train_data_pos_loss')
    train_msk_loss = tf.keras.metrics.Mean(name='train_msk_loss')
    train_data_accuracy = tf.keras.metrics.Accuracy(name='train_data_accuracy')
    train_msk_accuracy = tf.keras.metrics.Accuracy(name='train_msk_accuracy')
    d_loss = tf.keras.metrics.Mean(name='d_loss')
    d_accuracy = tf.keras.metrics.Accuracy(name='d_accuracy')
    
    
    summary_writer_2 = tf.summary.create_file_writer(current_dir + 'logs')
    @tf.function
    def train_step_2_data(inp, mask, tar):
        seq_len_p2 = mask.shape[-1]
        batch_size = mask.shape[0]
        enc_inp, tar_inp, tar_real = prep_in_out(inp, tar, batch_size, seq_len_p2)
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(mask, seq_len_p2)
        
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
        init_msk = msk_cum[:,:-1,:,tf.newaxis]
        msk_chg = mask[:,1:,:,tf.newaxis]
        tar_msk = msk_cum[:,1:,:]
        x = tf.concat([init_msk, msk_chg], axis=-1) ### batch_size, seq_len, seq_len, 2
        x = tf.reshape(x, (-1, x.shape[-2], x.shape[-1]))
        x = 2*x-1 ###### rescale to -1 and 1
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
        d_loss.reset_states()
        d_accuracy.reset_states()
        for element in dataset:
            if include_pos_loss:
                inp, mask, tar = element
                mask_cum = tf.cumsum(mask, axis=-2)
            else:
                inp, tar = element
            tar_real = tf.squeeze(tar, axis=-1)
            batch_size = inp.shape[0]
            seq_len_p2 = inp.shape[-1]
            state_size = seq_len_p2 + 1
            tar_inp = tf.ones([batch_size, 1, 1], tf.int64)*start_token_dec
            
            enc_padding_mask = tf.zeros([batch_size, 1, 1, seq_len_p2])
            dec_padding_mask = enc_padding_mask
            
            mask_list = []
            out_list = []
            for i in range(state_size-1):
                enc_inp = inp[:, i, :]
                enc_inp = enc_inp[:,tf.newaxis,:]
                combined_mask = None
                predictions, _, predicted_mask = transformer_2(enc_inp, tar_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)
                predictions = predictions[:, :, -1:, :]
                out_list.append(tf.squeeze(predictions, axis=-2))
                last_mask = predicted_mask[:,-1,:]
                last_mask = tf.one_hot(tf.argmax(last_mask, axis=-1), seq_len_p2)[:,:,tf.newaxis]
                init_mask = tf.squeeze(enc_padding_mask,[1,2])
                init_mask = init_mask[:,:,tf.newaxis]
                x = tf.concat([init_mask,last_mask], axis=-1)
                predict_msk = msk_transform_2(2*x-1, False)
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
                    eval_val_2(val_dataset_2_er, name='Validation_ER')
                    eval_val_2(val_dataset_2_mix, name='Validation_MIX')
            if (epoch + 1) % 5 == 0:
                ckpt_save_path_data = ckpt_manager_2_data.save()
                ckpt_save_path_msk = ckpt_manager_2_msk.save()
                print("Saving checkpoint for epoch {} at {} and {}".format(epoch + 1, ckpt_save_path_data, ckpt_save_path_msk))
            print('Epoch {} Batch {}:\nTraining_Data_Loss {:.4f} Training_Data_Accuracy {:.4f}\nTraining_Msk_Loss {:.4f} Training_Msk_Accuracy {:.4f}'.format(epoch+1, batch, train_data_loss.result(), train_data_accuracy.result(), train_msk_loss.result(), train_msk_accuracy.result()))
            eval_val_2(val_dataset_2_er, name='Validation_ER')
            eval_val_2(val_dataset_2_mix, name='Validation_MIX')
            print('Time taken for 1 epoch: {} secs\n'.format(time.time()-start))
            


    # In[39]:
    print("Saving embeddings")
    EMB = transformer_2.emb(tf.eye(binary_size+2))
    np.save(current_dir+"sel_sort_emb.npy",EMB)


   