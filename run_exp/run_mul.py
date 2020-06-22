from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.append('../data/')
sys.path.append('../model/')
sys.path.append('../')
from mul_data import mul_data
from model import *
from utils import *

def run_mul(current_dir, reload_from_dir_1, reload_dir_1, checkpoint_path_1):
    binary_size = 12
    d_model = 28

    Train_size = 15000
    BATCH_SIZE = 64
    Val_size = 1500

    num_layers = 6
    dff = 128

    target_vocab_size = binary_size  # no inf
    dropout_rate = 0.1
    res_ratio = 1.5

    start_token_dec = 0

    make_sym = True
    
    if reload_from_dir_1:
        EPOCHS_1 = 1
    else:
        EPOCHS_1 = 300
    seq_len_1 = 1
    seq_len_2 = 1
    seq_len = seq_len_1 + seq_len_2

    state_size = seq_len_1

    out_num_1 = True
    out_pos_1 = False
    assert(out_num_1 or out_pos_1)
    USE_positioning_1 = True
    pos = 3

    max_prod = 2 ** binary_size-1 ### 0-max_prod
    max_num = np.int64(np.floor(np.sqrt(max_prod)))
    
    with open("{}parameters.txt".format(current_dir), 'w') as fi:
        fi.write("binary_size: {}\nd_model: {}\nTrain_SIZE: {}\nBATCH_SIZE: {}\nVal_size: {}\nnum_layers: {}\ndff: {}\ntarget_vocab_size: {}\ndropout_rate: {}\nmake_sym: {}\nEPOCHS_1: {}\nseq_len_1: {}\nseq_len_2: {}\nout_num_1: {}\nout_pos_1: {}\nUSE_positioning_1: {}\nmax_prod: {}\npos: {}".format(
            binary_size, d_model, Train_size, BATCH_SIZE, Val_size, num_layers, dff, target_vocab_size, dropout_rate, make_sym, EPOCHS_1, seq_len_1, seq_len_2, out_num_1, out_pos_1, USE_positioning_1, max_prod, pos) + reload_from_dir_1 * "\nreload_dir_1: {}".format(reload_dir_1))
    
    Train_dataset_1, Val_dataset_1, Texmp_size = mul_data(current_dir, reload_from_dir_1, reload_dir_1, max_num, max_prod, Train_size, Val_size, binary_size)
    def encode_t1(tr, seed, srt_add):
        srt_add = np.array([srt_add])
        return tr, seed, srt_add
    def tf_encode_t1(tr, seed, srt_add):
        return tf.py_function(encode_t1, [tr, seed, srt_add], [tf.int64, tf.int64, tf.int64])
    train_dataset_1 = Train_dataset_1.map(tf_encode_t1)
    train_dataset_1 = train_dataset_1.cache()
    train_dataset_1 = train_dataset_1.shuffle(Texmp_size).batch(BATCH_SIZE)
    train_dataset_1 = train_dataset_1.prefetch(tf.data.experimental.AUTOTUNE)

    exmp = next(iter(train_dataset_1))
    print(exmp[0][0,:])
    print(exmp[1][0])
    print(exmp[2][0,:])

    val_dataset_1 = Val_dataset_1.map(tf_encode_t1)
    val_dataset_1 = val_dataset_1.batch(BATCH_SIZE)
    exmp = next(iter(val_dataset_1))
    print(exmp[0][0,:])
    print(exmp[1][0])
    print(exmp[2][0,:])
    
    transformer_1 = Transformer(num_layers, d_model, binary_size, dff, pos, target_vocab_size, make_sym, USE_positioning_1, out_num_1, out_pos_1, res_ratio, dropout_rate)
    learning_rate = CustomSchedule(d_model)
    optimizer_1 = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                        epsilon=1e-9)
    ckpt_1 = tf.train.Checkpoint(transformer_1=transformer_1,
                           optimizer_1=optimizer_1)

    ckpt_manager_1 = tf.train.CheckpointManager(ckpt_1, checkpoint_path_1, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager_1.latest_checkpoint:
        ckpt_1.restore(ckpt_manager_1.latest_checkpoint)
        print ('Model_1 checkpoint restored!!')
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')
    d_loss = tf.keras.metrics.Mean(name='d_loss')
    d_accuracy = tf.keras.metrics.Accuracy(name='d_accuracy')


    def create_masks_1(seed, seq_1, seq_2= -1):
        if seq_2 == -1:
            seq_2 = seq_1
        state_size = seq_1
        mask_source = tf.one_hot(seed, seq_1+seq_2)[:, tf.newaxis, :]
        change_mask = tf.concat([tf.eye(state_size), tf.zeros((state_size, seq_2))], -1)
        enc_padding_mask = tf.maximum(change_mask[tf.newaxis, :, :], mask_source)
        combined_mask = None
        enc_padding_mask = 1-enc_padding_mask
        enc_padding_mask = enc_padding_mask[:,:,tf.newaxis,:]
        dec_padding_mask = enc_padding_mask
        return enc_padding_mask, combined_mask, dec_padding_mask
    
    summary_writer_1 = tf.summary.create_file_writer(current_dir + "logs")
    @tf.function
    def train_step_1(inp, seed, tar, seq_1, binary_size):
        batch_size = seed.shape[0]
        state_size = seq_1
        enc_inp = tf.tile(inp[:, tf.newaxis, :],[1, state_size, 1])
        dec_inp = tf.ones((batch_size, state_size, 1), dtype=tf.int64)*start_token_dec
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks_1(seed, seq_1)
        with tf.GradientTape() as tape:
            predictions, _ = transformer_1(enc_inp, dec_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
            pred = tf.squeeze(predictions, -2)
            loss = loss_function(binary_encoding(tar, binary_size), pred)
        gradients = tape.gradient(loss, transformer_1.trainable_variables)
        optimizer_1.apply_gradients(zip(gradients, transformer_1.trainable_variables))
        train_loss(loss)
        tf.summary.scalar("loss", train_loss.result(), step=optimizer_1.iterations)
        pred_binary = tf.cast(tf.greater(pred, 0), tf.int64)
        train_accuracy(tar, back2int(pred_binary))
        
    def eval_val_1(dataset, seq_1, binary_size, name='Validation'):
        d_loss.reset_states()
        d_accuracy.reset_states()
        state_size = seq_1
        for element in dataset:
            inp, seed, tar = element
            batch_size = seed.shape[0]
            enc_inp = tf.tile(inp[:, tf.newaxis, :],[1, state_size, 1])
            dec_inp = tf.ones((batch_size, state_size, 1), dtype=tf.int64)*start_token_dec
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks_1(seed, seq_1)
            predictions, _ = transformer_1(enc_inp, dec_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)
            pred = tf.squeeze(predictions, -2)
            loss = loss_function(binary_encoding(tar, binary_size), pred)
            d_loss(loss)
            pred_binary = tf.cast(tf.greater(pred, 0), tf.int64)
            d_accuracy(tar, back2int(pred_binary))
        print('{}_Loss {:.4f} {}_Accuracy {:.4f}'.format(name, d_loss.result(), name, d_accuracy.result()))
        return d_accuracy.result()
    
    
    for epoch in range(EPOCHS_1):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        with summary_writer_1.as_default():
            for (batch, (inp, seed, tar)) in enumerate(train_dataset_1):
                train_step_1(inp, seed, tar, seq_len_1, binary_size)
                
                if batch % 500 == 0:
                    print('Epoch {} Batch {}:\nTraining_loss {:.4f} Training_Accuracy {:.4f}'.format(
                        epoch + 1, batch, train_loss.result(), train_accuracy.result()))
                    d_acc = eval_val_1(val_dataset_1, seq_len_1, binary_size)
                    tf.summary.scalar("val_acc", d_acc, step=optimizer_1.iterations)
            if (epoch+1) % 5 == 0:
                ckpt_save_path = ckpt_manager_1.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
            print('Epoch {}:\nTraining_Loss {:.4f} Training_Accuracy {:.4f}'.format(epoch + 1,
                                                        train_loss.result(),
                                                        train_accuracy.result()))
            d_acc = eval_val_1(val_dataset_1, seq_len_1, binary_size)
            tf.summary.scalar('val_acc', d_acc, step=optimizer_1.iterations)
            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    
    
    def evaluate(inp_sentence, seed, seq_1, seq_2=1):
        print("Original seq:")
        print(inp_sentence)
        print("seed")
        print(seed)
        batch = inp_sentence.shape[0]
        seed = seed + seq_1
        state_size = seq_1
        inp_sentence = np.array(inp_sentence).astype('int64')
        inp_sentence = tf.convert_to_tensor(inp_sentence)
        
        enc_inp = tf.tile(inp_sentence[:, tf.newaxis, :], [1, state_size, 1])
        dec_inp = tf.ones((batch, state_size, 1), dtype=tf.int64)*start_token_dec
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks_1(seed, seq_1, seq_2)
        predictions, _ = transformer_1(enc_inp, dec_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)
        pred = tf.squeeze(predictions)
        pred = back2int(tf.cast(tf.greater(pred, 0), tf.int64))
        print("mul")
        print(pred)
    
    EMB = transformer_1.emb(tf.eye(binary_size))
    np.save(current_dir+"emb.npy",EMB)
    
    seq_len_1 = 7
    seq_len_2 = 1

    exmp = np.array([6, 4, 3, 5, 2, 7, 8, 4])
    exmp = exmp[np.newaxis, :]
    evaluate(exmp, np.array([0]), seq_len_1, seq_len_2)

    num = 2
    op1 = np.random.choice(range(max_num))
    max_op2 = np.int64(np.floor(max_prod/op1))
    test_exmp = np.random.choice(range(max_op2), (num, seq_len_1), replace=False)
    test_exmp = np.concatenate([test_exmp, np.ones((test_exmp.shape[0], 1))*op1], -1)
    test_special = []
    seq_len = seq_len_1 + seq_len_2
    for i in range(4):
        step = i + 1
        start_test = np.random.choice(range(max_num-(seq_len-1)*step), (num, 1), replace=False)
        test_i = np.tile(start_test, [1, seq_len])
        test_i += np.arange(0, seq_len*step, step)
        test_special.append(test_i)
    test_special = np.concatenate(test_special, axis=0)
    np.apply_along_axis(np.random.shuffle, 1, test_special)
    test_exmp = np.concatenate([test_exmp, test_special], axis=0)
    test_mul_out = np.zeros((test_exmp.shape[0], seq_len_1), dtype='int64')
    test_seed = np.zeros((test_exmp.shape[0],), dtype='int64')

    def join(seq, seq_len_1):
        seql = len(seq)
        ptr = 0
        mul_output = []
        while (ptr!=seq_len_1):
            mul_output.append(seq[ptr]*seq[seq_len_1])
            ptr += 1
        return mul_output

    for i, seq in enumerate(test_exmp):
        mul_output = join(seq, seq_len_1)
        test_mul_out[i,:] = mul_output
    evaluate(test_exmp, test_seed, seq_len_1, seq_len_2)
    print("actual:")
    print(test_mul_out)

    test_small_list = []
    for mul_num in range(4):
        test_small = test_special[:,:seq_len_1]
        np.apply_along_axis(np.random.shuffle, 1, test_small)
        test_small_2 = np.ones((test_small.shape[0], 1))*mul_num
        test_small_list.append(tf.concat([test_small, test_small_2], -1))
    test_small = tf.concat(test_small_list, 0)

    test_small_mul_out = np.zeros((test_small.shape[0], seq_len_1), dtype='int64')
    test_small_seed = np.zeros((test_small.shape[0],), dtype='int64')

    for i, seq in enumerate(test_small):
        mul_output = join(seq, seq_len_1)
        test_small_mul_out[i,:] = mul_output
    evaluate(test_small, test_small_seed, seq_len_1, seq_len_2)
    print("actual:")
    print(test_small_mul_out)

    