from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from layers import *

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, binary_size, dff, pos, 
               target_vocab_size, make_sym, use_pos, out_num, out_pos, res_ratio, rate=0.1):
        super(Transformer, self).__init__()
        
        self.num_layers = num_layers
        self.NN = point_wise_feed_forward_network(1, dff)
        self.encoder = Encoder(num_layers, d_model, binary_size, dff, 
                               pos, make_sym, res_ratio, rate)

        self.decoder = Decoder(num_layers, d_model, binary_size, dff, 
                               pos, make_sym, res_ratio, rate)
        self.emb = tf.keras.layers.Dense(d_model, use_bias=False)
        self.use_pos = use_pos
        self.out_num = out_num
        self.out_pos = out_pos
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
    def call(self, inp, tar, training, enc_padding_mask, 
           com_mask, dec_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask, self.NN, self.emb, self.use_pos)  # (batch_size, inp_seq_len, d_model)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, com_mask, dec_padding_mask, self.NN, self.emb, self.use_pos)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return_list = []
        if self.out_num:
            return_list.append(final_output)
        return_list.append(attention_weights)
        if self.out_pos:
            last_att_weights = attention_weights['decoder_layer{}_block2'.format(self.num_layers)]
            last_att_weights = tf.reshape(last_att_weights, [-1, last_att_weights.shape[1]*last_att_weights.shape[2], last_att_weights.shape[-1]])
            return_list.append(last_att_weights)
        return return_list

class mask_transform(tf.keras.layers.Layer):
    '''
    1D Convnet for learning mask transformation
    '''
    def __init__(self, num_filters, filter_size, rate=0.1):
        super(mask_transform, self).__init__()
        self.CNN = tf.keras.layers.Conv1D(num_filters, filter_size, padding='same', activation='relu')
        self.final = tf.keras.layers.Dense(1)
        self.dropout = tf.keras.layers.Dropout(rate)
    def call(self, x, training):
        cnn_out = self.CNN(x) ###### batch_size*state_size, seq_len, 2
        cnn_out = self.dropout(cnn_out,training=training)
        out = self.final(cnn_out)
        return tf.squeeze(out, axis=-1)