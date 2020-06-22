from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import sys
sys.path.append('../')
from utils import binary_encoding

def point_wise_feed_forward_network(d_model, dff):
    '''
    two layer MLP
    '''
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, state_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, state_size, seq_len, d_model)
  ])

def generate_similarity_score(q, k, NN, make_sym):
    '''
    two layer MLP for attention
    single head
    '''
    seq_len_k = tf.shape(k)[-2]
    seq_len_q = tf.shape(q)[-2]
    q_inp = tf.tile(q[:,:,:,tf.newaxis,:],[1,1,1,seq_len_k,1])
    k_inp = tf.tile(k[:,:,tf.newaxis,:,:],[1,1,seq_len_q,1,1])
    combined = tf.concat([q_inp, k_inp], -1)
    sim_weights = NN(combined)
    if make_sym:
        combined_2 = tf. concat([k_inp, q_inp],-1)
        sim_weights_2 = NN(combined_2)
        sim_weights += sim_weights_2
    return tf.squeeze(sim_weights, [-1])

def scaled_general_attention(q, k, v, mask, NN, make_sym):
    scaled_attention_logits = generate_similarity_score(q, k, NN, make_sym)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

def get_angles(pos, i, d_model):
    angle_rates = 1/np.power(10000, (2*(i//2)) / np.float32(d_model))
    return pos*angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis,:], d_model)
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:,1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

def add_pos(x, mask, pos_enc):
    if mask is None:
        mask = tf.zeros((x.shape[0], x.shape[1], 1, x.shape[2]))
    mask = tf.squeeze(1-mask, -2)
    x_ind = tf.math.cumsum(mask, -1)
    return x+tf.gather_nd(pos_enc, tf.cast(x_ind[:,:,:,tf.newaxis], dtype=tf.int64))


class Attention(tf.keras.layers.Layer):
    def __init__(self, d_model, make_sym):
        super(Attention, self).__init__()

        self.d_model = d_model
        self.make_sym = make_sym

        self.w = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
    
    def call(self, v, k, q, mask, NN):
        batch_size = tf.shape(q)[0]
    
        q = self.w(q)  # (batch_size, state_size, seq_len, d_model)
        k = self.w(k)  # (batch_size, state_size, seq_len, d_model)
        v = self.w(v)  # (batch_size, state_size, seq_len, d_model)
    
        scaled_attention, attention_weights = scaled_general_attention(
            q, k, v, mask, NN, self.make_sym)
        output = self.dense(scaled_attention)  # (batch_size, state_size, seq_len_q, d_model)
        return output, attention_weights

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, make_sym, res_ratio, rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.res_ratio = res_ratio
        self.mha = Attention(d_model, make_sym)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask, NN):

        attn_output, attention_weights = self.mha(x, x, x, mask, NN)  # (batch_size, state_size, seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(self.res_ratio*x + (2-self.res_ratio)*attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(self.res_ratio*out1 + (2-self.res_ratio)*ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, make_sym, res_ratio, rate=0.1):
        super(DecoderLayer, self).__init__()
        
        self.res_ratio = res_ratio
        self.mha1 = Attention(d_model, make_sym)
        self.mha2 = Attention(d_model, make_sym)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, enc_output, training, com_mask, dec_mask, NN):
        # enc_output.shape == (batch_size, state_size, 1, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, com_mask, NN)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1((2-self.res_ratio)*attn1 + self.res_ratio*x)

        
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, dec_mask, NN)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2((2-self.res_ratio)*attn2 + self.res_ratio*out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3((2-self.res_ratio)*ffn_output + self.res_ratio*out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, binary_size, dff, pos, make_sym, res_ratio,
               rate=0.1):
        super(Encoder, self).__init__()

        self.binary_size = binary_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(pos, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, dff, make_sym, res_ratio, rate) 
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask, NN, emb, use_pos):

        x = binary_encoding(x, self.binary_size)
        x = emb(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        if use_pos:
            x = add_pos(x, mask, self.pos_encoding)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask, NN)

        return x  # (batch_size, state_size, seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, binary_size, dff, pos, make_sym, res_ratio,
               rate=0.1):
        super(Decoder, self).__init__()

        self.binary_size = binary_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_encoding = positional_encoding(pos, d_model)

        self.dec_layers = [DecoderLayer(d_model, dff, make_sym, res_ratio, rate) 
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, enc_output, training, 
           com_mask, dec_mask, NN, emb, use_pos):

        attention_weights = {}
        
        x = binary_encoding(x, self.binary_size)
        x = emb(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        if use_pos:
            x += tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                 com_mask, dec_mask, NN)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, state_size, seq_len, d_model)
        return x, attention_weights

