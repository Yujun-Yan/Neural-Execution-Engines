from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np

def binary_encoding(x, binary_size):  ############ different tasks have different binary_size
    pow_base = tf.reverse(tf.range(binary_size, dtype=tf.int64),[-1])
    out = tf.bitwise.bitwise_and(tf.expand_dims(x, -1), tf.bitwise.left_shift(tf.constant(1, dtype=tf.int64), pow_base))
    out = tf.cast(tf.greater(out, 0), tf.float32)
    return out

def back2int(x): ##x: int64
    binary_size = x.shape[-1]
    pow_base = tf.bitwise.left_shift(tf.constant(1, dtype=tf.int64),tf.reverse(tf.range(binary_size, dtype=tf.int64),[-1])[:, tf.newaxis])
    out = tf.matmul(x, pow_base)
    return tf.squeeze(out,-1)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred, weights=None):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
    loss_ = loss_object(real, pred)
    if weights is not None:
        loss_ *= weights
    return tf.reduce_mean(loss_)

def loss_pos(real, pred, weights=None):
    loss_obj = tf.keras.losses.CategoricalCrossentropy(reduction='none')
    loss_ = loss_obj(real, pred)
    if weights is not None:
        loss_ *= weights
    return tf.reduce_mean(loss_)

class Graph_mst():
    def __init__(self, vertices, inf):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)] for row in range(vertices)]
        self.inf = inf
        
    def minDistance(self, node_val, mstset):
        mini = self.inf + 1
        for v in range(self.V):
            if (node_val[v]<mini) and (mstset[v]==False):
                mini = node_val[v]
                min_index = v
        return min_index, mini
    
    def primmst(self, src, traces=False):
        node_val = [self.inf] * self.V
        node_val[src] = 0
        mstset = [False] * self.V
        next_node = []
        weight = []
        if traces:
            train_seq = np.zeros((self.V, self.V), dtype=np.int64)
            train_msk = np.zeros((self.V+1, self.V), dtype=np.float32)
            test_seq = np.zeros((self.V, 1), dtype=np.int64)
        for i in range(self.V):
            u, mini = self.minDistance(node_val, mstset)
            if traces:
                train_seq[i,:] = np.array(node_val)
                train_msk[i+1,u] = 1
                test_seq[i,:] = mini
            if mini==self.inf and not traces:
                break
            next_node.append(u)
            weight.append(mini)
            mstset[u] = True
            for v in range(self.V):
                if self.graph[u][v]>0 and mstset[v]==False and node_val[v] >self.graph[u][v]:
                    node_val[v] = self.graph[u][v]
        if traces:
            train_msk[self.V,u] = 1
            return train_seq, train_msk, test_seq
        else:
            return weight[1:], next_node[1:]          
    
    
class Graph_paths():
    def __init__(self, vertices, inf):
        self.V = vertices
        self.inf = inf
        self.graph = [[0 for column in range(vertices)] for row in range(vertices)]
        
    def minDistance(self, dist, sptSet):
        mini = self.inf + 1
        for v in range(self.V):
            if (dist[v]<mini) and (sptSet[v]==False):
                mini = dist[v]
                min_index = v
        return min_index, mini
    
    def dijkstra(self, src, traces=False):
        
        dist = [self.inf] * self.V
        dist[src] = 0
        sptSet = [False] * self.V
        
        if traces:
            train_seq = np.zeros((self.V, self.V), dtype=np.int64)
            train_msk = np.zeros((self.V+1, self.V),dtype=np.float32)
            test_seq = np.zeros((self.V, 1), dtype=np.int64)
        
        for cout in range(self.V):
            u, mini = self.minDistance(dist, sptSet)
            if traces:
                train_seq[cout,:] = np.array(dist)
                train_msk[cout+1,u] = 1
                test_seq[cout,:] = mini
            
            sptSet[u] = True
            
            for v in range(self.V):
                if (self.graph[u][v]>0) and (sptSet[v]==False) and (dist[v]>dist[u] + self.graph[u][v]):
                    dist[v] = dist[u] + self.graph[u][v]
        if traces:
            train_msk[self.V,u] = 1
            return train_seq, train_msk.astype(float), test_seq
        else:
            return dist