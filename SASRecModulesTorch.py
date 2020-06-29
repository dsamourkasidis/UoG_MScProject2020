# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utility import normalize



"""def positional_encoding(dim, sentence_length, dtype=torch.float32):

    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    return torch.tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


def embedding(inputs, 
              vocab_size, 
              num_units, 
              zero_pad=True, 
              scale=True,
              l2_reg=0.0,
              scope="embedding", 
              with_t=False,
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
        
    For example,
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]    
    ```    
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       #initializer=tf.contrib.layers.xavier_initializer(),
                                       regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        
        if scale:
            outputs = outputs * (num_units ** 0.5) 
    if with_t: return outputs,lookup_table
    else: return outputs"""


class multihead_attention(nn.Module):
    def __init__(self, hidden_size, num_units=None, num_heads=8,dropout_rate=0,causality=False,
                        with_qk=False):
        super(multihead_attention,self).__init__()
        self.num_units=num_units
        self.num_heads = num_heads
        self.dropout_rate=dropout_rate
        self.causality = causality
        self.with_qk=with_qk
        self.hidden_size=hidden_size
        self.fc1 = nn.Linear(self.hidden_size,num_units)
        self.fc2 = nn.Linear(self.hidden_size,num_units)
        self.fc3 = nn.Linear(self.hidden_size,num_units)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(self.dropout_rate)
                        
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    # Set the fall back option for num_units
    
    def forward(self,queries,keys):
        if self.num_units is None:
            self.num_units = queries.size(-1)
            # Linear projections

        
        Q = self.fc1(queries) # (N, T_q, C)
        K = self.fc2(keys) # (N, T_k, C)
        V = self.fc3(keys) # (N, T_k, C)
    
        # Split and concat
        q_split = int(Q.size(2)/self.num_heads)
        k_split = int(K.size(2)/self.num_heads)
        v_split = int(V.size(2)/self.num_heads)
        Q_ = torch.cat(torch.split(Q, q_split, dim=2), dim=0) # (h*N, T_q, C/h) 
        K_ = torch.cat(torch.split(K, k_split, dim=2), dim=0) # (h*N, T_k, C/h) 
        V_ = torch.cat(torch.split(V, v_split, dim=2), dim=0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = torch.matmul(Q_, K.permute(0, 2, 1)) # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.size(-1) ** 0.5)
        
        # Key Masking
        key_masks = torch.sign(torch.abs(torch.sum(keys,-1))) # (N, T_k)
        key_masks = torch.cat(self.num_heads*[key_masks]) # (h*N, T_k)
        key_masks = torch.cat(queries.size(1)*[key_masks.unsqueeze(1)], dim=1) # (h*N, T_q, T_k)
        
        paddings = torch.ones_like(outputs)*(-2**32+1)
        outputs = torch.where(torch.eq(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Causality = Future blinding
        if self.causality:
            diag_vals = torch.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = torch.tril(diag_vals) # (T_q, T_k)
            masks = torch.cat(outputs.size(0)*[tril.unsqueeze(0)]) # (h*N, T_q, T_k)
   
            paddings = torch.ones_like(masks)*(-2**32+1)
            outputs = torch.where(torch.eq(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Activation
        outputs = self.softmax(outputs) # (h*N, T_q, T_k)
         
        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries,-1))) # (N, T_q)
        query_masks = torch.cat(self.num_heads*[query_masks]) # (h*N, T_q)
        query_masks = torch.cat(keys.size(1)*[query_masks.unsqueeze(-1)], dim=2) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)
    
        # Dropouts
        
        outputs = self.dropout(outputs)
               
        # Weighted sum
        outputs = torch.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        o_split = int(outputs.size(0)/self.num_heads)
        outputs = torch.cat(torch.split(outputs, o_split, dim=0), dim=2 ) # (N, T_q, C)
              
        # Residual connection
        outputs += queries
              
        # Normalize
        #outputs = normalize(outputs) # (N, T_q, C)
 
        if self.with_qk: return Q,K
        else: return outputs
        
        
class feedforward(nn.Module):
    """# Inner layer
    params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
              "activation": F.relu, "use_bias": True}
    outputs = tf.layers.conv1d(**params)
    outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
    # Readout layer
    params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
              "activation": None, "use_bias": True}
    outputs = tf.layers.conv1d(**params)
    outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        
        # Residual connection
    outputs += inputs
        
        # Normalize
        #outputs = normalize(outputs)
    
    return outputs"""
    def __init__(self, num_units=[2048, 512], dropout_rate=0.2):
        super(feedforward,self).__init__()
        self.inner_cnn = nn.Conv1d(10,num_units[0],1)
        self.readout_cnn = nn.Conv1d(num_units[0],10,1)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward (self,inputs):
        x = F.relu(self.inner_cnn(inputs))
        x = self.dropout(x)
        x = self.readout_cnn(x)
        x = self.dropout(x)
        x += inputs
        
        return x

