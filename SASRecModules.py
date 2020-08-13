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


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate, state_size, block_id, prev_blk):
        super(SelfAttentionBlock, self).__init__()
        self.block_id = block_id
        self.prev_blk = prev_blk
        # Multihead Attention Layer
        self.multihead_attention = self.create_attention(hidden_size, num_heads, dropout_rate)
        self.multihead_attention_norm = LayerNorm(hidden_size)
        # Feedforward Layer
        self.feedforward = self.create_feedforward(hidden_size, dropout_rate, state_size)
        self.feedforward_norm = LayerNorm(hidden_size)

    def create_attention(self, hidden_size, num_heads, dropout_rate):
        return MultiheadAttention(num_units=hidden_size,
                                  num_heads=num_heads, dropout_rate=dropout_rate,
                                  causality=True, with_qk=False, hidden_size=hidden_size)

    def create_feedforward(self, hidden_size, dropout_rate, state_size):
        return Feedforward(in_channels=state_size - 1, num_units=[hidden_size, hidden_size],
                           dropout_rate=dropout_rate)

    def forward(self, x):
        y = self.multihead_attention(queries=self.multihead_attention_norm(x), keys=x)
        y = self.feedforward(self.feedforward_norm(y))
        return y


class SelfAttentionBlockAdjacentBlock(SelfAttentionBlock):
    def __init__(self, hidden_size, num_heads, dropout_rate, state_size, block_id, prev_blk):
        super(SelfAttentionBlockAdjacentBlock, self).__init__(hidden_size, num_heads, dropout_rate, state_size,
                                                              block_id, prev_blk)

    def create_attention(self, hidden_size, num_heads, dropout_rate):
        if self.prev_blk is not None and self.is_adjacent_with_previous(self.prev_blk, self.block_id):
            return self.prev_blk.multihead_attention
        return MultiheadAttention(num_units=hidden_size,
                                  num_heads=num_heads, dropout_rate=dropout_rate,
                                  causality=True, with_qk=False, hidden_size=hidden_size)

    def create_feedforward(self, hidden_size, dropout_rate, state_size):
        if self.prev_blk is not None and self.is_adjacent_with_previous(self.prev_blk, self.block_id):
            return self.prev_blk.feedforward
        return Feedforward(in_channels=state_size - 1, num_units=[hidden_size, hidden_size],
                           dropout_rate=dropout_rate)

    def is_adjacent_with_previous(self, prev_block, block_id):
        return (block_id // 2) == (prev_block.block_id // 2)


class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_units=None, num_heads=8,dropout_rate=0,causality=False,
                        with_qk=False, previous_layer=None):
        super(MultiheadAttention,self).__init__()
        self.num_units=num_units
        self.num_heads = num_heads
        self.dropout_rate=dropout_rate
        self.causality = causality
        self.with_qk=with_qk
        self.hidden_size=hidden_size
        if previous_layer is None:
            self.fc1 = nn.Linear(self.hidden_size,num_units)
            self.fc2 = nn.Linear(self.hidden_size,num_units)
            self.fc3 = nn.Linear(self.hidden_size,num_units)
        else:
            self.fc1 = previous_layer.fc1
            self.fc2 = previous_layer.fc2
            self.fc3 = previous_layer.fc3
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
        outputs = outputs * query_masks # broadcasting. (N, T_q, C)
    
        # Dropouts
        
        outputs = self.dropout(outputs)
               
        # Weighted sum
        outputs = torch.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        o_split = int(outputs.size(0)/self.num_heads)
        outputs = torch.cat(torch.split(outputs, o_split, dim=0), dim=2 ) # (N, T_q, C)
              
        # Residual connection
        outputs = outputs + queries
              
        # Normalize
        #outputs = normalize(outputs) # (N, T_q, C)
 
        if self.with_qk: return Q,K
        else: return outputs
        
        
class Feedforward(nn.Module):
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
    def __init__(self, in_channels, num_units=[2048, 512], dropout_rate=0.2, previous_layer=None):
        super(Feedforward,self).__init__()
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        if previous_layer is None:
            self.conv1 = nn.Conv1d(num_units[0], num_units[0], 1)
            self.conv2 = nn.Conv1d(num_units[0], num_units[0], 1)
        else:
            self.conv1 = previous_layer.conv1
            self.conv2 = previous_layer.conv2

        
    def forward (self,inputs):
        x = inputs.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.dropout2(x)

        x = x.permute(0, 2, 1)
        x = x + inputs
        
        return x


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, epsilon=1e-8):
        super(LayerNorm, self).__init__()
        self.epsilon = epsilon
        self.beta = nn.Parameter(torch.zeros(hidden_size, requires_grad=True))
        self.gamma = nn.Parameter(torch.ones(hidden_size, requires_grad=True))

    def forward(self, x):
        shape = x.shape

        mean = x.mean(dim=len(shape)-1, keepdim=True)
        variance = x.var(dim=len(shape)-1, keepdim =True)

        x = (x - mean) / torch.sqrt(variance + self.epsilon)
        y = self.gamma * x + self.beta

        return y