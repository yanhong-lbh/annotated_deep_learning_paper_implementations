# https://nn.labml.ai/transformers/mha.html

import math
from typing import Optional, List

import torch
from torch import nn

from labml import tracker

# This class is a building block in a multi-head attention mechanism. 
# The idea behind multi-head attention is to use multiple attention "heads" that
# can focus on different parts of the input simultaneously. By dividing the input 
# into different heads, the model can learn different aspects of the relationships 
# between the elements in the sequence, thereby capturing more complex patterns.

# prepare for multi-head attention

class PrepareForMultiHeadAttention(nn.Module):
    '''
    this module does a linear transformation and slipts the vector into
    given number of heads for multi-head attention.
    THis is used to transform key, query and value vectors 
    '''
    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        # linear layer for linear transformation
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        # number of heads
        self.heads = heads
        # number of dimensions in vectors in each head
        self.d_k = d_k
    
    def forward(self, x: torch.Tensor):
        '''
        input has shape [seq_len, batch_size, d_model]
        or [batch_size, d_model].
        we apply the lienar transformation to the last dimension and 
        split that into the heads.
        
        output has shape [seq_len, batch_size, heads, d_k] or 
        [batch_size, heads, d_model]
        '''
        # store the shape of the tensor without the last dimension
        head_shape = x.shape[:-1]
        # apply a linear transformation to the last dimension of the input tensor
        # changing its size from d_model to heads * d_k
        x = self.linear(x)
        # split last dimension into heads
        # *head_shape unpacks the tuple 'head_shape' 
        # which helps in maintaining the other dimensions of the tensor
        # Then, the last dimension is split into two dimensions, 
        # 'self.heads' and 'self.d_k' 
        # this effectively separates the transformed vectors into differnt heads
        x = x.view(*head_shape, self.heads, self.d_k)
        
        return x
        
class MultiHeadAttention(nn.Module):
    """
    This computes scaled multi-headed attention for given query, key
    and value vectors.
    In simpler terms, it finds keys that matches the query, and gets
    the value of those keys.
    It uses dot-product of query and key as the indicator of how matching
    they are. Before taking the softmax, the dot-products are scaled by 
    1/sqrt(d_k). This is done to avoid large dot-product values casuing
    softmax to give very small gradienct when d_k is large.
    TODO: i don't understand the last sentence
    Softmax is calcualted along the axis of the sequence (or time).
    """
    # heads is the number of heads
    # d_model is the number of features in the query, key and value vectors
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        super().__init__()
        # number of features per head
        self.d_k = d_model // heads
        # number of heads
        self.heads = heads
        # initialize three instances of 'PrepareForMultiHeadAttention' (for query,
        # key, and value)
        # these transform the query, key and value vectors for multi-headed attention
        # [seq_len, batch_size, heads, d_k]
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)
        
        # softmax for attention along the time dimension of key
        self.softmax = nn.Softmax(dim=1)
        
        # output layer
        self.output = nn.Linear(d_model, d_model)
        
        # dropout
        self.dropout = nn.Dropout(dropout_prob)
        
        # scaling factor before the softmax
        self.scale = 1 / math.sqrt(self.d_k)
        
        # we store attentions so that it can be used for logging, 
        # or other computations if needed
        self.attn = None
        
    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        This method can be overriden for other variations like relative attention
        """
        # calculate QK^T
        
        # https://pytorch.org/docs/stable/generated/torch.einsum.html
        
        # The output is computed by multiplying the input operands element-wise, 
        # with their dimensions aligned based on the subscripts, 
        # and then summing out the dimensions whose subscripts are not part of the output
        
        # uses the Einstein summation convention to perform a specific combination 
        # of multiplication and summation across the dimensions of the query and key
        # tensors
        # ibhd: this represents the dimension of the query tensor
            # i: sequence length of the query
            # b: batch size
            # h: number of heads
            # d: dimension of each head's query vector
        # jbhd: This represents the dimensions of the key tensor,
            # j: sequence length of the key (often the same as the sequence length
            # of the query)
            # b, h, d: same as for the query tensor
        # ->ijbh: This specifies the desired output dimensions
            # i: sequence length of the query
            # j: sequence length of the key
            # b: batch size
            # h: number of heads
        # torch.einsum('ibhd, jbhd->ijbh', query, key) calculates the dot product
        # of the query and key vectors for each head and each element in the batch.
        return torch.einsum('ibhd, jbhd->ijbh', query, key)
    
    # A mask is used to selectively ignore certain keys during the attention computation
    # could prevent tbe model from attending to future positions in a sequence
    # or could mask out padding tokens
    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        """
        mask has shape [seq_len_q, seq_len_k, batch_size], where first dimension 
        is the query dimension. If the query dimension is equal to 1 it will be broadbasted.
        
        Resulting mask has shape
        [seq_len_q, seq_len_k, batch_size, heads]
        """
        
        # check that the mask shape is compatible with the shapes of the query and key
        # the first dim of the mask is either 1 (so it can be broadcasted across 
        # the query sequence) or equal to the sequence length of the query
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        # check the second dimension of the mask is equal to the sequence length 
        # of the key
        assert mask.shape[1] == key_shape[0]
        # check the third dimension of the mask is either 1 (so it can be broadcasted
        # across the batch) or equal to the batch size
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]
        
        # same mask applied to all heads
        # add a dimension of size 1 at the end of the tensor 
        # resulting in a shape of [seq_len_q, seq_len_k, batch_size, 1]
        # last dimension represents the 'heads' dimension
        mask = mask.unsqueeze(-1)
        
        return mask
    
    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        query, key and value are the tensors that store collection of query, key
        and value vectors. They have shape 
        [seq_len, batch_size, d_model].
        mask has shape [seq_len, seq_len, batch_size] and mask[i, j, b]
        indicates whether for batch b, query at position i has access to key-value 
        at position j.
        """
        # query, key and value have shape 
        # [seq_len, batch_size, d_model]
        seq_len, batch_size, _ = query.shape
        
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)
        
        # Prepare query, key and value for attention computation.
        # These will then have shape
        # [seq_len, batch_size, heads, d_k]
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        
        # compute attention scores QK^T. This gives a tensor of shape
        # [seq_len, seq_len, batch_size, heads]
        scores = self.get_scores(query, key)
        
        # scale scores
        scores *= self.scale
        
        # apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # softmax attention along the key sequence dimension 
        attn = self.softmax(scores)
        
        # save attentions if debugging
        tracker.debug('attn', attn)
        
        # Apply dropout
        attn = self.dropout(attn)
        
        # multiply by values
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)
        
        # save attentions for any other calculations
        self.attn = attn.detach()
    
        # concatenate multiple heads
        x = x.reshape(seq_len, batch_size, -1)
        
        # output layer
        return self.output(x)
        
    
    
        
        
        