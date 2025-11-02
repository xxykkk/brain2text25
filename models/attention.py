import torch.nn as nn
import torch


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden_size, head_nums):
        super().__init__()
        self.Q=nn.Linear(hidden_size, hidden_size)
        self.K=nn.Linear(hidden_size, hidden_size)
        self.V=nn.Linear(hidden_size, hidden_size)
        self.linear=nn.Linear(hidden_size, hidden_size)
        self.head_nums=head_nums
        self.head_dim = hidden_size // head_nums
    def forward(self, q, key_value, causal_mask=None, pad_mask=None):
        (bs, N, _) = q.shape
        q=self.Q(q)
        k=self.K(key_value)
        v=self.V(key_value) # (bs, len, dim)
 
        # (bs, len, head_nums, dim/head_nums) -> (bs, head_nums, len, dim/head_nums)
        q=q.reshape(bs, N, self.head_nums, -1).transpose(1,2)
        k=k.reshape(bs, N, self.head_nums, -1).transpose(1,2)
        v=v.reshape(bs, N, self.head_nums, -1).transpose(1,2)
 
        # (len, dim) @ (dim, len) = (len, len)
        qk = q @ k.transpose(-1, -2) / (self.head_dim**0.5)
 
        if causal_mask is not None:
            # causal_mask = 0.  (masked) 或 1. (可见)
            qk = qk.masked_fill(causal_mask == 0, -1e9)
        if pad_mask is not None:
            qk = qk.masked_fill(pad_mask == 0,   -1e9)
 
        qk = torch.softmax(qk, dim=-1)
 
        res = qk @ v  #(bs,head_nums,len,dim/head_nums)
        res = res.transpose(1,2).reshape(bs, N, -1)
 
        res=self.linear(res)
 
        return res


class SelfAttention(nn.Module):
    def __init__(self, hidden_size,):
        super().__init__()
        self.Q=nn.Linear(hidden_size, hidden_size)
        self.K=nn.Linear(hidden_size, hidden_size)
        self.V=nn.Linear(hidden_size, hidden_size)
        self.linear=nn.Linear(hidden_size, hidden_size)
    def forward(self, x, causal_mask=None, pad_mask=None):
        bs, hd = x.shape[0], x.shape[2]
        q=self.Q(x)
        k=self.K(x)
        v=self.V(x) # (bs, len, dim)
 
        # (len, dim) @ (dim, len) = (len, len)
        qk = q @ k.transpose(-1, -2) / (hd**0.5)
 
        if causal_mask is not None:
            # causal_mask = 0.  (masked) 或 1. (可见)
            qk = qk.masked_fill(causal_mask == 0, -1e9)
        if pad_mask is not None:
            qk = qk.masked_fill(pad_mask == 0,   -1e9)
        
        qk = torch.softmax(qk, dim=-1)
 
        res = qk @ v 
 
        res=self.linear(res)
 
        return res


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, head_nums):
        super().__init__()
        self.head_nums=head_nums
        self.hidden_size=hidden_size
        self.head_dim = hidden_size // head_nums
        self.Q=nn.Linear(hidden_size, hidden_size)
        self.K=nn.Linear(hidden_size, hidden_size)
        self.V=nn.Linear(hidden_size, hidden_size)
        self.linear=nn.Linear(hidden_size, hidden_size)
    def forward(self, x, causal_mask=None, pad_mask=None):
        (bs, N, _) = x.shape
        # print(x.shape)
        q=self.Q(x)
        k=self.K(x)
        v=self.V(x) # (bs, len, dim)
        # print(v.shape)
        q=q.reshape(bs, N, self.head_nums, -1).transpose(1,2)
        k=k.reshape(bs, N, self.head_nums, -1).transpose(1,2)
        v=v.reshape(bs, N, self.head_nums, -1).transpose(1,2)

        # print(v.shape)
        # print(pad_mask.shape)
        # input("NEfs")
 
        # (len, dim) @ (dim, len) = (len, len)
        qk = q @ k.transpose(-1, -2) / (self.head_dim**0.5)
 
        if causal_mask is not None:
            # causal_mask = 0.  (masked) 或 1. (可见)
            qk = qk.masked_fill(causal_mask == 0, -1e9)
        if pad_mask is not None:
            qk = qk.masked_fill(pad_mask == 0,   -1e9)
        
        qk = torch.softmax(qk, dim=-1)
 
        res = qk @ v 
        res = res.transpose(1,2).reshape(bs, N, -1)
 
        res=self.linear(res)
 
        return res
 

 
if __name__=='__main__':
    '''
    multi-head attention
    '''
    q=torch.rand(2, 192, 64)
    encoder_output=torch.rand(2, 192, 64)
    # print(x.transpose(-1, -2).shape)
    model=MultiHeadCrossAttention(64, 8)
    res=model(q, encoder_output)
    print(res.shape)

    '''
    self-attention
    '''
    x=torch.rand(2, 192, 64)
    # print(x.transpose(-1, -2).shape)
    model=MultiHeadSelfAttention(64, 8)
    res=model(x)
    print(res.shape)