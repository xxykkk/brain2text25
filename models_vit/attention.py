from torch import nn
import torch
from torch.nn import functional as F

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop_rate=0.0, proj_drop_rate=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.out = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = q @ k.transpose(-2, -1) * self.scale #(B, num_head, N, C) @ (B, num_head, C, N) = (B, num_head, N, N) 
        attn = attn.softmax(dim=-1) #(B, num_head, N, N)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).reshape(B,N,C) # (B, num_head, N, N) @ (B, num_head, N, C) = (B, num_head, N, C)
        x = self.out(x)
        x = self.out_drop(x)

        return x



if __name__ == "__main__":
    x = torch.randn(1, 3, 768)
    model = SelfAttention(dim = 768, num_heads = 12)
    y = model(x)
    print(y.shape)