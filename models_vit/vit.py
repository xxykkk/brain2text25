from turtle import forward
from torch import nn
import torch
from torch.nn import functional as F

from attention import SelfAttention
from embedding import VisionPatchEmbedding, PositionEmbedding
from mlp import MLP


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def drop_path(self, x, drop_prob, training):
        if drop_prob == 0. or not training:
            return x
        keep_prob       = 1 - drop_prob
        shape           = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor   = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_() 
        output          = x.div(keep_prob) * random_tensor
        return output

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, attn_drop_rate=0.0, proj_drop_rate=0.0, drop_path = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads, qkv_bias, attn_drop_rate, proj_drop_rate) 
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop_rate=proj_drop_rate)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, num_classes,
                 embed_dim, num_heads, depth, 
                 mlp_ratio=4.0, qkv_bias=False, attn_drop_rate=0.0, proj_drop_rate=0.0):
        super().__init__()
        self.patch_embed = VisionPatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.pos_embed = PositionEmbedding(self.patch_embed.num_patches, embed_dim)
        
        self.pos_drop = nn.Dropout(proj_drop_rate) # 丢弃率
        self.norm = nn.LayerNorm(embed_dim) # 归一化

        self.blocks = nn.Sequential(
            *[
                Block(
                    embed_dim, num_heads, mlp_ratio = mlp_ratio,
                    attn_drop_rate = attn_drop_rate, proj_drop_rate = proj_drop_rate,
                    qkv_bias = qkv_bias, 
                )for i in range(depth)
            ]
        )
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        # x = self.head(x[:, 0])
        return x[:, 0]
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
        



if __name__ == "__main__":
    model = VisionTransformer(224, 16, 3, 
                              num_classes=30, embed_dim=768, 
                              num_heads=12, depth=12, qkv_bias=True, 
                              attn_drop_rate=0.1, proj_drop_rate=0.1)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)

    print(model)