from torch import nn
import torch
from torch.nn import functional as F
# https://www.cnblogs.com/SkyXZ/p/18927874

class VisionPatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, flatter=True):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
        #self.norm = nn.LayerNorm(embed_dim) # vit官方代码没有
        self.flatter = flatter
        self.num_patches = (image_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)
        if self.flatter:
            x = x.flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, N, C]
        #x = self.norm(x)
        return x
    


class PositionEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)  
    def forward(self, x):
        # img_token_pos_embed = F.interpolate(
        #     img_token_pos_embed, size=self.features_shape, mode='bicubic', align_corners=False
        # )
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        x = torch.cat((cls_tokens, x), dim=1)          # 拼 CLS token
        x = x + self.pos_embed                         # 直接加全部 pos_embed
        x = self.pos_drop(x)

        return x


