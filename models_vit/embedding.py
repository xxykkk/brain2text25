from torch import nn

# https://www.cnblogs.com/SkyXZ/p/18927874

class VisionPatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, flatter=True):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.flatter = flatter

    def forward(self, x):
        x = self.proj(x)
        if self.flatter:
            x = x.flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, N, C]
        x = self.norm(x)
        return x