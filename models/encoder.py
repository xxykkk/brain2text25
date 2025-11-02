from torch import nn
from .attention import MultiHeadSelfAttention
import torch

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_size, dropout_prob=0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadSelfAttention(hidden_size, num_heads)  # 多头注意力层
        self.dropout1 = nn.Dropout(dropout_prob)  # Dropout 层
        self.layer_norm1 = nn.LayerNorm(hidden_size)  # LayerNorm 层
 
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, ff_size),  # 前馈层1
            nn.ReLU(),  # 激活函数
            nn.Linear(ff_size, hidden_size)  # 前馈层2
        )
        self.dropout2 = nn.Dropout(dropout_prob)  # Dropout 层
        self.layer_norm2 = nn.LayerNorm(hidden_size)  # LayerNorm 层
    
    def forward(self, x, pad_mask=None):
        # 多头注意力子层
        attn_output = self.multi_head_attention(x, pad_mask=pad_mask)  # (batch_size, seq_len, hidden_size)
        attn_output = self.dropout1(attn_output)  # Dropout
        out1 = self.layer_norm1(x + attn_output)  # 残差连接 + LayerNorm
        
        # 前馈神经网络子层
        ff_output = self.feed_forward(out1)  # (batch_size, seq_len, hidden_size)
        ff_output = self.dropout2(ff_output)  # Dropout
        out2 = self.layer_norm2(out1 + ff_output)  # 残差连接 + LayerNorm
        
        return out2


if __name__ == '__main__':
    x=torch.rand(2, 192, 64) # batch_size, seq_len, hidden_size
    model=EncoderLayer(64, 8, 32) # hidden_size, num_heads, ff_size
    print(model(x).shape)
