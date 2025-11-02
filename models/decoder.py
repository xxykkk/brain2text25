from torch import nn
import torch
from .attention import MultiHeadSelfAttention, MultiHeadCrossAttention

class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_size, dropout_prob=0.1):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(hidden_size, num_heads)  # 多头注意力层
        self.dropout1 = nn.Dropout(dropout_prob)  # Dropout 层
        self.layer_norm1 = nn.LayerNorm(hidden_size)  # LayerNorm 层
 
        self.cross_attention = MultiHeadCrossAttention(hidden_size, num_heads)  # 多头注意力层
        self.dropout2 = nn.Dropout(dropout_prob)  # Dropout 层
        self.layer_norm2 = nn.LayerNorm(hidden_size)  # LayerNorm 层
 
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, ff_size),  # 前馈层1
            nn.ReLU(),  # 激活函数
            nn.Linear(ff_size, hidden_size)  # 前馈层2
        )
        self.dropout3 = nn.Dropout(dropout_prob)  # Dropout 层
        self.layer_norm3 = nn.LayerNorm(hidden_size)  # LayerNorm 层
    
    def forward(self, x, encoder_output, tgt_causal_mask=None, tgt_pad_mask=None, src_pad_mask=None):
        # 多头注意力子层
        attn_output = self.self_attention(x, tgt_causal_mask, tgt_pad_mask)  # (batch_size, seq_len, hidden_size)
        attn_output = self.dropout1(attn_output)  # Dropout
        out1 = self.layer_norm1(x + attn_output)  # 残差连接 + LayerNorm
 
        cross_out=self.cross_attention(out1, encoder_output, pad_mask = src_pad_mask)
        cross_attn_output = self.dropout2(cross_out)  # Dropout
        out2 = self.layer_norm2(out1 + cross_attn_output)  # 残差连接 + LayerNorm
        
        # 前馈神经网络子层
        ff_output = self.feed_forward(out2)  # (batch_size, seq_len, hidden_size)
        ff_output = self.dropout3(ff_output)  # Dropout
        out3 = self.layer_norm3(out2 + ff_output)  # 残差连接 + LayerNorm
        
        return out3
 

if __name__=='__main__':
    x=torch.rand(2, 192, 64)
    encoder_output=torch.rand(2, 192, 64)
    model=DecoderLayer(64, 8, 32)
    print(model(x, encoder_output).shape)
