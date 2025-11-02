from .attention import MultiHeadSelfAttention, MultiHeadCrossAttention
from torch import nn
import torch
from .encoder import EncoderLayer
from .decoder import DecoderLayer
from .embedding import TokenEmbedding, PositionalEmbedding
import math

class Transformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_len = 3000,
                 hidden_size=512,
                 num_heads=8,
                 num_encoder_layers=4,
                 num_decoder_layers=5,
                 ff_size=1024,
                 dropout=0.3,
                 pad_idx=0):
        super().__init__()

        self.pad_idx = pad_idx
        self.hidden_size = hidden_size

        # Embedding
        self.token_emb = TokenEmbedding(vocab_size, hidden_size)
        self.pos_emb   = PositionalEmbedding(max_len, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(hidden_size, num_heads, ff_size, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(hidden_size, num_heads, ff_size, dropout)
            for _ in range(num_decoder_layers)
        ])

        # 输出映射到词表
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    # ---------------- Utilities ----------------
    def _make_pad_mask(self, seq, seq_len_dim=1):
        """返回 [bs, 1, 1, seq_len] 的 mask；1 表示可见，0 表示 pad"""
        return (seq != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def _make_causal_mask(self, size, device):
        """生成下三角 causal mask: [size, size]"""
        mask = torch.tril(torch.ones(size, size, device=device)).bool()
        return mask

    # --------------- Forward (训练 / 推理) ---------------
    def forward(self, src, tgt, src_pad=None):
        """
        src: [bs, src_len]  输入
        tgt: [bs, tgt_len]  目标（teacher forcing）
        返回 logits: [bs, tgt_len, vocab_size]
        """
        bs, src_len = src.shape[0], src.shape[1]
        tgt_len = tgt.shape[1]
        device = src.device

        # 1) Embedding & Positional
        src_emb = self.pos_emb(src)  
        #src_emb = self.dropout(self.pos_emb(self.token_emb(src) * math.sqrt(self.hidden_size))) # [bs, src_len, hidden_size]
        tgt_emb = self.dropout(self.pos_emb(self.token_emb(tgt) * math.sqrt(self.hidden_size))) # [bs, tgt_len, hidden_size]

        # 2) padding mask
        src_pad_mask = self._make_pad_mask(src if src_pad is None else src_pad)# [bs, 1, 1, src_len]src_pad_mask = self._make_pad_mask(src)   # [bs, 1, 1, src_len]
        tgt_pad_mask = self._make_pad_mask(tgt)   # [bs, 1, 1, tgt_len]

        # 3) Encoder
        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, pad_mask=src_pad_mask)    # keep shape

        # 4) Decoder causal mask
        causal_mask = self._make_causal_mask(tgt_len, device)   # [tgt_len, tgt_len]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)     # [1,1,tgt_len,tgt_len]

        # print(enc_out.shape)

        # 5) Decoder
        dec_out = tgt_emb
        for layer in self.decoder_layers:
            dec_out = layer(dec_out,
                            enc_out,
                            tgt_causal_mask=causal_mask,
                            tgt_pad_mask=tgt_pad_mask,
                            src_pad_mask=src_pad_mask)

        # 6) 输出到词表
        logits = self.lm_head(dec_out)     # [bs, tgt_len, vocab]
        return logits


if __name__=='__main__':
    #x=torch.randint(0, 42, (2, 2500))
    x = torch.rand(2, 2500, 512)
    src_pad = torch.randint(0, 1, (2, 2500)).long()

    tgt=torch.randint(0, 42, (2, 2500))
    model=Transformer(43, 2500, hidden_size = 512,)
    print(model(x, tgt, src_pad).shape)


# torch.Size([2, 8, 2500, 2500]) torch.Size([2, 1, 1, 2500, 512])
# torch.Size([2, 8, 2500, 2500]) torch.Size([2, 1, 1, 2500])