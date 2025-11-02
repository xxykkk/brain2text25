from torch import nn
import torch
import math
from transformers import AutoTokenizer, AutoModelForMaskedLM



class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)  # 嵌入层
    
    def forward(self, x):
        # x 形状: (batch_size, seq_len)
        embedded = self.embedding(x)  # 嵌入后的形状: (batch_size, seq_len, hidden_size)
        return embedded
 
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 创建位置编码表，大小为 (max_len, hidden_size)
        # position: (max_len, 1)，表示序列中的位置索引，例如 [[0.], [1.], [2.], ...]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # div_term: (hidden_size / 2)，用于计算位置编码的分母
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        
        # 初始化位置编码矩阵 pe 为零矩阵，大小为 (max_len, hidden_size)
        pe = torch.zeros(1, max_len, hidden_size)
        
        # 计算位置编码矩阵，广播机制将 dive_term 扩展为 (1, hidden_size )
        # 偶数索引列使用 sin 函数
        # print(position.shape, div_term.shape) # torch.Size([2500, 1]) torch.Size([256])
        # print((position * div_term).shape) #torch.Size([2500, 256])
        # print(torch.sin(position * div_term).shape) #torch.Size([2500, 256])
        # input("pos_emb")
        pe[0, :, 0::2] = torch.sin(position * div_term)
        # 奇数索引列使用 cos 函数
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        # 将位置编码矩阵注册为 buffer，模型训练时不会更新它
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x 的形状: (batch_size, seq_len, hidden_size)
        seq_len = x.size(1)
        
        # 将位置编码加到输入张量上
        # self.pe[:seq_len, :] 的形状为 (seq_len, hidden_size)
        # unsqueeze(0) 使其形状变为 (1, seq_len, hidden_size)，便于与输入张量相加
        pe = self.pe[:, :seq_len, :].to(x.dtype)
        x = x + pe
        
        # 返回加上位置编码后的张量
        return x
    


def test_previous():
    # 1) 取一个真正大模型的词表（BERT base）
    # Load model directly
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    # model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")

    vocab_size = tokenizer.vocab_size          # 30522
    hidden_size = 768                          # 跟 bert-base-uncased 一致

    # 2) 实例化自定义的 TokenEmbedding
    model = TokenEmbedding(vocab_size, hidden_size)

    # 3) 准备若干句子
    sentences = [
        "Transformers are amazing!",
        "Let's test our token embedding layer."
    ]

    # 4) 使用 tokenizer 转成 token ids；同时统一 sequence length（padding）
    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"]           # shape: (batch_size, seq_len)

    print("Token IDs:\n", input_ids)
    print("Shape:", input_ids.shape)

    # 5) 喂进嵌入层
    with torch.no_grad():
        embedded = model(input_ids)            # (batch_size, seq_len, hidden_size)

    print("Embedded Tensor Shape:", embedded.shape, '\n')
    
    embedding_table = model.embedding.weight           # (vocab_size, hidden_size)

    # -------- 从 embedded 反推出 token id --------------
    # 把 (batch, seq_len, hidden) 拉平成 (batch*seq_len, hidden)
    flat_emb = embedded.view(-1, hidden_size)          # (B*S, H)

    # 点积得到相似度，越大越相似
    sim = torch.matmul(flat_emb, embedding_table.t())  # (B*S, V)
    print('embedding_table.shape:', embedding_table.shape)

    # 对 vocab 维度取 argmax 得到最相似的 token id
    pred_ids = torch.argmax(sim, dim=-1)               # (B*S, )

    # reshape 回原来的 (batch, seq_len)
    pred_ids = pred_ids.view(embedded.size(0), embedded.size(1))

    print("\nRecovered ids:")
    print(pred_ids)

    # -------- 2. 解码成字符串 -----------------------------
    decoded_text = tokenizer.batch_decode(
        pred_ids,
        skip_special_tokens=False,    # 为了对照，先不去掉 [PAD]/[CLS]/[SEP]
        clean_up_tokenization_spaces=True
    )

    print('original text')
    print(sentences)
    
    print("\nDecoded text:")
    for i, sent in enumerate(decoded_text):
        print(f"{i}: {sent}")

    
    pos_emb_model = PositionalEmbedding(vocab_size, hidden_size)
    pos_emb_embedded = pos_emb_model(embedded)
    print('\npos_emb_embedded.shape:', pos_emb_embedded.shape)


if __name__ == "__main__":
    test_previous()
    
