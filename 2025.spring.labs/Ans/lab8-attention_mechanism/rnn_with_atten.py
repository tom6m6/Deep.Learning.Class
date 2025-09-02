# import torch
# from torch import nn

# torch.manual_seed(2022)

# class Sequence_Modeling(nn.Module):
#     def __init__(self, vocab_size, embedding_size, num_outputs, hidden_size):
#         super(Sequence_Modeling, self).__init__()
#         self.emb_layer = nn.Embedding(vocab_size, embedding_size)
#         self.encoder = nn.RNN(embedding_size, hidden_size, batch_first=True)
#         self.mlp1 = nn.Linear(hidden_size, hidden_size)
#         self.decoder = nn.RNN(embedding_size+hidden_size, hidden_size, batch_first=True)
#         self.softmax = nn.Softmax(dim=2)
#         self.mlp2 = nn.Linear(hidden_size, num_outputs)

#     def encode(self, enc_x, state):
#         enc_emb = self.emb_layer(enc_x)
#         enc_hidden, state = self.encoder(enc_emb, state)

#         return enc_hidden, state

#     def decode(self, dec_y, enc_hidden, state, is_first_step = True):
#         '''
#         dec_y --> (B, S), where B = batch_size, S = sequence length
#         enc_hidden --> (B, S, H), where B = batch_size, S = sequence length, H = hidden_size
#         state --> (1, B, H), where B = batch_size, H = hidden_size
#         is_first_step --> {True, Flase}, it is True when this is the first step of decoding, otherwise it is False
#         请用RNN+attention补全解码器，其中打分函数使用点积模型
#         sent_outputs的大小应为(B, S, O) where O = num_outputs, state的大小应为(1, B, H)
#         '''

#         dec_emb = self.emb_layer(dec_y)

#         # ==========
#         # todo '''请补全代码'''
#         # ==========
#         trg_len = dec_y.shape[1]

#         outputs = []
#         if is_first_step:
#             state = self.mlp1(enc_hidden).mean(1).unsqueeze(1)
#             # print(state.shape)
#         else:
#             state = state.transpose(1, 0)


#         for t in range(trg_len):
#             scores = torch.bmm(state, enc_hidden.transpose(2, 1))
            
#             # 计算attention weights
#             alpha = self.softmax(scores)  # (batch_size, 1, seq_len)

#             # 计算context vector
#             cont_vec = torch.bmm(alpha, enc_hidden).squeeze(1)
#             input_vec = torch.cat([cont_vec, dec_emb[:, t, :]], 1).unsqueeze(1)
#             sent_hidden, state = self.decoder(input_vec, state.transpose(0, 1))
#             state = state.transpose(0, 1)

#             # 输出预测字符
#             pred = self.mlp2(sent_hidden)
#             outputs += [pred]

#         sent_outputs = torch.cat(outputs, dim = 1)
#         return sent_outputs, state


# if __name__ == '__main__':
#     model = Sequence_Modeling()


import torch 
from torch import nn

torch.manual_seed(2022)

class Sequence_Modeling(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_outputs, hidden_size):
        super(Sequence_Modeling, self).__init__()
        
        # 字符嵌入层，将词表索引映射为向量 (V,) → (E,)
        self.emb_layer = nn.Embedding(vocab_size, embedding_size)  # 输入: (B, S) → 输出: (B, S, E)

        # 编码器 RNN：输入 embedding_size，输出 hidden_size
        self.encoder = nn.RNN(embedding_size, hidden_size, batch_first=True)  # 输入: (B, S, E) → 输出: (B, S, H)

        # 用于初始化 decoder 初始状态 h0 的 MLP，H → H
        self.mlp1 = nn.Linear(hidden_size, hidden_size)

        # 解码器 RNN，输入维度为 embedding + context，即 E + H → H
        self.decoder = nn.RNN(embedding_size + hidden_size, hidden_size, batch_first=True)

        # 注意力分数的 softmax，用于计算 α (attention weights)
        self.softmax = nn.Softmax(dim=2)

        # 输出层，将 decoder hidden 映射为分类概率分布 H → V
        self.mlp2 = nn.Linear(hidden_size, num_outputs)

    def encode(self, enc_x, state):
        # enc_x: (B, S)，每个字符是索引
        enc_emb = self.emb_layer(enc_x)  # (B, S, E)，字符转向量
        enc_hidden, state = self.encoder(enc_emb, state)  # enc_hidden: (B, S, H)，state: (1, B, H)

        return enc_hidden, state

    def decode(self, dec_y, enc_hidden, state, is_first_step=True):
        '''
        dec_y: (B, S) — 解码器输入的索引序列
        enc_hidden: (B, S, H) — encoder 输出的所有隐藏状态（z₁...zₛ）
        state: (1, B, H) — 解码器初始隐藏状态
        is_first_step: 是否为解码起始，用于初始化状态
        '''

        # 获取 decoder 的 embedding，形状为 (B, S, E)
        dec_emb = self.emb_layer(dec_y)

        trg_len = dec_y.shape[1]  # S: 序列长度

        outputs = []

        if is_first_step:
            # 初始化 decoder 的隐藏状态 h₀：
            # 对 encoder 每一时刻的输出 zₜ 先过 mlp1，再在时间维度取平均
            # enc_hidden: (B, S, H) → mlp1 → (B, S, H) → mean(1) → (B, H) → unsqueeze(1) → (B, 1, H)
            state = self.mlp1(enc_hidden).mean(1).unsqueeze(1)
        else:
            # 后续步骤需要把 state 转换成 decoder 输入要求的 (B, 1, H)
            state = state.transpose(1, 0)

        for t in range(trg_len):
            # 计算 attention scores（点积）：当前 decoder 状态和 encoder 所有隐藏状态的相似度
            # state: (B, 1, H)，enc_hidden.transpose(2, 1): (B, H, S)
            # → scores: (B, 1, S)
            scores = torch.bmm(state, enc_hidden.transpose(2, 1))

            # attention 权重 α：归一化分数 (B, 1, S)
            alpha = self.softmax(scores)

            # 根据注意力权重加权 encoder 输出，得到 context 向量
            # alpha: (B, 1, S), enc_hidden: (B, S, H) → cont_vec: (B, 1, H) → squeeze(1): (B, H)
            cont_vec = torch.bmm(alpha, enc_hidden).squeeze(1)

            # 将 context 向量与当前解码器输入向量拼接，作为 decoder RNN 的输入
            # dec_emb[:, t, :] = (B, E), cont_vec = (B, H) → cat: (B, H+E) → unsqueeze(1): (B, 1, H+E)
            input_vec = torch.cat([cont_vec, dec_emb[:, t, :]], 1).unsqueeze(1)

            # decoder 解码一步：输入 input_vec，hidden 为当前状态
            # input_vec: (B, 1, H+E), state.transpose(0,1): (1, B, H)
            sent_hidden, state = self.decoder(input_vec, state.transpose(0, 1))  # sent_hidden: (B, 1, H)

            # 把 state 再转回 (B, 1, H) 用于下一步 attention 计算
            state = state.transpose(0, 1)

            # 通过输出层预测每一类字符的概率
            # sent_hidden: (B, 1, H) → pred: (B, 1, V)
            pred = self.mlp2(sent_hidden)

            outputs += [pred]

        # 将每个时间步的预测拼接起来 → (B, S, V)
        sent_outputs = torch.cat(outputs, dim=1)

        # 返回预测分布和最终隐藏状态（(B, S, V), (1, B, H)）
        return sent_outputs, state


if __name__ == '__main__':
    model = Sequence_Modeling()
