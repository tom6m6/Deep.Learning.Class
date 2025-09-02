import torch
from torch import nn

torch.manual_seed(2022)

class Sequence_Modeling(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_outputs, hidden_size):
        super(Sequence_Modeling, self).__init__()
        self.emb_layer = nn.Embedding(vocab_size, embedding_size)
        self.encoder = nn.RNN(embedding_size, hidden_size, batch_first=True)
        self.mlp1 = nn.Linear(hidden_size, hidden_size)
        self.decoder = nn.RNN(embedding_size+hidden_size, hidden_size, batch_first=True)
        self.softmax = nn.Softmax(dim=2)
        self.mlp2 = nn.Linear(hidden_size, num_outputs)

    def encode(self, enc_x, state):
        enc_emb = self.emb_layer(enc_x)
        enc_hidden, state = self.encoder(enc_emb, state)

        return enc_hidden, state

    def decode(self, dec_y, enc_hidden, state, is_first_step = True):
        '''
        dec_y --> (B, S), where B = batch_size, S = sequence length
        enc_hidden --> (B, S, H), where B = batch_size, S = sequence length, H = hidden_size
        state --> (1, B, H), where B = batch_size, H = hidden_size
        is_first_step --> {True, Flase}, it is True when this is the first step of decoding, otherwise it is False
        请用RNN+attention补全解码器，其中打分函数使用点积模型
        sent_outputs的大小应为(B, S, O) where O = num_outputs, state的大小应为(1, B, H)
        '''

        dec_emb = self.emb_layer(dec_y) # (B, S, E)

        # ==========
        # todo '''请补全代码'''
        # ==========
        dec_outputs = []
        if not is_first_step:
            state = state.transpose(1, 0) # (1, B, H) -> (B, 1, H)
        else:
            # 初始化隐藏状态
            state = self.mlp1(enc_hidden).mean(1).unsqueeze(1) # (B, S, H) -> (B, 1, H)
        

        for t in range(dec_y.shape[1]): # S
            # Attention
            attn_scores = torch.bmm(state, enc_hidden.transpose(2, 1)) # (B, 1, H), (B, H, S) -> (B, 1, S)
            attn_weights = self.softmax(attn_scores)  # (B, 1, S)
            attn = torch.bmm(attn_weights, enc_hidden) # (B, 1, S), (B, S, H) -> (B, 1, H)
            # Context Vector
            context_vec = attn.squeeze(1) # (B, 1, H) -> (B, H)
            dec_input = torch.cat([context_vec, dec_emb[:, t, :]], 1).unsqueeze(1) # (B, H), (B, E) -> (B, H + E) -> (B, 1, H + E)
            dec_output, state = self.decoder(dec_input, state.transpose(0, 1)) # (B, 1, H + E), (1, B, H) -> (B, 1, H), (1, B, H)
            state = state.transpose(0, 1) # (1, B, H) -> (B, 1, H)
            # prediction
            pred = self.mlp2(dec_output) # (B, 1, H) -> (B, 1, O)
            dec_outputs += [pred]

        sent_outputs = torch.cat(dec_outputs, dim = 1) # (B, 1, O) -> (B, S, O)
        return sent_outputs, state


if __name__ == '__main__':
    model = Model_NP()