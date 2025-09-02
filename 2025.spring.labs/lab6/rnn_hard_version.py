import torch
from torch import nn
import numpy as np

np.random.seed(2022)
torch.manual_seed(2022)

class GRU(nn.Module):
    def __init__(self, num_inputs, num_hiddens):
        super(GRU, self).__init__()
        # 隐藏层参数
        '''
        请声明GRU中的各类参数
        '''
        # 重置门
        self.W_xz = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens)), dtype=torch.float32))
        self.W_hz = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens, num_hiddens)), dtype=torch.float32))
        self.b_z = torch.nn.Parameter(torch.zeros(num_hiddens))
        # 更新门
        self.W_xr = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens)), dtype=torch.float32))
        self.W_hr = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens, num_hiddens)), dtype=torch.float32))
        self.b_r = torch.nn.Parameter(torch.zeros(num_hiddens))
        # 隐藏状态
        self.W_xh = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens)), dtype=torch.float32))
        self.W_hh = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens, num_hiddens)), dtype=torch.float32))
        self.b_h = torch.nn.Parameter(torch.zeros(num_hiddens))


    def forward(self, inputs, H):
        '''
        利用定义好的参数补全GRU的前向传播，
        不能调用pytorch中内置的GRU函数及操作
        '''
        # ==========
        # todo '''请补全GRU网络前向传播'''
        num_hiddens_size = self.b_z.shape[0]
        B, S, _ = inputs.shape
        outputs = torch.zeros(B, S, num_hiddens_size)

        for i in range(S):
            r_t = torch.sigmoid(
                inputs[:, i] @ self.W_xr + H @ self.W_hr + self.b_r.unsqueeze(0))
            z_t = torch.sigmoid(
                inputs[:, i] @ self.W_xz + H @ self.W_hz + self.b_z.unsqueeze(0))
            h_t_hat = torch.tanh(
                inputs[:, i] @ self.W_xh + (r_t * H) @ self.W_hh + self.b_h.unsqueeze(0))
            H = z_t * H + (1 - z_t) * h_t_hat
            outputs[:, i] = H
        # ==========
        return outputs, H


class Sequence_Modeling(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_outputs, hidden_size):
        super(Sequence_Modeling, self).__init__()
        self.emb_layer = nn.Embedding(vocab_size, embedding_size)
        self.gru_layer = GRU(embedding_size, hidden_size)
        self.linear = nn.Linear(hidden_size, num_outputs)

    def forward(self, sent, state):
        '''
        sent --> (B, S) where B = batch size, S = sequence length
        sent_emb --> (B, S, I) where B = batch size, S = sequence length, I = num_inputs
        state --> (B, 1, H), where B = batch_size, num_hiddens
        你需要利用定义好的emb_layer, gru_layer和linear，
        补全代码实现歌词预测功能，
        sent_outputs的大小应为(B, S, O) where O = num_outputs, state的大小应为(B, 1, H)
        '''

        # ==========
        # todo '''请补全代码'''
        sent_emb = self.emb_layer(sent)
        sent_hidden, state = self.gru_layer(sent_emb, state)
        sent_outputs = self.linear(sent_hidden)
        # ==========
        return sent_outputs, state


if __name__ == '__main__':
    model = Sequence_Modeling()