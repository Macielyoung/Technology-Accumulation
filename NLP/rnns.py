# RNN series model

import torch
import torch.nn as nn

class RNNs(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(RNNs, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, n_layers)
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers)
        self.bilstm = nn.LSTM(input_size, hidden_size, n_layers, bidirectional=True)
        self.gru = nn.GRU(input_size, hidden_size, n_layers)

    def forward(self, rnn_x, rnn_h0, lstm_x, lstm_h0, lstm_c0, gru_x, gru_h0):
        rnn_output, _ = self.rnn(rnn_x, rnn_h0)
        lstm_output, (hn, cn) = self.lstm(lstm_x, (lstm_h0, lstm_c0))
        bilstm_output, _ = self.bilstm(lstm_x)
        # lstm_output, _ = self.lstm(lstm_x)
        gru_output, _ = self.gru(gru_x, gru_h0)
        return rnn_output, lstm_output, bilstm_output, gru_output

input_size = 5
hidden_size = 10
batch_size = 3
n_layers = 2
seq_len = 8

# seq_len * hidden * embedding(序列输入)
rnn_x = torch.randn(seq_len, batch_size, input_size)
# layer * batch * hidden(隐藏层)
rnn_h0 = torch.randn(n_layers, batch_size, hidden_size)

# seq_len * batch * embedding(序列输入)
lstm_x = torch.randn(seq_len, batch_size, input_size)
# layer * batch * hidden(隐藏状态)
lstm_h0 = torch.randn(n_layers, batch_size, hidden_size)
# layer * batch * hidden(细胞状态)
lstm_c0 = torch.randn(n_layers, batch_size, hidden_size)

# seq_len * batch * embedding(序列输入)
gru_x = torch.randn(seq_len, batch_size, input_size)
# layer * batch * hidden(隐藏状态)
gru_h0 = torch.randn(n_layers, batch_size, hidden_size)

# embedding * hidden * layer
rnns_model = RNNs(input_size, hidden_size, n_layers)
rnn_output, lstm_output, bilstm_output, gru_output = rnns_model(rnn_x, rnn_h0, lstm_x, lstm_h0, lstm_c0, gru_x, gru_h0)
print(rnn_output.shape)
print(lstm_output.shape)
print(bilstm_output.shape)
print(gru_output.shape)