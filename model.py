import torch
import torch.nn as nn
from torch.autograd import Variable

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # TODO refactor

class SagaRNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, num_layers):
    super(SagaRNN, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.encoder = nn.Embedding(input_size, hidden_size)

    self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers)
    self.decoder = nn.Linear(hidden_size, output_size)

  def forward(self, x, hidden):
    batch_size = x.size(0)

    encoded = self.encoder(x)
    out, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
    out = self.decoder(out.view(batch_size, -1))

    return out, hidden

  def init_hidden(self, batch_size):
    return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_size))


