import torch
import torch.nn as nn

import os
import time

def save_model(model):
  model_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model')
  if not os.path.exists(model_directory):
    os.mkdir(model_directory)
  timestamped_model_filename = os.path.join(model_directory, time.strftime("%Y%m%d-%H%M%S") + '_saga_model.pt')
  model_filename = os.path.join(model_directory, 'saga_model.pt')
  torch.save(model, timestamped_model_filename)
  torch.save(model, model_filename)
  return model_filename, timestamped_model_filename


def load_model(model_filename):
  model_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model')
  model_filename = os.path.join(model_directory, model_filename)
  model = torch.load(model_filename)
  return model

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


