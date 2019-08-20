import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import string
import time
from torch.utils.data import DataLoader


from model import *
from helpers import *
  
  


def train(dataset_filename, num_epochs):
  # TODO make input param
  chunk_len = 200
  batch_size = 100
  learning_rate = 0.01
  
  # Prepare data and make data loader
  text, chars = read_file(dataset_filename)
  codec = Codec(chars)
  x_train, y_train = get_dataset(text, codec)
  data_loader = SagaDataLoader(x_train, 
                               y_train, 
                               chunk_len, 
                               batch_size)

  # Set up model
  input_size = len(chars)
  output_size = len(chars)
  hidden_size = 100
  num_layers = 2
  saga_model = SagaRNN(input_size,
                       hidden_size,
                       output_size,
                       num_layers)
  
  optimizer = torch.optim.Adam(saga_model.parameters(), lr=learning_rate)
  criterion = nn.CrossEntropyLoss()

  print("Training for {} epochs...".format(num_epochs))
  for epoch in range(num_epochs):

    xb, yb = data_loader.get_random_batch();

    hidden = saga_model.init_hidden(batch_size)
    
    loss = 0
    for cid in range(chunk_len):
      # Forward pass
      output, hidden = saga_model(xb[:,cid], hidden)
      loss += criterion(output, yb[:,cid])

    print("Epoch #{}: Loss {}".format(epoch, loss/chunk_len))

    # Backwards pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  # Save model after training
  print("Saving model (twice, once with timestamp to not overwrite)...")
  model_filename, ts_model_filename = save_model(saga_model)
  print("Saved model as {}".format(model_filename))
  print("Saved model as {}".format(model_filename))









