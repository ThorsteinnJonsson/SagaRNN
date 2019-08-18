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
  
  
def save_model(model):
  model_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model')
  if not os.path.exists(model_directory):
    os.mkdir(model_directory)
  timestamped_model_filename = os.path.join(model_directory, time.strftime("%Y%m%d-%H%M%S") + '_saga_model.pt')
  model_filename = os.path.join(model_directory, 'saga_model.pt')
  torch.save(model, timestamped_model_filename)
  torch.save(model, model_filename)
  return model_filename, timestamped_model_filename


def train():
  dataset_filename = "data/shakespear.txt"
  chunk_len = 200
  batch_size = 7
  num_epochs = 3
  
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
  hidden_size = 100
  output_size = len(chars)
  num_layers = 2
  saga_model = SagaRNN(input_size,
                       hidden_size,
                       output_size,
                       num_layers)
  
  learning_rate = 0.01
  optimizer = torch.optim.Adam(saga_model.parameters(), lr=learning_rate)

  criterion = nn.CrossEntropyLoss()

  saga_model.cpu(); # don't have a GPU :(
  
  print("Training for {} epochs...".format(num_epochs))

  for epoch in range(num_epochs):

    # Get random training data from dataset #TODO
    xb, yb = data_loader.get_random_batch();

    hidden = saga_model.init_hidden(batch_size)
    saga_model.zero_grad()
    loss = 0
    
    for c in range(chunk_len):
      # Forward pass
      output, hidden = saga_model(xb[:,c], hidden)
      loss += criterion(output, yb[:,c])

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









