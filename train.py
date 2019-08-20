import torch
import torch.nn as nn

import os

from model import *
from helpers import *
  
  
class Trainer():
  def __init__(self, model_load_path=""):
    self.is_model_loaded = False
    if (len(model_load_path) > 0):
      self.model = load_model(model_load_path)
      self.is_model_loaded = True
      print("Loaded pre-existing model from:\n {}".format(model_load_path))
    else:
      print("No pre-existing model specified, training will start from scratch.")

  def train(self, dataset_filename, 
                  num_epochs, 
                  batch_size,
                  chunk_len,
                  learning_rate):
    
    # Prepare data and make data loader
    text, chars = read_file(dataset_filename)
    codec = Codec(chars)
    data_loader = SagaDataLoader(text, 
                                 codec, 
                                 chunk_len, 
                                 batch_size)

    # Set up model
    if not (self.is_model_loaded):
      input_size = len(chars)
      output_size = len(chars)
      hidden_size = 100
      num_layers = 2
      self.model = SagaRNN(input_size,
                           hidden_size,
                           output_size,
                           num_layers)
    
    optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("Training for {} epochs...".format(num_epochs))
    for epoch in range(num_epochs):

      xb, yb = data_loader.get_random_batch();

      hidden = self.model.init_hidden(batch_size)
      
      loss = 0
      for cid in range(chunk_len):
        # Forward pass
        output, hidden = self.model(xb[:,cid], hidden)
        loss += criterion(output, yb[:,cid])

      print("Epoch #{}: Training loss {}".format(epoch, loss/chunk_len))

      # TODO validation loss

      # Backwards pass and optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # Save model after training
    print("Saving model (twice, once with timestamp to not overwrite)...")
    model_filename, ts_model_filename = save_model(self.model)
    print("Saved model as {}".format(model_filename))
    print("Saved model as {}".format(model_filename))









