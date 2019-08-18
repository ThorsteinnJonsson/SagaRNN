import string
import numpy as np
import random

import torch 
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

def read_file(filename):
    with open(filename, 'r') as fid:
      text = fid.read()
    chars = sorted(list(set(text)))
    return text, chars

class Codec():
  def __init__(self, chars):
    char_to_index = {}
    index_to_char = {}
    for i, c in enumerate(chars):
      char_to_index[c] = i
      index_to_char[i] = c
    self.char_to_index = char_to_index
    self.index_to_char = index_to_char

  def encode(self, text):
    return np.array([self.char_to_index[char] for char in text])

  def decode(self, encoded_text):
    return ''.join([self.index_to_char[index] for index in encoded_text])



def get_dataset(input_text, codec):
  # Encode chars as integers
  encoded_text = codec.encode(input_text)

  x_train = torch.from_numpy(encoded_text[0:500])
  y_train = torch.from_numpy(encoded_text[1:501])
  # x_train = torch.from_numpy(encoded_text[:-1]) # TODO re-enable. too slow for testing
  # y_train = torch.from_numpy(encoded_text[1:])
  x_train = x_train.type(torch.LongTensor).unsqueeze(0)
  y_train = y_train.type(torch.LongTensor).unsqueeze(0)

  return x_train, y_train

class SagaDataLoader():
  def __init__(self, x, y, chunk_len, batch_size):
    self.x = x
    self.y = y
    self.chunk_len = chunk_len
    self.batch_size = batch_size
    self.data_len = x.shape[1]
    assert(self.data_len == y.shape[1])

  def get_random_batch(self):
    xb = torch.LongTensor(self.batch_size, self.chunk_len)
    yb = torch.LongTensor(self.batch_size, self.chunk_len)

    for ib in range(self.batch_size):
      start_index = random.randint(0, self.data_len - self.chunk_len)
      end_index = start_index + self.chunk_len
      xb[ib] = self.x[0, start_index:end_index]
      yb[ib] = self.y[0, start_index:end_index]

    return xb, yb
