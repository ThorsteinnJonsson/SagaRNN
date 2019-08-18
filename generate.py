import torch
import os

from helpers import *
from model import *

def load_model():
  model_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model')
  model_filename = os.path.join(model_directory, 'saga_model.pt')
  model = torch.load(model_filename)
  return model

# TODO seed, prediction length as input parameters
def generate(model, codec, seed='A', predict_len=100, temperature=0.8):
  batch_size = 1
  prediction = seed

  encoded_input = torch.from_numpy(codec.encode(seed)).type(torch.LongTensor).unsqueeze(0)

  hidden = model.init_hidden(batch_size)

  # Build up hidden state
  for p in range(len(seed) - 1):
    _, hidden = model(encoded_input[:, p], hidden)

  inp = encoded_input[:,-1]

  for p in range(predict_len):
    output, hidden = model(inp, hidden)

    # Sample as multinomial distribution to make more varied input
    output_dist = output.data.view(-1).div(temperature).exp()
    top_i = torch.multinomial(output_dist, 1).numpy()

    # Add predicted character to string and use as next input
    predicted_char = codec.decode(top_i)
    prediction += predicted_char
    inp = torch.from_numpy(codec.encode(predicted_char)).type(torch.LongTensor).unsqueeze(0)

  return prediction


def generate_sample():

  model = load_model()

  # Get dictionary
  # TODO save when training and only load dictionary instead of creating it again every time 
  dataset_filename = "data/shakespear.txt"
  text, chars = read_file(dataset_filename)
  codec = Codec(chars)

  generated_text = generate(model, codec)

  print("Generated text:")
  print(generated_text)