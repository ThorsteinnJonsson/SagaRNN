import torch
import os

from helpers import *
from model import *

def generate(model, codec, seed, prediction_length, temperature=0.8):
  # Initialize memory for prediction
  prediction = ['0']*(len(seed) + prediction_length)
  for idx, char in enumerate(seed):
    prediction[idx] = char

  encoded_input = torch.from_numpy(codec.encode(seed)).type(torch.LongTensor).unsqueeze(0)

  # Build up hidden state
  batch_size = 1
  hidden = model.init_hidden(batch_size)
  for p in range(len(seed) - 1):
    _, hidden = model(encoded_input[:, p], hidden)

  inp = encoded_input[:,-1]

  for p in range(prediction_length):
    pos = len(seed) + p
    output, hidden = model(inp, hidden)

    # Sample as multinomial distribution to make more varied input
    output_dist = output.data.view(-1).div(temperature).exp()
    top_i = torch.multinomial(output_dist, 1).numpy()

    # Add predicted character to string and use as next input
    predicted_char = codec.decode(top_i)
    prediction[pos] = predicted_char
    inp = torch.from_numpy(codec.encode(predicted_char)).type(torch.LongTensor).unsqueeze(0)

  return ''.join(prediction)


def generate_sample(seed, model_filename, prediction_length, dataset_filename):

  model = load_model(model_filename)

  # Get dictionary
  text, chars = read_file(dataset_filename)
  codec = Codec(chars)

  generated_text = generate(model, codec, seed, prediction_length)

  print("Generated text:")
  print(generated_text)