import os
import argparse
import string

import torch

from train import *
from generate import *

def do_train():
  print("Training...")
  print("================================================")
  #TODO make optional input param
  dataset_filename = "data/icelandic_sagas.txt"
  num_epochs = 250
  train(dataset_filename, num_epochs)
  print("================================================")

def do_generate():
  print("Generating...")
  print("================================================")
  #TODO make optional input param
  seed = "A" 
  prediction_length = 1000 
  dataset_filename = "data/icelandic_sagas.txt"
  generate_sample(seed, prediction_length, dataset_filename)
  print("================================================")

if __name__ == "__main__":

  argparser = argparse.ArgumentParser()
  argparser.add_argument('mode', type=str, help="Specify as \"train\" or \"generate\"")
  args = argparser.parse_args()

  if args.mode == "train":
    do_train()
  elif args.mode == "generate":
    do_generate()
  else:
    print ("Mode \"" + args.mode + "\" not recognized. Please specify it as either \"train\" or \"generate\"")