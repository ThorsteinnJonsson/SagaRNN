import os
import argparse
import string

import torch

from train import *

def do_train():
  train()

def do_generate():
  print ("Generating...")

if __name__ == "__main__":

  argparser = argparse.ArgumentParser()
  argparser.add_argument('mode', type=str, help="Specify as \"train\" or \"generate\"")
  args = argparser.parse_args()

  print("RNN")
  if args.mode == "train":
    do_train()
  elif args.mode == "generate":
    do_generate()
  else:
    print ("Mode \"" + args.mode + "\" not recognized. Please specify it as either \"train\" or \"generate\"")