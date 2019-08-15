import os
import argparse

def Train():
  print("Training...")

def Generate():
  print ("Generating...")

if __name__ == "__main__":

  argparser = argparse.ArgumentParser()
  argparser.add_argument('mode', type=str, help="Specify as \"train\" or \"generate\"")
  args = argparser.parse_args()

  print("RNN")
  if args.mode == "train":
    Train()
  elif args.mode == "generate":
    Generate()
  else:
    print ("Mode \"" + args.mode + "\" not recognized. Please specify it as either \"train\" or \"generate\"")