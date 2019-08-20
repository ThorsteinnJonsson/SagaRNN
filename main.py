import argparse

from train import *
from generate import *

def get_args():
  argparser = argparse.ArgumentParser()
  argparser.add_argument('mode', type=str, help="Specify as \"train\" or \"generate\"")
  argparser.add_argument('--pretrained_model', type=str, default="")
  argparser.add_argument('--dataset_filename', type=str, default="data/icelandic_sagas.txt")
  # Training-specific args
  argparser.add_argument('--num_epochs', type=int, default=250)
  argparser.add_argument('--batch_size', type=int, default=100)
  argparser.add_argument('--chunk_len', type=int, default=200)
  argparser.add_argument('--learning_rate', type=float, default=0.01)
  
  # Generate-specific args
  argparser.add_argument('--prediction_len', type=int, default=1000)
  argparser.add_argument('--seed', type=str, default="A")

  return argparser.parse_args()


def do_train(args):
  print("Training...")
  print("================================================")
  trainer = Trainer()
  # trainer = Trainer("saga_model.pt")
  trainer.train(args.dataset_filename, 
                args.num_epochs, 
                args.batch_size, 
                args.chunk_len,
                args.learning_rate)
  print("================================================")

def do_generate(args):
  print("Generating...")
  print("================================================")
  args.model_filename = 'saga_model.pt' # TODO remove
  generate_sample(args.seed, 
                  args.model_filename, 
                  args.prediction_len, 
                  args.dataset_filename)
  print("================================================")

if __name__ == "__main__":

  args = get_args()

  if args.mode == "train":
    do_train(args)
  elif args.mode == "generate":
    do_generate(args)
  else:
    print ("Mode \"" + args.mode + "\" not recognized. Please specify it as either \"train\" or \"generate\"")