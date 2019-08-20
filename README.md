# SagaRNN

Inspired by Andrej Karpathy's [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), I was curious to see how well a recurrent neural network (RNN) would work for generating [Icelandic Sagas](https://en.wikipedia.org/wiki/Sagas_of_Icelanders). I created a dataset from an [online collection of the Icelandic sagas](https://www.snerpa.is/net/isl/band.htm) and built a neural network in PyTorch. Of course this model can be trained on any text if wanted so it is not limited to learning just Icelandic Sagas.

## Running the code
To train
```
python3 main.py train
```
To generate text from trained model
```
python3 main.py generate --pretrained_model my_model_name.pt
```
To see additional optional arguments
```
python3 main.py -h
```

## Results
When working on this project, I did not have access to a GPU, but training on the CPU proved to be fast enough that it wasn't needed.