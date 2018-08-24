# lstm_autoencoder

A time serie can be described with an LSTM Autoencoder. Usually, LSTM's are required to have fixed timesteps in order for the decoder part of the autoencoder to know beforehand how many timesteps should produce. However, this version of LSTM Autoencoder allows to describe timeseries based on random samples with unfixed timesteps.

In this LSTM autoencoder version, the decoder part is capable of producing, from an encoded version, as many timesteps as desired, serving the purposes of also predicting future steps.


## Installation

It is required `keras`, `tensorflow` under the hood, `pandas` for the example and `pyfolder` for save/load of the trained model.
They can be installed with pip: 

```bash 
pip3 install -r requirements.txt
```

Tensorflow is not included in the requirements.txt, so it must be manually installed:
```bash 
pip3 install tensorflow
```

## Usage

There is an example in a [example.py](example.py)


