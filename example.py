#!/usr/bin/python3
# -*- coding: utf-8 -*-

# MIT License
#
# Copyright (c) 2018 Iván de Paz Centeno
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pandas as pd
from lstm_autoencoder import LSTMAutoencoder
import numpy as np

__author__ = "Iván de Paz Centeno"
__version__ = "0.0.1"


# This example shows how to use the autoencoder on a dynamic time series

number_of_features = 10
latent_space = 4

autoencoder = LSTMAutoencoder(latent_space=latent_space, input_features=number_of_features)

# As an example, let's generate random sampled data, whose values are between [-1, 1].


def generate_sample(timesteps, features):
    return np.tanh(np.random.standard_normal(size=(1, timesteps, features)))


input1 = generate_sample(timesteps=2, features=number_of_features)
input2 = generate_sample(timesteps=4, features=number_of_features)


# input1 and input2 are two samples with different timesteps: the first has 2 timesteps and the second 4; however
# they share the same number of features.

def display(data):
    if len(data.shape) == 3:
        data = data[0]

    print(pd.DataFrame(data))


# Let's see how the data looks like:
print("\nFor input1:")
display(input1)

print("\nFor input2:")
display(input2)

# We can train the autoencoder with this two different time series. Note that both time series can belong to the
# same main time serie.
inputs = [input1, input2]

autoencoder.fit(inputs, epochs=1000)

# Once trained, we can use it to encode the data of each input:

encoded_input1 = autoencoder.encode(input1)
encoded_input2 = autoencoder.encode(input2)


# The encoded version of both inputs have the exactly same shape: 1 row with "latent_space" columns.

print("\nEncoded input1")
display(encoded_input1)

print("\nEncoded input2")
display(encoded_input2)

# We can rebuild the original inputs based on these encoded inputs:
decoded_input1 = autoencoder.decode(encoded_input1,
                                    timesteps=input1.shape[1])    # We need to specify how many timesteps the decoded version should have.
decoded_input2 = autoencoder.decode(encoded_input2,
                                    timesteps=input2.shape[1])    # We need to specify how many timesteps the decoded version should have.

# Compare both:
print("\nFor input1")
display(input1)
display(decoded_input1)

print("\nFor input2")
display(input2)
display(decoded_input2)

