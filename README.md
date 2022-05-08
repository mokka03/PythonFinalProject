# Classification on fashion-MNIST database using VGGnet-like architecture

## Purpose
We want to classify the pictures of fashion MNIST database at least with 85 % accuracy. Fashion-MNIST is a dataset of [Zalando](https://jobs.zalando.com/en/tech/?gh_src=22377bdd1us)'s article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

## System architecture
We want to implement a VGGnet-like architecture using PyTorch machine learning framework (see on figure below). It contains twa major parts: a convolutional network to gain the features from the images, and a fully connected network to classify the input:

Features:
- 2D convolutional layer: Input channels: 1, Output channels 32, Kernel size: 3x3, Stride: 1x1, Padding: 1x1
- 2D Batch normalization
- ReLu
- 2D Max pooling: Kernel size: 2x2, Stride: 2, Padding: 0
- 2D convolutional layer: Input channels: 32, Output channels 64, Kernel size: 3x3, Stride: 1x1, Padding: 1x1
- 2D Batch normalization
- ReLu
- 2D Max pooling: Kernel size: 2x2, Stride: 2, Padding: 0

Classifier:


![model](model.png)

## Results

## Contribution of group members
Benjámin Kispál: Implementation of the network, and the optimization part. Levente Maucha: load and preprocess the data. But we mainly work together, and discuss the upcoming problems.

## References
[https://pytorch.org/](https://pytorch.org/)