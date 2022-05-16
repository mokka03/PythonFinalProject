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
- Linear: In features: 3136, Out features: 4096
- ReLU
- Linear: In features: 4096, Out features: 1024
- ReLU
- Dropout
- ReLU
- Linear: In features: 1024, Out features: 256
- ReLU
- Linear: In features: 256, Out features: 10


![model](model.png)

To optimize the model we use the built in cross entropy loss function and Stochastic Gradient Descent optimizer in PyTorch.

## Code
The utils.py contains some utility functions (such as an object that can be used as a context manager to measure time or a funtion that prints the used video memory) and the definition for the class with which we can implement a VGG-like model.
Most of the heavy lifting is done in train.py, which is where the training and testing happens and also where we plot the results. For reproducability we set seeds for all of the modules that use some form of random number generation. Since networks perform better on normalized data, we use pytorch's normalize transformation on our dataset, which needs the mean and standard deviation of the dataset as parameter, so at first we only use our trainset to calculate these parameters and than we get them again with the intended transformations. Using the VGGRegressionModel (which we will use for classification instead of regression) from the utils file, we create a simple model that has 2 convolutions with max-pooling after both of them. The dataset, optimizer and error function we use are all included with the pytorch module. For optimizer we've choosen pytorch's Stochastic Gradient Descent optimizer and for error function we picked cross entropy loss, which is a standard error funciton for classification. Next we implemented everything to do with training: a way to load previous models; the training loop (along with a way to use cuda if thats desired); saving the model after every epoch; and finally the ploting of the loss at every epoch. Last but not least we evaluated the network on the testset (for which we also applied normalizations with the same mean and std values) and calculated the accuracy.

## Results
We train the network for 10 epochs, and could reach 90.22 % accuracy on the test dataset.

<img src="fashion-MNIST/loss.png" width="400"/>

## Requirements
The code was tested with PyTorch 1.11.0, Python 3.9.9, CUDA 11.3, and up-to-date versions of a few other standard packages, older versions might not work out-of-the-box. It can run on both CPU and GPU (Nvidia), but GPU operation is preferred, CPU is much slower and not tested. GPU requirements are the same as required by PyTorch.

## Contribution of group members
Benjámin Kispál: Implementation of the network, and the optimization part. Levente Maucha: load and preprocess the data and other functions. But we mainly work together, and discuss the upcoming problems.

## References
[https://pytorch.org/](https://pytorch.org/)

[https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/](https://viso.ai/deep-learning/vgg-very-deep-convolutional-networks/)
