---
icon: lucide/network
---

# Neural Networks

## Biological Inspiration

Artificial neural networks are inspired by biological neurons.

## Network Structure

A neural network (NN) has 3 types of layers:

- **Input layer**
- **Hidden layer**
- **Output layer**

Deep neural networks (DNN) usually have more hidden layers, but still use the same 3 types of layers.

## Forward Pass Example

For an input image of size 28×28 (784 pixels):

$$
Z_1 = w_{11} x_1 + w_{12} x_2 + w_{13} x_3 + b_1
$$

$$
a_1 = f(Z_1) = \text{ReLU}(Z_1)
$$

Weights are typically initialized randomly.

## Three Steps to Training a Neural Network

1. **Forward propagation**: Push the example through the network to get a predicted output
2. **Compute the cost**: Calculate the difference between predicted output and actual data
3. **Backward propagation**: Push back the derivative of the error and apply it to each weight, so that next time it results in a lower error

## Training Pipeline

The training pipeline consists of choosing:

- The prediction model
- The loss function
- The optimizer

Once these choices are made, we can feed the input data and labels to start the training process.

!!! info "FashionMNIST Task"

    Predict if an input image belongs to one of the following classes:  
    T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, or Ankle boot.

    Fashion-MNIST is a dataset comprising 28×28 grayscale images of 70,000 fashion products from 10 categories, with 7,000 images per category. The training set has 60,000 images and the test set has 10,000 images.
