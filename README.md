# Autoencoder Neural Networks with MNIST

This repository contains Python implementations of autoencoder neural networks for solving tasks related to the MNIST dataset. These tasks include digit reconstruction, next-digit prediction, and digit addition. The project demonstrates various configurations of autoencoders, exploring their performance on clean and noisy data and comparing them with PCA-based approaches.

## Features

- **Digit Reconstruction**: Reconstructs digits from the MNIST dataset using autoencoders.
- **Next-Digit Prediction**: Predicts the next digit in a sequence using an autoencoder with custom latent representations.
- **Digit Addition**: Uses autoencoders to learn addition operations between digits.
- **Noisy Data Handling**: Demonstrates the denoising capabilities of autoencoders.
- **PCA Comparison**: Compares the reconstruction results with PCA-based dimensionality reduction.

## Files

- `autoencoder_adder.py`: Implementation of an autoencoder for performing addition of two digits.
- `autoencoder_adder_softmax.py`: Similar to `autoencoder_adder.py` but includes a softmax layer for classification.
- `autoencoder_next_digit.py`: Implementation for predicting the next digit in a sequence.
- `autoencoder_next_digit_softmax.py`: Variation of `autoencoder_next_digit.py` using a softmax layer in the latent space.
- `autoencoder_reconstruct.py`: Core implementation for reconstructing MNIST digits with and without noise.
- `report.pdf`: Detailed report documenting the methodologies, experiments, and results of this project.
