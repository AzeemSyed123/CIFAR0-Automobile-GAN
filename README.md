# CIFAR-10 Automobile GAN

A Generative Adversarial Network (GAN) implementation using TensorFlow/Keras to generate realistic images of automobiles from the CIFAR-10 dataset.

## Overview

This project implements a Wasserstein GAN with Gradient Penalty (WGAN-GP) to generate 32x32 pixel images of automobiles.  The model learns to create realistic car images by training a generator network to fool a discriminator network.

## Architecture

### Generator
- **Input**: 100-dimensional random noise vector
- **Architecture**: 
  - Dense layer → Reshape to 4×4×512
  - Multiple Conv2DTranspose layers with BatchNormalization and LeakyReLU
  - Progressively upsamples to 32×32×3
  - Final Conv2D with tanh activation

### Discriminator
- **Input**: 32×32×3 images
- **Architecture**:
  - Multiple Conv2D layers with strides for downsampling
  - BatchNormalization and LeakyReLU activations
  - Dropout for regularization
  - Dense output layer

### Training Method
- **Loss Function**:  Wasserstein loss with gradient penalty
- **Discriminator Updates**: 5 updates per generator update
- **Optimizer**: Adam (learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
- **Training Duration**: 1000 epochs
- **Batch Size**: 64

## Features

The notebook includes demonstrations of: 

1. **Fixed Noise Consistency**: Shows that same noise generates same image
2. **Noise Interpolation**: Smooth transitions between generated images
3. **Single Dimension Variation**: Effect of changing individual latent dimensions
4. **2D Latent Space Grid**: Exploration of latent space structure

## Requirements

```python
tensorflow>=2.x
numpy
matplotlib
