# VAE-vs-DCGAN


# Assignment 4 - Deep Learning: VAE vs DCGAN on CIFAR-10

## 🧠 Author
**Raasikh Naveed**

## 📘 Course
Deep Learning (898BD)

## 📌 Project Overview
This project compares two popular generative deep learning models:

- **Variational Autoencoder (VAE)**
- **Deep Convolutional Generative Adversarial Network (DCGAN)**

on the **CIFAR-10 dataset** to evaluate their performance in generating realistic images.

## 🎯 Objective
To explore the generative capabilities of VAE and DCGAN models through both **qualitative** and **quantitative** metrics including:

- **SSIM (Structural Similarity Index)**
- **MSE (Mean Squared Error)**
- **Time per Epoch**
- **Visual Analysis of Generated Images**

## 🏗️ Methodology
- Preprocessed CIFAR-10 dataset (normalized between -1 and 1).
- Used **Adam optimizer** for both models with tuning on learning rates and batch sizes.
- Tracked training loss, epoch times, and visual outputs.
- Evaluated performance using SSIM and MSE.

## 🧬 Model Architecture

### 🔸 Variational Autoencoder (VAE)
- Encoder: Convolutional layers for downsampling
- Latent Space: Mean and log variance vectors (mu, logvar)
- Decoder: Transposed convolution layers to reconstruct images
- Loss Function: Binary Cross-Entropy + KL Divergence

### 🔸 Deep Convolutional GAN (DCGAN)
- Generator: Transposed convolutions from random noise
- Discriminator: CNN to classify real vs fake
- Loss Function: Binary Cross-Entropy for both generator and discriminator

## 📊 Results

### VAE:
- Training Loss: Decreased from 240 to ~180
- Time per Epoch: 15–19s
- MSE: **0.1543**
- SSIM: **0.1615**
- Output: Structurally consistent, but less visually sharp

### DCGAN:
- Generator Loss: Increased, indicating difficulty against strong discriminator
- Discriminator Loss: Low and stable
- Time per Epoch: 20–21.6s
- MSE: **0.3571**
- SSIM: **0.0022**
- Output: Visually sharp images with reduced structural fidelity

## 💡 Conclusions

- **VAE** excels in structured image generation with smooth latent representations.
- **DCGAN** produces sharper and more realistic images, but with higher pixel deviation.
- Trade-offs exist: VAE for structural consistency, DCGAN for visual quality.

## ❓ Research Question

> How do VAE and DCGAN compare in terms of generative performance on CIFAR-10 across structural and perceptual evaluation metrics?

## 🧪 Technologies Used
- PyTorch
- NumPy
- Matplotlib
- torchvision
