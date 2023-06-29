# Custom-ViT-Visual-Transformer

# Vision Transformer (ViT)

This repository contains the code for training a Vision Transformer (ViT) model using the MNIST dataset. The ViT model is implemented using PyTorch and is trained to classify handwritten digits. 
MNIST contains 60,000 handwritten numbers for training and 10,000 handwritten numbers for testing.

# Example of Data

![Alt Text](Example.png)


## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- Python (3.6 or higher)
- PyTorch (1.8.1 or higher)
- NumPy (1.19.5 or higher)


pip install torch numpy
Usage
Clone the repository:

shell
Copy code
git clone https://github.com/your_username/vision-transformer.git
cd vision-transformer
Run the script:

shell
Copy code
python main.py
Code Explanation
The code is structured as follows:

The create_patches function is used to split the input images into patches for processing by the ViT model.
The MultiHeadSelfAttention class implements the multi-head self-attention mechanism used in the ViT model.
The ViTBlock class defines a single block of the ViT model, which consists of a multi-head self-attention layer and a feed-forward neural network.
The VisionTransformer class defines the complete ViT model, including multiple ViT blocks.
The get_positional_embeddings function generates positional embeddings used in the ViT model.
The data_augmentation function applies data augmentation techniques to the input images.
The train function performs the training loop, including forward and backward passes, optimization, and evaluation.
The main script loads the MNIST dataset, creates data loaders, initializes the ViT model, defines the loss function and optimizer, and trains the model.
Output
The training process produces the following output:

yaml
Copy code
Epoch 1/30, Train Loss: 2.1635, Test Loss: 2.0296, Accuracy: 0.4228
Epoch 2/30, Train Loss: 2.0531, Test Loss: 1.9152, Accuracy: 0.5475
...
Epoch 29/30, Train Loss: 1.6283, Test Loss: 1.5587, Accuracy: 0.9033
Epoch 30/30, Train Loss: 1.6282, Test Loss: 1.5587, Accuracy: 0.9035
The output shows the training progress over multiple epochs. For each epoch, it displays the training loss, test loss, and accurac
