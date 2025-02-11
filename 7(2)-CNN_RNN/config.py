from typing import Literal
import torch

# device = "cpu"
if torch.backends.mps.is_available():
    device = torch.device("mps")  # macOS MPS 지원
    print("Using MPS (Apple Metal Performance Shaders) for training")
else:
    print ("MPS device not found.")

d_model = 2048

# Word2Vec
window_size = 7
method: Literal["cbow", "skipgram"] = "skipgram"
lr_word2vec = 1e-03
num_epochs_word2vec = 100

# GRU
hidden_size = 2048
num_classes = 4
lr = 5e-03
num_epochs = 100
batch_size = 2048

