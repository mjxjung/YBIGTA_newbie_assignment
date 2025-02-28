import torch
import random
import numpy as np
from torch import nn, optim, Tensor, LongTensor, FloatTensor
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from datasets import load_dataset

from tqdm import tqdm
from sklearn.metrics import f1_score

from word2vec import Word2Vec
from model import MyGRULanguageModel
from config import *
import os

def set_seed(seed: int = 42):
    """
    Fix the seed to ensure experiment reproducibility.
    
    Args:
        seed (int): The random seed value (default: 42)
    """
    random.seed(seed)  # Fix Python random seed
    np.random.seed(seed)  # Fix NumPy random seed
    torch.manual_seed(seed)  # Fix PyTorch random seed (CPU)
    torch.cuda.manual_seed(seed)  # Fix PyTorch random seed (GPU)


if __name__ == "__main__":
   # Set seed for reproducibility
    set_seed(42)

    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    # load Word2Vec checkpoint and get trained embeddings
    word2vec = Word2Vec(vocab_size, d_model, window_size, method)
    checkpoint = torch.load("word2vec.pt")
    word2vec.load_state_dict(checkpoint)
    embeddings = word2vec.embeddings_weight()

    # declare model, criterion and optimizer
    model = MyGRULanguageModel(d_model, hidden_size, num_classes, embeddings).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 기존 체크포인트가 있으면 로드
    if os.path.exists("checkpoint_finetuned1.pt"):
        print("Loading existing model checkpoint_finetuned1 from 'checkpoint_finetuned1.pt'...")
        checkpoint = torch.load("checkpoint_finetuned1.pt")
        model.load_state_dict(checkpoint)

    # load train, validation dataset
    dataset = load_dataset("google-research-datasets/poem_sentiment")

    g = torch.Generator()
    g.manual_seed(42)  # DataLoader에서 재현성을 유지하기 위한 시드 설정

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, generator=g)
    validation_loader = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=True, generator=g)

    # train
    for epoch in tqdm(range(num_epochs)):
        loss_sum = 0
        for data in train_loader:
            optimizer.zero_grad()
            input_ids = tokenizer(data["verse_text"], padding=True, return_tensors="pt")\
                .input_ids.to(device)
            labels = data["label"].to(device)
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        preds = []
        labels = []
        with torch.no_grad():
            for data in validation_loader:
                input_ids = tokenizer(data["verse_text"], padding=True, return_tensors="pt")\
                    .input_ids.to(device)
                logits = model(input_ids)
                labels += data["label"].tolist()
                preds += logits.argmax(-1).cpu().tolist()

        macro = f1_score(labels, preds, average='macro')
        micro = f1_score(labels, preds, average='micro')
        print(f"loss: {loss_sum/len(train_loader):.6f} | macro: {macro:.6f} | micro: {micro:.6f}")

    # save model checkpoint
    torch.save(model.cpu().state_dict(), "checkpoint.pt")
    print(f"Model checkpoint saved at 'checkpoint.pt'")
