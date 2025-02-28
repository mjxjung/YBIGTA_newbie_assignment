import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam
import torch.nn.functional as F
import random

from transformers import PreTrainedTokenizer
from typing import Literal
from config import *

class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method

        self.softmax = nn.LogSoftmax(dim=1)

        # 🔹 Xavier Initialization 적용 (더 빠른 수렴)
        nn.init.xavier_uniform_(self.embeddings.weight)
        nn.init.xavier_uniform_(self.weight.weight)

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)

        # 🔹 학습률 감소 스케줄러 적용 (Loss 개선 없을 시 학습률 감소)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        if self.method == "skipgram":
            self._train_skipgram(corpus, tokenizer, num_epochs, criterion, optimizer, scheduler)
        else: 
            self._train_cbow(corpus, tokenizer, num_epochs, criterion, optimizer, scheduler)

    def _train_cbow(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        num_epochs: int,
        criterion,
        optimizer,
        scheduler
    ) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x, y = self.cbow_data(corpus, tokenizer)
        self.to(device)
       
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for i in range(0, len(x), batch_size):
                x_ = torch.tensor(x[i:i+batch_size], dtype=torch.long, device=device)
                y_ = torch.tensor(y[i:i+batch_size], dtype=torch.long, device=device)
                
                optimizer.zero_grad()
                pred = self.cbow(x_)  
                loss = criterion(pred, y_)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

            # 🔹 학습률 스케줄러 업데이트
            scheduler.step(avg_loss)

    def _train_skipgram(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        num_epochs: int,
        criterion,
        optimizer,
        scheduler
    ) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x, y = self.skip_data(corpus, tokenizer)
        self.to(device)

        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for i in range(0, len(x), batch_size):
                x_ = torch.tensor(x[i:i+batch_size], dtype=torch.long, device=device)
                y_ = torch.tensor(y[i:i+batch_size], dtype=torch.long, device=device)
                
                optimizer.zero_grad()
                pred = self.skipgram(x_)  
                loss = criterion(pred, y_)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

            # 🔹 학습률 스케줄러 업데이트
            scheduler.step(avg_loss)

    def skipgram(self, x):
        x = self.embeddings(x)
        x = self.weight(x)
        return self.softmax(x)  # 🔹 Softmax 활성화 적용

    def cbow(self, x):
        x = self.embeddings(x).sum(dim=1)
        x = self.weight(x)
        return self.softmax(x)  # 🔹 Softmax 활성화 적용

    @staticmethod
    def pad(sentence, tokenizer, window_size=window_size):
        pad = [tokenizer.pad_token_id] * window_size
        return pad + tokenizer(sentence).input_ids[1:-1] + pad
    
    def cbow_data(self, token_data, tokenizer):
        x, y = [], []

        for sentence in token_data:
            pad = self.pad(sentence, tokenizer)
            for i in range(self.window_size, len(pad) - self.window_size):
                context = (
                    pad[i - self.window_size : i] +
                    pad[i + 1: i + self.window_size + 1]
                )
                target = pad[i]
                x.append(context)
                y.append(target)
        
        return x, y

    def skip_data(self, corpus, tokenizer):
        x, y = [], []
        for sentence in corpus:
            pad_seq = self.pad(sentence, tokenizer, self.window_size)
            for i in range(self.window_size, len(pad_seq) - self.window_size):
                center = pad_seq[i]

                # 🔹 동적인 Window 크기 적용
                dynamic_window = random.randint(1, self.window_size)
                left_context = pad_seq[i - dynamic_window : i]
                right_context = pad_seq[i + 1 : i + dynamic_window + 1]

                for w in (left_context + right_context):
                    x.append(center)
                    y.append(w)

        return x, y
