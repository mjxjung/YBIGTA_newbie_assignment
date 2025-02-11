import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal

# 구현하세요!


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
        # 구현하세요!
        self.softmax = nn.LogSoftmax(dim = 1)

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
        # 구현하세요!
        if self.method == "skipgram":
            self._train_skipgram(corpus, tokenizer, num_epochs, criterion, optimizer)
        else:
            self._train_cbow(corpus, tokenizer, num_epochs, criterion, optimizer)


    def _train_cbow(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        num_epochs: int,
        criterion: nn.CrossEntropyLoss,
        optimizer: Adam
    ) -> None:
        # 구현하세요!
        device = torch.device("mps")  # macOS MPS 지원
        x, y = self.cbow_data(corpus, tokenizer)
        self.to(device)

        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for i in range(0, len(x), batch_size):
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                x_tensor = torch.tensor(x_batch, dtype=torch.long, device=device)
                y_tensor = torch.tensor(y_batch, dtype=torch.long, device=device)

                optimizer.zero_grad()
                pred = self.cbow(x_tensor)
                loss = criterion(pred, y_tensor)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    def _train_skipgram(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        num_epochs: int,
        criterion: nn.CrossEntropyLoss,
        optimizer: Adam
    ) -> None:
        # 구현하세요!
        pass

    # 구현하세요!
    pass