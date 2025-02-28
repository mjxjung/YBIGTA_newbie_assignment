import torch
from torch import nn, Tensor


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        # 구현하세요!
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = True

        # 입력에서 3개의 게이트로 나눠 처리
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias = self.bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias= self.bias)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        # 구현하세요!
        """
        x: (batch_size, input_size)
        h: (batch_size, hidden_size)
        """
        # 선형 변환을 거쳐 게이트 값 계산
        gate_x = self.x2h(x)
        gate_h = self.h2h(h)

        # 3개의 게이트로 나누기 (Reset, Input, New)
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        # 게이트 연산 수행 (Resnet, Update/ candiate 상태)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

        # 최종 hidden state 업데이트
        hy = (1 - inputgate) * newgate + inputgate * h  # 공식 그대로 적용
        return hy


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        """
        input_size: 입력 벡터 크기
        hidden_size: 은닉 상태 크기
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size) # GRUCell 사용
        # 구현하세요!

    def forward(self, inputs: Tensor) -> Tensor:
        # 구현하세요!
        batch_size, seq_len, _ = inputs.shape  # 입력 크기 체크
        h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            h = self.cell(x_t, h)  
        
        return h 

