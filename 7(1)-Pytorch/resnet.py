import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os
from typing import Type, List, Optional

class BasicBlock(nn.Module):
    expansion: int = 1 # 출력한 채널 수의 확장 비율

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super(BasicBlock, self).__init__()

        self.conv1: nn.Conv2d = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(planes)
        # 첫 번째 리뷰(입력채널, 출력채널, )
        self.conv2: nn.Conv2d = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(planes)
        # 두 번째 리뷰(입력채널=출력채널=planes, stride=1로 유지지 )
        self.shortcut: nn.Sequential = nn.Sequential() # 비어있으므로 아무 연산도 수행하지 않는다.
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        ## TODO
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        
        return out
        
    
class ResNet(nn.Module):
    def __init__(self, block: Type[BasicBlock], num_blocks: List[int], num_classes: int = 2, init_weights: bool = True) -> None:
        super().__init__()

        self.in_channels: int = 64

        ## TODO
        # Resnet layer를 구현하세요!
        # Hint: 두번째 layer부터는 _make_layer 메서드를 활용하세요! 
        
        self.avg_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.fc: nn.Linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: Type[BasicBlock], out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:

        strides: List[int] = [stride] + [1] * (num_blocks - 1)
        layers: List[nn.Module] = []
        
        ## TODO
        
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ## TODO
        output = None
        return output