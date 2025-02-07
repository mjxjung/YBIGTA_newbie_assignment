import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Any
from resnet import ResNet, BasicBlock
from config import *

NUM_CLASSES = 10  

if torch.backends.mps.is_available():
    # mps_device = torch.device("mps")
    # x = torch.ones(1, device=mps_device)
    # print (x)
    device = torch.device("mps")  # macOS MPS 지원
    print("Using MPS (Apple Metal Performance Shaders) for training")
else:
    print ("MPS device not found.")


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")  
# if device.type == "cuda":
#     print(f"GPU Name: {torch.cuda.get_device_name(0)}")

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 데이터 증강
    transforms.RandomCrop(32, padding=4),  # 데이터 증강
    transforms.ToTensor(),  # 반드시 적용해야 함 (PIL.Image → Tensor)
    transforms.Normalize((0.5,), (0.5,))  # 정규화
])

test_transform = transforms.Compose([
    transforms.ToTensor(),  # 반드시 적용해야 함
    transforms.Normalize((0.5,), (0.5,))
])

# CIFAR-10 데이터셋 로드
train_dataset = datasets.CIFAR10(root="./data", train=True, transform=train_transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.CIFAR10(root="./data", train=False, transform=test_transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# resnet 18 선언하기
## TODO
# basic block: ResNet-18에서 사용하는 블록임
# 블록 개수 [2, 2, 2, 2]
# cifar class 분류인만큼 10개 클래스 분류
model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=NUM_CLASSES).to(device)

criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
optimizer: optim.Adam = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

# 학습 
def train(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> None:
    model.train()
    total_loss: float = 0
    correct: int = 0
    total: int = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy: float = 100. * correct / total
    print(f"Train Loss: {total_loss / len(loader):.4f}, Accuracy: {accuracy:.2f}%")

# 평가 
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> None:
    model.eval()
    total_loss: float = 0
    correct: int = 0
    total: int = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy: float = 100. * correct / total
    print(f"Test Loss: {total_loss / len(loader):.4f}, Accuracy: {accuracy:.2f}%")

# 학습 및 평가 루프
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    train(model, train_loader, criterion, optimizer, device)
    evaluate(model, test_loader, criterion, device)

# 모델 저장
torch.save(model.state_dict(), "resnet18_checkpoint.pth")
print(f"Model saved to resnet18_checkpoint.pth")
