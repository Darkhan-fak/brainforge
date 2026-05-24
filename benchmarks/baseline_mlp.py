"""Baseline: обычная MLP на CIFAR-10."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import time
import json
import os

class BaselineMLP(nn.Module):
    """Обычная MLP с таким же количеством параметров как CorticalColumn."""
    def __init__(self, in_features=3072, hidden=256, out_features=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )
    
    def forward(self, x):
        return self.net(self.flatten(x))

def get_data(data_dir="./data", subset_size=None):
    """Загрузить CIFAR-10. subset_size — для быстрых тестов."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    if subset_size:
        train_set = Subset(train_set, range(subset_size))
    return train_set, test_set

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Одна эпоха тренировки."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / len(loader), 100.0 * correct / total

def evaluate(model, loader, criterion, device):
    """Оценка на тестовом наборе."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return total_loss / len(loader), 100.0 * correct / total

def run_baseline(epochs=20, batch_size=128, hidden=256, 
                 subset_size=None, save_path="benchmarks/results"):
    """Запустить baseline MLP."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    train_set, test_set = get_data(subset_size=subset_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    model = BaselineMLP(in_features=3072, hidden=hidden, out_features=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Baseline MLP: {n_params:,} parameters")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    results = {
        "model": "BaselineMLP",
        "params": n_params,
        "hidden": hidden,
        "epochs": epochs,
        "train_samples": len(train_set),
        "history": []
    }
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        results["history"].append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "test_loss": round(test_loss, 4),
            "test_acc": round(test_acc, 2),
        })
        
        print(f"Epoch {epoch:2d}/{epochs} | "
              f"Train: {train_acc:.1f}% (loss {train_loss:.4f}) | "
              f"Test: {test_acc:.1f}% (loss {test_loss:.4f})")
    
    elapsed = time.time() - start_time
    results["total_time_seconds"] = round(elapsed, 1)
    results["final_test_acc"] = results["history"][-1]["test_acc"]
    
    # Сохранить результаты
    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/baseline_mlp.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nГотово за {elapsed:.1f}с")
    print(f"Финальная точность: {results['final_test_acc']}%")
    print(f"Результаты сохранены в {save_path}/baseline_mlp.json")
    
    return results

if __name__ == "__main__":
    # Быстрый тест: 5 эпох, 5000 примеров
    print("=== Быстрый тест (5 эпох, 5000 примеров) ===")
    run_baseline(epochs=5, subset_size=5000)
    
    # Раскомментировать для полного бенчмарка:
    # print("\n=== Полный бенчмарк (20 эпох, все данные) ===")
    # run_baseline(epochs=20)
