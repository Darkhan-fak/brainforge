"""BrainNet CorticalColumn на CIFAR-10."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import time
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brainnet import CorticalColumn, TopoLoss

def get_data(data_dir="./data", subset_size=None):
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

def train_one_epoch(model, loader, criterion, topo_loss, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        # Flatten: [B, 3, 32, 32] -> [B, 3072]
        x = images.view(images.size(0), -1)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, labels)
        # Добавить TopoLoss
        if topo_loss is not None:
            loss = loss + topo_loss(model.get_activations())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / len(loader), 100.0 * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            x = images.view(images.size(0), -1)
            outputs = model(x)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return total_loss / len(loader), 100.0 * correct / total

def run_brainnet(epochs=20, batch_size=128, hidden=256,
                 inhibitory_ratio=0.2, use_lateral=True,
                 topo_weight=0.1, subset_size=None,
                 save_path="benchmarks/results",
                 save_name="brainnet_cifar10.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    train_set, test_set = get_data(subset_size=subset_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    # CorticalColumn: 3072 вход (32x32x3), 256 скрытый, 10 классов
    model = CorticalColumn(
        in_features=3072,
        hidden_features=hidden,
        out_features=10,
        inhibitory_ratio=inhibitory_ratio,
        use_lateral=use_lateral,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CorticalColumn: {n_params:,} parameters")
    print(f"  inhibitory_ratio={inhibitory_ratio}")
    print(f"  use_lateral={use_lateral}")
    print(f"  topo_weight={topo_weight}")
    
    criterion = nn.CrossEntropyLoss()
    topo = TopoLoss(weight=topo_weight) if topo_weight > 0 else None
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    results = {
        "model": "CorticalColumn",
        "params": n_params,
        "hidden": hidden,
        "inhibitory_ratio": inhibitory_ratio,
        "use_lateral": use_lateral,
        "topo_weight": topo_weight,
        "epochs": epochs,
        "train_samples": len(train_set),
        "history": []
    }
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, topo, optimizer, device
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
    
    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/{save_name}", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nГотово за {elapsed:.1f}с")
    print(f"Финальная точность: {results['final_test_acc']}%")
    print(f"Результаты сохранены в {save_path}/{save_name}")
    
    return results

if __name__ == "__main__":
    print("=== Быстрый тест BrainNet (5 эпох, 5000 примеров) ===")
    run_brainnet(epochs=5, subset_size=5000)
    
    # Раскомментировать для полного бенчмарка:
    # print("\n=== Полный бенчмарк (20 эпох, все данные) ===")
    # run_brainnet(epochs=20)
