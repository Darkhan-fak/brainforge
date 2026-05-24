"""Скачать CIFAR-10 для бенчмарка."""
import torchvision
import torchvision.transforms as transforms
import os

def download_cifar10(data_dir="./data"):
    """Скачивает CIFAR-10 если ещё не скачан."""
    os.makedirs(data_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
    ])
    
    print("Скачиваю CIFAR-10 train set...")
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    print(f"Train: {len(train_set)} изображений")
    
    print("Скачиваю CIFAR-10 test set...")
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    print(f"Test: {len(test_set)} изображений")
    
    # Проверка: загрузить один батч
    loader = torch.utils.data.DataLoader(train_set, batch_size=4)
    images, labels = next(iter(loader))
    print(f"Batch shape: {images.shape}")  # [4, 3, 32, 32]
    print(f"Labels: {labels}")
    print("Данные готовы!")
    
    return train_set, test_set

if __name__ == "__main__":
    import torch
    download_cifar10()
