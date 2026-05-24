# BrainForge — Пошаговая инструкция для AI-кодера

## ЧТО ЭТО
Библиотека PyTorch-модулей, вдохновлённых архитектурой мозга.
Цель: создать нейросеть, которая учится эффективнее стандартных
благодаря компонентам из реальной нейронауки.

## ТЕКУЩИЙ СТАТУС
Этап: ПУБЛИКАЦИЯ (Шаг 10 из 10)

---

## ШАГ 1: НАСТРОЙКА СРЕДЫ
### Что сделать:
```bash
# 1. Убедиться что Python >= 3.9
python3 --version

# 2. Создать виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate

# 3. Установить PyTorch (CPU версия для начала)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Установить дополнительные зависимости
pip install numpy matplotlib tqdm

# 5. Установить проект в режиме разработки
pip install -e .

# 6. Проверить что всё работает
python -c "import torch; print(f'PyTorch {torch.__version__} OK')"
python -c "from brainnet import CorticalColumn; print('BrainNet import OK')"
```

### Ожидаемый результат:
Обе команды python -c выводят OK без ошибок.

### Если есть GPU (CUDA):
```bash
# Вместо CPU-версии установить CUDA-версию
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### После завершения:
Обнови статус вверху этого файла на:
`Этап: ТЕСТИРОВАНИЕ МОДУЛЕЙ (Шаг 2 из 10)`

---

## ШАГ 2: ТЕСТИРОВАНИЕ СУЩЕСТВУЮЩИХ МОДУЛЕЙ
### Что сделать:
```bash
python tests/test_brainnet.py
```

### Ожидаемый результат:
```
[PASS] CorticalColumn forward shape
[PASS] CorticalColumn activations stored
[PASS] CorticalColumn gradient flow
... (все 14 тестов)
All tests passed!
```

### Если тесты падают:
Записать ошибку в секцию ERROR LOG внизу этого файла.
Исправить. Перезапустить тесты.

### После завершения:
Обнови статус на: `Этап: СКАЧИВАНИЕ ДАННЫХ (Шаг 3 из 10)`

---

## ШАГ 3: СКАЧИВАНИЕ ДАННЫХ ДЛЯ БЕНЧМАРКА
### Что сделать:
Создать файл `benchmarks/download_data.py`:

```python
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
```

### Запустить:
```bash
python benchmarks/download_data.py
```

### Ожидаемый результат:
```
Скачиваю CIFAR-10 train set...
Train: 50000 изображений
Скачиваю CIFAR-10 test set...
Test: 10000 изображений
Batch shape: torch.Size([4, 3, 32, 32])
Данные готовы!
```

### После завершения:
Обнови статус на: `Этап: BASELINE МОДЕЛЬ (Шаг 4 из 10)`

---

## ШАГ 4: BASELINE — ОБЫЧНАЯ НЕЙРОСЕТЬ
### Что сделать:
Создать файл `benchmarks/baseline_mlp.py`.
Это обычная нейросеть (MLP) без brain-inspired компонентов.
Нужна для сравнения — чтобы потом показать что BrainNet лучше.

```python
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
```

### Запустить:
```bash
python benchmarks/baseline_mlp.py
```

### Ожидаемый результат:
```
Device: cuda (или cpu)
Baseline MLP: ~850,000 parameters
Epoch  1/5 | Train: 25-35% | Test: 25-35%
...
Epoch  5/5 | Train: 40-50% | Test: 38-45%
Результаты сохранены в benchmarks/results/baseline_mlp.json
```

### Важно:
Запиши финальную точность. Это число, которое BrainNet должен побить.

### После завершения:
Обнови статус на: `Этап: BRAINNET БЕНЧМАРК (Шаг 5 из 10)`

---

## ШАГ 5: BRAINNET НА ТОМ ЖЕ БЕНЧМАРКЕ
### Что сделать:
Создать файл `benchmarks/brainnet_cifar10.py`.
Тот же бенчмарк, но с CorticalColumn вместо MLP.

```python
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
                 save_path="benchmarks/results"):
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
    with open(f"{save_path}/brainnet_cifar10.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nГотово за {elapsed:.1f}с")
    print(f"Финальная точность: {results['final_test_acc']}%")
    print(f"Результаты сохранены в {save_path}/brainnet_cifar10.json")
    
    return results

if __name__ == "__main__":
    print("=== Быстрый тест BrainNet (5 эпох, 5000 примеров) ===")
    run_brainnet(epochs=5, subset_size=5000)
    
    # Раскомментировать для полного бенчмарка:
    # print("\n=== Полный бенчмарк (20 эпох, все данные) ===")
    # run_brainnet(epochs=20)
```

### Запустить:
```bash
python benchmarks/brainnet_cifar10.py
```

### Ожидаемый результат:
Точность CorticalColumn при том же количестве параметров.
Сравни с baseline из Шага 4.

### После завершения:
Обнови статус на: `Этап: СРАВНЕНИЕ РЕЗУЛЬТАТОВ (Шаг 6 из 10)`

---

## ШАГ 6: СРАВНЕНИЕ РЕЗУЛЬТАТОВ
### Что сделать:
Создать файл `benchmarks/compare.py`.
Читает JSON-результаты обоих бенчмарков и строит графики.

```python
"""Сравнение Baseline MLP vs BrainNet CorticalColumn."""
import json
import os

def compare(results_dir="benchmarks/results"):
    # Загрузить результаты
    with open(f"{results_dir}/baseline_mlp.json") as f:
        baseline = json.load(f)
    with open(f"{results_dir}/brainnet_cifar10.json") as f:
        brainnet = json.load(f)
    
    print("=" * 60)
    print("СРАВНЕНИЕ: Baseline MLP vs BrainNet CorticalColumn")
    print("=" * 60)
    print(f"{'Метрика':<30} {'Baseline':>12} {'BrainNet':>12}")
    print("-" * 60)
    print(f"{'Параметры':<30} {baseline['params']:>12,} {brainnet['params']:>12,}")
    print(f"{'Финальная точность (%)':<30} {baseline['final_test_acc']:>12.1f} {brainnet['final_test_acc']:>12.1f}")
    print(f"{'Время (сек)':<30} {baseline['total_time_seconds']:>12.1f} {brainnet['total_time_seconds']:>12.1f}")
    
    diff = brainnet['final_test_acc'] - baseline['final_test_acc']
    print(f"\n{'Разница':<30} {diff:>+12.1f}%")
    
    if diff > 0:
        print(">>> BrainNet ЛУЧШЕ baseline")
    elif diff < 0:
        print(">>> BrainNet ХУЖЕ baseline — нужна отладка")
    else:
        print(">>> Одинаково")
    
    # Сравнение по эпохам (sample efficiency)
    # На какой эпохе каждая модель достигла определённой точности?
    target_acc = 35.0  # порог для сравнения
    baseline_epoch = None
    brainnet_epoch = None
    
    for h in baseline["history"]:
        if h["test_acc"] >= target_acc and baseline_epoch is None:
            baseline_epoch = h["epoch"]
    for h in brainnet["history"]:
        if h["test_acc"] >= target_acc and brainnet_epoch is None:
            brainnet_epoch = h["epoch"]
    
    print(f"\nЭпоха достижения {target_acc}% точности:")
    print(f"  Baseline: эпоха {baseline_epoch or 'не достигнуто'}")
    print(f"  BrainNet: эпоха {brainnet_epoch or 'не достигнуто'}")
    
    if baseline_epoch and brainnet_epoch and brainnet_epoch < baseline_epoch:
        efficiency = (1 - brainnet_epoch / baseline_epoch) * 100
        print(f"  >>> BrainNet достиг цели на {efficiency:.0f}% быстрее")

if __name__ == "__main__":
    compare()
```

### Запустить:
```bash
python benchmarks/compare.py
```

### После завершения:
Обнови статус на: `Этап: ABLATION STUDY (Шаг 7 из 10)`

---

## ШАГ 7: ABLATION STUDY — КАКОЙ КОМПОНЕНТ ПОМОГАЕТ
### Что сделать:
Создать файл `benchmarks/ablation.py`.
Запустить BrainNet с каждым компонентом вкл/выкл.
Это покажет вклад каждого brain-inspired компонента.

Запустить 4 конфигурации:
1. CorticalColumn без ничего (no inhibitory, no lateral, no topo)
2. + InhibitoryLayer only
3. + LateralConnections only
4. + TopoLoss only
5. Всё вместе (полный BrainNet)

Каждый запуск — вызов run_brainnet() с разными параметрами.
Сохранить результаты в отдельные JSON.

### После завершения:
Обнови статус на: `Этап: ДОКУМЕНТАЦИЯ (Шаг 8 из 10)`

---

## ШАГ 8: ДОКУМЕНТАЦИЯ И ВИЗУАЛИЗАЦИЯ
### Что сделать:
1. Обновить README.md с реальными результатами бенчмарков
2. Добавить графики (matplotlib): accuracy curves для каждой конфигурации
3. Добавить таблицу ablation study в README
4. Описать каждый модуль с примерами кода

### После завершения:
Обнови статус на: `Этап: GIT INIT (Шаг 9 из 10)`

---

## ШАГ 9: ИНИЦИАЛИЗАЦИЯ GIT И ПЕРВЫЙ КОММИТ
### Что сделать:
```bash
# Создать .gitignore
echo "__pycache__/
*.pyc
.venv/
data/
*.egg-info/
dist/
build/
benchmarks/results/
.DS_Store" > .gitignore

# Инициализировать репозиторий
git init
git add .
git commit -m "feat: BrainNet v0.1.0 — cortex-inspired PyTorch modules

- CorticalColumn: 6-layer cortical column as nn.Module
- InhibitoryLayer: 20% inhibitory neurons with lateral inhibition
- LateralConnections: intra-layer connections (80% biological ratio)
- TopoLoss: topographic organization loss
- Benchmarks: CIFAR-10 comparison vs baseline MLP
- 14 passing tests"
```

### После завершения:
Обнови статус на: `Этап: ПУБЛИКАЦИЯ (Шаг 10 из 10)`

---

## ШАГ 10: ПУБЛИКАЦИЯ НА GITHUB
### Что сделать:
1. Создать репозиторий на GitHub: `brainforge`
2. Добавить remote и push
3. Убедиться что README отображается корректно
4. Добавить topics: `pytorch`, `neuroscience`, `brain-inspired`, `deep-learning`

```bash
git remote add origin https://github.com/<username>/brainforge.git
git push -u origin main
```

### После завершения:
Обнови статус на: `Этап: PHASE 1 COMPLETE. Переход к Phase 2: NeuroSleep`

---

## СТРУКТУРА ПРОЕКТА
```
brainforge/
├── CLAUDE.md              ← этот файл (инструкция)
├── README.md              ← документация проекта
├── pyproject.toml         ← настройки пакета
├── .gitignore
├── brainnet/              ← Phase 1: Архитектура
│   ├── __init__.py        ← публичный API
│   ├── cortical_column.py ← 6-слойная кортикальная колонка
│   ├── inhibitory.py      ← тормозные нейроны
│   ├── lateral.py         ← латеральные связи
│   └── topoloss.py        ← топографическая loss-функция
├── neurosleep/            ← Phase 2 (после Phase 1)
├── rlhbf/                 ← Phase 3 (после Phase 2)
├── tests/
│   └── test_brainnet.py   ← тесты
├── benchmarks/
│   ├── download_data.py   ← скачивание CIFAR-10
│   ├── baseline_mlp.py    ← baseline для сравнения
│   ├── brainnet_cifar10.py← BrainNet бенчмарк
│   ├── compare.py         ← сравнение результатов
│   ├── ablation.py        ← ablation study
│   └── results/           ← JSON с результатами
└── data/                  ← скачанные данные (в .gitignore)
```

## НЕЙРОНАУЧНЫЙ КОНТЕКСТ (для понимания кода)

### Почему 6 слоёв?
Реальная кора мозга имеет 6 слоёв (L1-L6), каждый с разными типами клеток:
- L1: очень мало нейронов, в основном дендриты (skip connections)
- L2/3: маленькие пирамидальные клетки (обработка внутри коры)
- L4: звёздчатые клетки (ВХОД — получает данные от органов чувств)
- L5: крупные пирамидальные клетки (ВЫХОД — отправляет команды)
- L6: полиморфные клетки (обратная связь)

### Почему 20% тормозных?
В мозге ~20% нейронов — тормозные (GABAergic). Они подавляют соседей,
создавая более чёткие представления. В обычных нейросетях их нет.

### Почему латеральные связи?
80% связей в мозге — ВНУТРИ одного слоя, не между слоями.
Обычные нейросети имеют только прямые связи (feed-forward).

### Почему TopoLoss?
В мозге похожие функции обрабатываются соседними нейронами
(топографические карты). TopoLoss штрафует если это нарушено.

---

## ERROR LOG
(Записывай ошибки и их решения ниже)

