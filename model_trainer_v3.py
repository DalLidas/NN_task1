#!/usr/bin/env python3
"""
model_trainer_v3.py

Обучает модель FractalNet на датасете Stanford Dogs.

Поддерживает:
 - выбор оптимизатора (--optimizer adam | adasmoothdelta)
 - параметры обучения (--epochs, --batch-size, --lr, --input)
 - отдельные датасеты: train/evaluate/test (./input/train/images, ./input/evaluate/images, ./input/test/images)
 - произвольные каналы (--channels 32 64 96 128)
 - размер классификатора (--classifier-dim 256)
 - чекпоинты (--checkpoint-every N)
 - возобновление обучения (--resume путь)

Пример:
 - python model_trainer_v3.py --optimizer adam --epochs 100 --batch-size 128 --C 4 --checkpoint-every 10
"""

import argparse
import json
from pathlib import Path
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tqdm import tqdm


# ======================================================
# Слой объединения (mean)
# ======================================================
class JoinMean(nn.Module):
    def forward(self, inputs):
        return torch.stack(inputs, dim=0).mean(dim=0)


# ======================================================
# Drop-path + объединение
# ======================================================
class DropPathJoin(nn.Module):
    """
    Объединение + локальный drop-path: при обучении случайно удаляем пути
    с вероятностью drop_prob, гарантируя хотя бы один путь.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0.0:
            return torch.stack(inputs, dim=0).mean(dim=0)

        keep = []
        for t in inputs:
            if torch.rand(1).item() > self.drop_prob:
                keep.append(t)

        if len(keep) == 0:
            idx = torch.randint(0, len(inputs), (1,)).item()
            keep.append(inputs[idx])

        return torch.stack(keep, dim=0).mean(dim=0)


# ======================================================
# Фрактальный блок (рекурсивный)
# ======================================================
class FractalBlock(nn.Module):
    """
    Реализация по статье:
    f1(x) = Conv(x)
    fC(x) = join( f(C-1)(f(C-1)(x)), Conv(x) )
    """

    def __init__(self, channels: int, C: int = 4, kernel_size: int = 3, drop_path_prob: float = 0.15):
        super().__init__()
        self.C = int(C)
        self.join = DropPathJoin(drop_prob=drop_path_prob)

        if self.C == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.sub = FractalBlock(
                channels,
                C=self.C - 1,
                kernel_size=kernel_size,
                drop_path_prob=drop_path_prob
            )
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.C == 1:
            return self.conv(x)
        a = self.sub(self.sub(x))
        b = self.conv(x)
        return self.join([a, b])


# ======================================================
# FractalNet модель
# ======================================================
class FractalNet(nn.Module):
    def __init__(
            self,
            num_classes: int = 120,
            channels: List[int] = (64, 128, 256, 512),
            classifier_dim: int = 256,
            C: int = 4,
            drop_path_prob: float = 0.15
    ):
        super().__init__()

        blocks = []
        in_ch = 3
        for out_ch in channels:
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                FractalBlock(out_ch, C=C, drop_path_prob=drop_path_prob),
                nn.MaxPool2d(2)
            )
            blocks.append(block)
            in_ch = out_ch

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, classifier_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(classifier_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ======================================================
# AdaSmoothDelta оптимизатор
# ======================================================
class AdaSmoothDelta(optim.Optimizer):
    """
    Реализация AdaSmoothDelta:
    - сглаживание градиентов (EMA)
    - сглаживание дельты обновления
    - адаптивный шаг как в Adadelta
    """

    def __init__(self, params, lr=1.0, rho=0.9, smooth=0.1, eps=1e-6):
        defaults = dict(lr=lr, rho=rho, smooth=smooth, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            rho = group['rho']
            smooth = group['smooth']
            eps = group['eps']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]

                if len(state) == 0:
                    state['grad_avg'] = torch.zeros_like(p)
                    state['delta_avg'] = torch.zeros_like(p)
                    state['acc_grad'] = torch.zeros_like(p)
                    state['acc_delta'] = torch.zeros_like(p)

                grad_avg = state['grad_avg']
                delta_avg = state['delta_avg']
                acc_grad = state['acc_grad']
                acc_delta = state['acc_delta']

                # Сглаживание градиентов (EMA)
                grad_avg.mul_(1 - smooth).add_(grad * smooth)

                # Exponential moving average как в Adadelta
                acc_grad.mul_(rho).addcmul_(grad_avg, grad_avg, value=1 - rho)

                # Шаг аналогичен Adadelta
                update = (acc_delta + eps).sqrt() / (acc_grad + eps).sqrt() * grad_avg

                # Сглаживание обновления
                delta_avg.mul_(1 - smooth).add_(update * smooth)

                # Применение шага
                p.add_(delta_avg, alpha=-lr)

                # Обновление статистики
                acc_delta.mul_(rho).addcmul_(delta_avg, delta_avg, value=1 - rho)

        return loss


# ======================================================
# ФУНКЦИИ ОБУЧЕНИЯ / ОЦЕНКИ / МЕТРИК
# ======================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Стандартный цикл обучения одной эпохи.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Обучение", leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_one_epoch_optimized(model, dataloader, criterion, optimizer, device, accumulation_steps=1, use_amp=True, clip_grad_norm=1.0):
    """
    Оптимизированный цикл обучения с:
    - AMP (ускорение в 2-3 раза)
    - Накоплением градиентов (Gradient accumulation)
    - Отсечением градиентов (Gradient clipping)
    - Неблокирующими передачами (Non-blocking transfers)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Инициализация AMP
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    pbar = tqdm(dataloader, desc="Обучение", leave=False)
    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(pbar):
        # Неблокирующая передача для параллельной загрузки
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Прямой проход с AMP
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Нормализация потерь для накопления градиентов
            loss = loss / accumulation_steps

        # Обратный проход с масштабированием градиентов
        scaler.scale(loss).backward()

        # Шаг накопления градиентов
        if (batch_idx + 1) % accumulation_steps == 0:
            # Убираем масштабирование градиентов перед отсечением
            scaler.unscale_(optimizer)

            # Отсечение градиентов для стабильности
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    clip_grad_norm
                )

            # Оптимизация и обновление scaler
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Статистика (умножаем на accumulation_steps для правильного учёта)
        running_loss += loss.item() * images.size(0) * accumulation_steps
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        # Обновление прогресс-бара
        current_loss = running_loss / total if total > 0 else 0
        current_acc = 100. * correct / total if total > 0 else 0
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.2f}%'
        })

    epoch_loss = running_loss / total if total > 0 else 0
    epoch_acc = correct / total if total > 0 else 0

    return epoch_loss, epoch_acc


def evaluate_simple(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Оценка", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            cur_loss = running_loss / total if total > 0 else 0.0
            cur_acc = correct / total if total > 0 else 0.0
            pbar.set_postfix(loss=f"{cur_loss:.4f}", acc=f"{cur_acc:.4f}")

    avg_loss = running_loss / total if total > 0 else 0.0
    overall_acc = correct / total if total > 0 else 0.0
    return avg_loss, overall_acc


def evaluate_with_per_class(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device, num_classes: int) -> Tuple[float, float, List[float]]:
    """
    Полная оценка: avg_loss, overall_acc, per_class_acc (классическая: correct_i / total_i)
    Используется только в конце (для final и best моделей).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    correct_per_class = [0 for _ in range(num_classes)]
    total_per_class = [0 for _ in range(num_classes)]

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="ПолнаяОценка", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            # учёт по классам
            for t, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                total_per_class[t] += 1
                if p == t:
                    correct_per_class[t] += 1

            cur_loss = running_loss / total if total > 0 else 0.0
            cur_acc = correct / total if total > 0 else 0.0
            pbar.set_postfix(loss=f"{cur_loss:.4f}", acc=f"{cur_acc:.4f}")

    avg_loss = running_loss / total if total > 0 else 0.0
    overall_acc = correct / total if total > 0 else 0.0

    per_class_acc = []
    for i in range(num_classes):
        if total_per_class[i] == 0:
            per_class_acc.append(0.0)
        else:
            per_class_acc.append(correct_per_class[i] / total_per_class[i])

    return avg_loss, overall_acc, per_class_acc


# ======================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ: именование файлов, загрузка чекпоинтов, обновление метаданных
# ======================================================
def next_model_index(out_dir: Path, model_name: str, optimizer_name: str) -> int:
    """
    Ищет файлы:
        trained_model_<model_name>_<optimizer>_vX.pth
    Возвращает следующий X.
    """
    pattern = re.compile(rf"trained_model_{re.escape(model_name)}_{re.escape(optimizer_name)}_v(\d+)\.pth$")
    max_idx = -1

    if out_dir.exists():
        for f in out_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                try:
                    idx = int(m.group(1))
                    max_idx = max(max_idx, idx)
                except:
                    pass

    return max_idx + 1


def load_checkpoint(path: Path, model: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> Dict[str, Any]:
    """
    Загружает чекпоинт. Если в файле только state_dict -> загружает только модель.
    Возвращает загруженный словарь.
    """
    data = torch.load(path, map_location=device)
    if isinstance(data, dict) and "model_state" in data:
        model.load_state_dict(data["model_state"])
    elif isinstance(data, dict) and "model_id" in data and "model_state" not in data:
        # может быть просто state_dict под ключами
        model.load_state_dict(data)
    elif isinstance(data, dict):
        # пытаемся загрузить как прямой state_dict (некоторые сохранённые файлы - чистые state_dict)
        try:
            model.load_state_dict(data)
        except Exception:
            # fallback: если ключ 'state_dict' существует
            if "state_dict" in data:
                model.load_state_dict(data["state_dict"])
    else:
        # если сырой state_dict
        model.load_state_dict(data)

    # загружаем оптимизатор, если присутствует
    if isinstance(data, dict) and "optimizer_state" in data:
        try:
            optimizer.load_state_dict(data["optimizer_state"])
        except Exception:
            pass

    return data


def load_or_create_metadata(meta_path: Path, model_id: str, base_model: Optional[str], args: argparse.Namespace, classes: List[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Если metadata.json уже есть и содержит запись с model_id -> возвращаем ссылку на запись и весь контейнер.
    Иначе добавляем новую запись (добавление) и возвращаем ссылку и контейнер.
    """
    entry_template = {
        "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S'),
        "model_id": model_id,
        "base_model": base_model,
        "optimizer": args.optimizer,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "C": args.C,
        "channels": args.channels,
        "classifier_dim": args.classifier_dim,
        "drop_path": args.drop_path,
        "classes": classes,
        "num_classes": len(classes),
        "start_time": None,
        "end_time": None,
        "last_checkpoint_epoch": None,
        "best": None,
        "history": {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        },
        # итоговые данные по классам и сводки test/best будут записаны позже:
        "final": None,
        "best_summary": None
    }

    if meta_path.exists():
        data = json.loads(meta_path.read_text())
        if isinstance(data, list):
            # находим существующую запись
            for entry in data:
                if entry.get("model_id") == model_id:
                    return entry, data
            # не найдено -> добавляем
            data.append(entry_template)
            meta_path.write_text(json.dumps(data, indent=2))
            return data[-1], data
        elif isinstance(data, dict):
            if data.get("model_id") == model_id:
                return data, data if isinstance(data, list) else [data]
            else:
                arr = [data, entry_template]
                meta_path.write_text(json.dumps(arr, indent=2))
                return arr[-1], arr
        else:
            # перезаписываем неизвестный формат
            meta_path.write_text(json.dumps([entry_template], indent=2))
            return entry_template, [entry_template]
    else:
        meta_path.write_text(json.dumps([entry_template], indent=2))
        return entry_template, [entry_template]


def persist_metadata(meta_path: Path, container: List[Dict[str, Any]]):
    meta_path.write_text(json.dumps(container, indent=2))


# ======================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adasmoothdelta'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--input', type=str, default='./input/train', help='корень обучающей выборки (содержит images/)')
    parser.add_argument('--evaluate', type=str, default='./input/evaluate', help='корень валидационной выборки (содержит images/)')
    parser.add_argument('--test', type=str, default='./input/test', help='корень тестовой выборки (содержит images/)')

    parser.add_argument('--output', type=str, default='./model')

    parser.add_argument('--model-name', type=str, default='fractal')
    parser.add_argument('--channels', type=int, nargs='+', default=[32, 64, 96, 128])
    parser.add_argument('--classifier-dim', type=int, default=256)

    parser.add_argument('--C', type=int, default=4)
    parser.add_argument('--drop-path', type=float, default=0.15)

    parser.add_argument('--checkpoint-every', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === ПУТИ К ДАННЫМ + загрузчики ===
    train_root = Path(args.input) / "images"
    val_root = Path(args.evaluate) / "images"
    test_root = Path(args.test) / "images"

    for p in (train_root, val_root, test_root):
        if not p.exists():
            raise FileNotFoundError(f"Не найдена необходимая папка датасета: {p}")

    image_size = 128  # как запрошено
    # image_size = 32
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_root, transform=transform)
    val_dataset = datasets.ImageFolder(val_root, transform=transform)
    test_dataset = datasets.ImageFolder(test_root, transform=transform)

    classes = train_dataset.classes
    num_classes = len(classes)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor = 2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # === МОДЕЛЬ ===
    model = FractalNet(
        num_classes=num_classes,
        channels=args.channels,
        classifier_dim=args.classifier_dim,
        C=args.C,
        drop_path_prob=args.drop_path
    ).to(device)

    # === ОПТИМИЗАТОР ===
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adasmoothdelta':
        optimizer = AdaSmoothDelta(model.parameters(), lr=args.lr)
    else:
        print(f"Неизвестный оптимизатор: {args.optimizer}")
        return

    # === ВЫХОДНЫЕ ПУТИ / model_id ===
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = next_model_index(out_dir, args.model_name, args.optimizer)
    model_id = f"trained_model_{args.model_name}_{args.optimizer}_v{idx}"
    final_model_path = out_dir / f"{model_id}.pth"
    best_model_filename = f"best_model_{args.model_name}_{args.optimizer}_v{idx}.pth"
    best_model_path = out_dir / best_model_filename

    meta_path = out_dir / "metadata.json"
    base_model = None

    # возобновление, если указано
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Файл для возобновления не найден: {resume_path}")
        ckpt = load_checkpoint(resume_path, model, optimizer, device)
        base_model = ckpt.get("model_id")  # может быть None, если не присутствует
        print(f"Возобновлены веса из {resume_path}; записанный base_model = {base_model}")

    # запись метаданных (одна на модель)
    entry_obj, container = load_or_create_metadata(meta_path, model_id, base_model, args, classes)
    entry_obj["start_time"] = datetime.now().strftime('%Y%m%d_%H%M%S')
    persist_metadata(meta_path, container)

    # цикл обучения
    criterion = nn.CrossEntropyLoss()
    best_val_acc = -1.0
    best_epoch = None

    print(f"Начало обучения: model_id={model_id}, device={device}, num_classes={num_classes}")
    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"\nЭпоха {epoch+1}/{args.epochs} начало: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # обучение
        #train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_loss, train_acc = train_one_epoch_optimized(model, train_loader, criterion, optimizer, device)

        # валидация (используется для выбора лучшей модели)
        val_loss, val_acc = evaluate_simple(model, val_loader, criterion, device)

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        # добавление в историю
        entry_obj["history"]["train_loss"].append(train_loss)
        entry_obj["history"]["train_acc"].append(train_acc)
        entry_obj["history"]["val_loss"].append(val_loss)
        entry_obj["history"]["val_acc"].append(val_acc)

        entry_obj["timestamp"] = datetime.now().strftime('%Y%m%d_%H%M%S')
        entry_obj["last_checkpoint_epoch"] = epoch + 1
        persist_metadata(meta_path, container)

        # проверка лучшей модели по val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            # сохранение state_dict лучшей модели
            torch.save(model.state_dict(), best_model_path)
            entry_obj["best"] = {
                "epoch": best_epoch,
                "val_acc": best_val_acc,
                "model_path": str(best_model_path)
            }
            persist_metadata(meta_path, container)
            print(f"Новая лучшая модель (эпоха {best_epoch}, val_acc={best_val_acc:.4f}) -> {best_model_path}")

        # сохранение чекпоинта, если необходимо (перезапись файла model_id)
        if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
            ckpt_path = out_dir / f"{model_id}.pth"
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_id": model_id,
                "epoch": epoch + 1
            }, ckpt_path)
            entry_obj["last_checkpoint_epoch"] = epoch + 1
            persist_metadata(meta_path, container)
            print(f"Чекпоинт сохранён: {ckpt_path}")

        print(f"Эпоха {epoch+1} завершена: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}; val_loss={val_loss:.4f}, val_acc={val_acc:.4f}; время={epoch_time:.1f}с")

    # обучение завершено
    entry_obj["end_time"] = datetime.now().strftime('%Y%m%d_%H%M%S')
    persist_metadata(meta_path, container)

    # сохранение итоговой модели (state_dict)
    torch.save(model.state_dict(), final_model_path)
    print(f"Итоговая модель сохранена: {final_model_path}")

    # === ФИНАЛЬНЫЕ МЕТРИКИ: вычисление точности по классам и полных метрик для двух моделей:
    # 1) final_model (только что сохранена)
    # 2) best_model (если существует)
    # Для каждой модели вычисляем метрики на train, val и test (test только сейчас)
    print("\nВычисление итоговых метрик по классам для итоговой модели и лучшей модели (если есть). Это может занять время.")

    def compute_all_metrics_for_model(model_path: Path, label: str) -> Dict[str, Any]:
        # загрузка весов (state_dict)
        data = torch.load(model_path, map_location=device)
        # определяем, содержит ли файл вложенный словарь или сырой state_dict
        if isinstance(data, dict) and "model_state" in data:
            model.load_state_dict(data["model_state"])
        elif isinstance(data, dict) and all(k in data for k in ("epoch", "optimizer_state")):
            # какой-то формат чекпоинта; пытаемся загрузить model_state если присутствует, иначе пробуем напрямую
            if "model_state" in data:
                model.load_state_dict(data["model_state"])
            elif "state_dict" in data:
                model.load_state_dict(data["state_dict"])
            else:
                try:
                    model.load_state_dict(data)
                except Exception:
                    pass
        else:
            try:
                model.load_state_dict(data)
            except Exception:
                # fallback: предполагаем, что data - это state_dict
                model.load_state_dict(data)

        # убеждаемся, что перемещено на устройство
        model.to(device)

        # вычисление метрик с помощью evaluate_with_per_class
        train_loss, train_acc, train_per_class = evaluate_with_per_class(model, train_loader, criterion, device, num_classes)
        val_loss, val_acc, val_per_class = evaluate_with_per_class(model, val_loader, criterion, device, num_classes)
        test_loss, test_acc, test_per_class = evaluate_with_per_class(model, test_loader, criterion, device, num_classes)

        # сортировка классов по val_per_class по убыванию для отчёта
        class_acc_pairs = list(zip(classes, val_per_class))
        class_acc_sorted = sorted(class_acc_pairs, key=lambda x: x[1], reverse=True)

        best_cls = class_acc_sorted[0]
        worst_cls = class_acc_sorted[-1]

        result = {
            "model_label": label,
            "model_path": str(model_path),
            "train": {
                "loss": train_loss,
                "acc": train_acc,
                "per_class": train_per_class
            },
            "val": {
                "loss": val_loss,
                "acc": val_acc,
                "per_class": val_per_class
            },
            "test": {
                "loss": test_loss,
                "acc": test_acc,
                "per_class": test_per_class
            },
            "val_best_class": {"name": best_cls[0], "acc": best_cls[1]},
            "val_worst_class": {"name": worst_cls[0], "acc": worst_cls[1]},
            "computed_at": datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        return result

    # вычисление для итоговой модели
    final_metrics = compute_all_metrics_for_model(final_model_path, label="final")
    entry_obj["final"] = final_metrics
    persist_metadata(meta_path, container)
    print("Метрики итоговой модели вычислены и сохранены в metadata в разделе 'final'.")

    # вычисление для лучшей модели, если существует и отличается от итоговой
    if entry_obj.get("best") and entry_obj["best"].get("model_path"):
        best_path = Path(entry_obj["best"]["model_path"])
        if best_path.exists():
            if str(best_path) != str(final_model_path):
                best_metrics = compute_all_metrics_for_model(best_path, label="best")
                entry_obj["best_summary"] = best_metrics
                persist_metadata(meta_path, container)
                print("Метрики лучшей модели вычислены и сохранены в metadata в разделе 'best_summary'.")
            else:
                # final == best
                entry_obj["best_summary"] = final_metrics
                persist_metadata(meta_path, container)
                print("Лучшая модель совпадает с итоговой; best_summary установлены как метрики итоговой модели.")
        else:
            print("Предупреждение: файл лучшей модели, указанный в metadata, не найден:", best_path)
    else:
        print("Лучшая модель не была записана в процессе обучения.")

    print("\nОбучение и финальная оценка завершены. Метаданные обновлены.")

if __name__ == "__main__":
    main()