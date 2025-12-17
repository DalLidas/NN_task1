#!/usr/bin/env python3
"""
plot_training_curves_simple.py

Создает графики потерь и точности от эпохи для конкретной модели.
Сохраняет графики в формате PNG.

Использование:
python plot_training_curves.py --model-dir ./model --model-id trained_model_fractal_adam_v0 --output-dir ./model
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(entry: Dict, output_dir: Path, figsize: tuple = (14, 10), dpi: int = 150) -> None:
    """
    Создает графики потерь и точности на основе записи модели.

    Args:
        entry: Словарь с данными модели из metadata.json
        output_dir: Директория для сохранения графиков
        figsize: Размер фигуры (ширина, высота)
        dpi: Разрешение для сохранения
    """
    if "history" not in entry:
        print(f"В записи модели '{entry.get('model_id', 'unknown')}' нет данных истории")
        return

    history = entry["history"]

    # Извлекаем данные
    train_loss = history.get("train_loss", [])
    train_acc = history.get("train_acc", [])
    val_loss = history.get("val_loss", [])
    val_acc = history.get("val_acc", [])

    if not train_loss:
        print(f"В истории модели '{entry.get('model_id', 'unknown')}' нет данных обучения")
        return

    epochs = list(range(1, len(train_loss) + 1))

    # Информация о модели для заголовков
    model_id = entry.get("model_id", "Unknown")
    optimizer = entry.get("optimizer", "Unknown")
    epochs_total = entry.get("epochs", len(epochs))
    lr = entry.get("lr", "Unknown")
    batch_size = entry.get("batch_size", "Unknown")

    title_suffix = f" (оптимизатор: {optimizer}, lr: {lr}, batch: {batch_size})"

    # Создаем директорию для выходных файлов
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Комбинированный график (потери и точность на одной фигуре)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle(f'Обучение модели: {model_id}{title_suffix}', fontsize=16)

    # График потерь
    ax1.plot(epochs, train_loss, 'b-', label='Обучающая', linewidth=2)
    if val_loss:
        ax1.plot(epochs, val_loss, 'r-', label='Валидационная', linewidth=2)
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Потери')
    ax1.set_title('Потери от эпохи')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Добавляем лучшую эпоху по валидационным потерям
    if val_loss:
        best_val_loss_epoch = np.argmin(val_loss) + 1
        best_val_loss = min(val_loss)
        ax1.axvline(x=best_val_loss_epoch, color='g', linestyle='--', alpha=0.7)
        ax1.text(best_val_loss_epoch, max(train_loss + val_loss) * 0.9,
                 f'Лучшая val_loss\nэпоха {best_val_loss_epoch}\n({best_val_loss:.4f})',
                 horizontalalignment='center',
                 verticalalignment='top',
                 fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # График точности
    ax2.plot(epochs, train_acc, 'b-', label='Обучающая', linewidth=2)
    if val_acc:
        ax2.plot(epochs, val_acc, 'r-', label='Валидационная', linewidth=2)
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Точность')
    ax2.set_title('Точность от эпохи')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Добавляем лучшую эпоху по валидационной точности
    if val_acc:
        best_val_acc_epoch = np.argmax(val_acc) + 1
        best_val_acc = max(val_acc)
        ax2.axvline(x=best_val_acc_epoch, color='g', linestyle='--', alpha=0.7)
        ax2.text(best_val_acc_epoch, min(train_acc + val_acc) * 1.1,
                 f'Лучшая val_acc\nэпоха {best_val_acc_epoch}\n({best_val_acc:.4f})',
                 horizontalalignment='center',
                 verticalalignment='bottom',
                 fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    combined_filename = output_dir / f"training_curves_{model_id}.png"
    plt.savefig(combined_filename, dpi=dpi, bbox_inches='tight')
    print(f"✓ Сохранен комбинированный график: {combined_filename}")
    plt.close()

    # 2. График потерь отдельно
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_loss, 'b-', label='Обучающая', linewidth=2)
    if val_loss:
        plt.plot(epochs, val_loss, 'r-', label='Валидационная', linewidth=2)

    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.title(f'Потери от эпохи\n{model_id}{title_suffix}')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Добавляем лучшую эпоху по валидационным потерям
    if val_loss:
        best_val_loss_epoch = np.argmin(val_loss) + 1
        best_val_loss = min(val_loss)
        plt.axvline(x=best_val_loss_epoch, color='g', linestyle='--', alpha=0.7)
        plt.text(best_val_loss_epoch, max(train_loss + val_loss) * 0.9,
                 f'Лучшая val_loss\nэпоха {best_val_loss_epoch}\n({best_val_loss:.4f})',
                 horizontalalignment='center',
                 verticalalignment='top',
                 fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    loss_filename = output_dir / f"loss_curve_{model_id}.png"
    plt.savefig(loss_filename, dpi=dpi, bbox_inches='tight')
    print(f"✓ Сохранен график потерь: {loss_filename}")
    plt.close()

    # 3. График точности отдельно
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_acc, 'b-', label='Обучающая', linewidth=2)
    if val_acc:
        plt.plot(epochs, val_acc, 'r-', label='Валидационная', linewidth=2)

    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.title(f'Точность от эпохи\n{model_id}{title_suffix}')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Добавляем лучшую эпоху по валидационной точности
    if val_acc:
        best_val_acc_epoch = np.argmax(val_acc) + 1
        best_val_acc = max(val_acc)
        plt.axvline(x=best_val_acc_epoch, color='g', linestyle='--', alpha=0.7)
        plt.text(best_val_acc_epoch, min(train_acc + val_acc) * 1.1,
                 f'Лучшая val_acc\nэпоха {best_val_acc_epoch}\n({best_val_acc:.4f})',
                 horizontalalignment='center',
                 verticalalignment='bottom',
                 fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    acc_filename = output_dir / f"accuracy_curve_{model_id}.png"
    plt.savefig(acc_filename, dpi=dpi, bbox_inches='tight')
    print(f"✓ Сохранен график точности: {acc_filename}")
    plt.close()

    # 4. Выводим статистику
    print(f"\nСтатистика для модели {model_id}:")
    print(f"   Всего эпох: {len(epochs)}")
    print(f"   Минимальная обучающая потеря: {min(train_loss):.4f}")
    if val_loss:
        print(f"   Минимальная валидационная потеря: {min(val_loss):.4f} (эпоха {best_val_loss_epoch})")
    print(f"   Максимальная обучающая точность: {max(train_acc):.4f}")
    if val_acc:
        print(f"   Максимальная валидационная точность: {max(val_acc):.4f} (эпоха {best_val_acc_epoch})")


def find_model_entry_by_id(metadata: list, model_id: str) -> Dict:
    """Находит запись модели по model_id."""
    for entry in metadata:
        if entry.get("model_id") == model_id:
            return entry
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Создает графики потерь и точности для конкретной модели"
    )

    parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help='Директория с моделями (ищет metadata.json внутри)'
    )

    parser.add_argument(
        '--model-id',
        type=str,
        required=True,
        help='ID конкретной модели для построения графиков (например: trained_model_fractal_adam_v0)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./plots',
        help='Директория для сохранения графиков (по умолчанию: ./plots)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI для сохранения изображений (по умолчанию: 150)'
    )

    args = parser.parse_args()

    # Определяем путь к metadata.json
    metadata_path = Path(args.model_dir) / "metadata.json"

    if not metadata_path.exists():
        print(f"Файл metadata.json не найден: {metadata_path}")
        return

    # Загружаем метаданные
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_content = json.load(f)
    except Exception as e:
        print(f"Ошибка при загрузке metadata.json: {e}")
        return

    # Преобразуем в список, если это словарь
    if isinstance(metadata_content, dict):
        metadata = [metadata_content]
    else:
        metadata = metadata_content

    # Ищем нужную модель
    entry = find_model_entry_by_id(metadata, args.model_id)

    if not entry:
        print(f"Модель с ID '{args.model_id}' не найдена в metadata.json")
        print("\n📋 Доступные модели:")
        for entry in metadata:
            if "model_id" in entry:
                print(f"  - {entry['model_id']}")
        return

    print(f"Найдена модель: {args.model_id}")

    # Создаем графики
    plot_training_curves(
        entry=entry,
        output_dir=Path(args.output_dir),
        dpi=args.dpi
    )

    print(f"\nВсе графики сохранены в директории: {args.output_dir}")


if __name__ == "__main__":
    main()