# FractalNet модель 

## Введение:
FractalNet — это самоподобная нейросеть, построенная по фрактальному правилу расширения.
Вместо обычных линейных слоёв или резидуальных переходов (как в ResNet), FractalNet формирует множественные пути разной длины от входа к выходу, объединяя их на каждом уровне.

## Подготовка окружения
Для разработки данных скриптов использовалась python версии 3.12 и соответствующие библиотеки (без Torch и TorchVision):
- requests

### Установка библиотек Torch и TorchVision
Чтобы установить соответствующие версии torch и torchvision необходимо сверить версию CUDA поддерживаемой вашей видеокартой, 
по возможности обновить до последней доступной версии и уже после с официального сайта [PyTorch](https://pytorch.org/get-started/locally/)
установить необходимые библиотеки (последняя стабильная версия PyTorch требует Python 3.9). В моём случае RTX 5070 Ti поддерживает 13.0 версию CUDA и установку библиотек можно выполнить с помощью данной команды:

```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Если нет видеокарты или используется встроенная видеоускоритель, то установку библиотек можно выполнить уже с помощью данной команды:

```bash
pip uninstall torch torchvision -y 
pip install torch torchvision
```

## Характеристики вычислительной техники и ПО:
- Процессор: AMD Ryzen 7 7700, 8 ядер, 3.8 ГГц - 5.3 ГГц, L3 32 МБ
- Оперативная память: 16x4 Гб, 6000 МГц, 30 - 40 - 40
- Видеокарта: NVIDIA GeForce RTX 5070 Ti, 16Гб
- CUDA Version: 13.0

## Источники:
- Статья авторства Gustav Larsson, Michael Maire и Gregory Shakhnarovich [FRACTALNET:
ULTRA-DEEP NEURAL NETWORKS WITHOUT RESIDUALS](https://arxiv.org/pdf/1605.07648).
- Статья авторства Jun Lu [AdaSmooth: An Adaptive Learning Rate Method based on Effective Ratio](https://arxiv.org/pdf/2204.00825v1)
- Документация [PyTorch](https://docs.pytorch.org/docs/stable/index.html)
- 