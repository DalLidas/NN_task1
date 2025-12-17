"""
Скачивает Stanford Dogs (images.tar, annotation.tar), распаковывает и делит по классам: 70% train / 15% evaluate / 15% test.
Поддерживает --from-local (ожидает /Images).
Опционально: --url-images и --url-annotations для кастомных ссылок.

Запуск:
python download_stanford_dogs.py --out ./input
"""


import argparse
import os
import tarfile
import shutil
import random
from pathlib import Path
import requests
from collections import defaultdict


def download_file(url, dest_path, chunk_size=1024*1024):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest_path}")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    print("Downloaded.")


def safe_extract(tar_path, dest_dir):
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=dest_dir)


def split_dataset(img_root: Path, out_root: Path, seed=42):
    random.seed(seed)
    classes = [d for d in img_root.iterdir() if d.is_dir()]
    stats = {}

    for cls in classes:
        imgs = sorted([p for p in cls.iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png']])
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * 0.70)
        n_eval = int(n * 0.15)
        n_test = n - n_train - n_eval
        parts = {
            'train': imgs[:n_train],
            'evaluate': imgs[n_train:n_train+n_eval],
            'test': imgs[n_train+n_eval:]
        }

        for part, files in parts.items():
            for p in files:
                dest = out_root / part / 'images' / cls.name
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest / p.name)
        stats[cls.name] = (len(parts['train']), len(parts['evaluate']), len(parts['test']))

    # print stats summary
    counts = [sum(v) for v in stats.values()]
    per_class_total = [sum(v) for v in stats.values()]
    print(f"Stats summary: counts={counts}, per_class_total={per_class_total}")

    totals = defaultdict(int)

    for v in stats.values():
        totals['train'] += v[0]
        totals['evaluate'] += v[1]
        totals['test'] += v[2]
    print("Totals (train/evaluate/test):", totals['train'], totals['evaluate'], totals['test'])

    per_class_counts = [sum(v) for v in stats.values()]
    mn = min(per_class_counts)
    mx = max(per_class_counts)
    avg = sum(per_class_counts)/len(per_class_counts)
    print(f"per-class total: min={mn}, max={mx}, mean={avg:.2f}")

    # also print per-class breakdown
    for cls, (a, b, c) in stats.items():
        print(f"{cls}: train={a}, eval={b}, test={c}")


def main():
    BASE_IMAGES_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    BASE_ANNOT_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='./input', help='output folder')
    parser.add_argument('--from-local', action='store_true', help='use raw_data/Images instead of downloading')
    parser.add_argument('--url-images', type=str, help='custom URL for images.tar')
    parser.add_argument('--url-annotations', type=str, help='custom URL for annotation.tar')
    args = parser.parse_args()

    out = Path(args.out)
    from_local = args.from_local
    BASE_IMAGES_URL = args.url_images or BASE_IMAGES_URL
    BASE_ANNOT_URL = args.url_annotations or BASE_ANNOT_URL

    if args.from_local:
        img_root = from_local / 'Images'
        if not img_root.exists():
            raise RuntimeError('from_local/Images not found')
    else:
        downloads = Path('./downloads')
        downloads.mkdir(exist_ok=True)
        images_tar = downloads / 'images.tar'
        ann_tar = downloads / 'annotation.tar'

        if not images_tar.exists():
            try:
                download_file(BASE_IMAGES_URL, images_tar)
            except Exception as e:
                print('Automatic download failed:', e)
                print('Please download images.tar manually from Stanford site and place it in ./downloads or use --from-local')
                return

        if not ann_tar.exists():
            try:
                download_file(BASE_ANNOT_URL, ann_tar)
            except Exception as e:
                print('Annotation download failed:', e)
                print('Proceeding without annotations.')

        extract_dir = Path('./extracted')
        extract_dir.mkdir(exist_ok=True)

        print('Extracting images...')

        safe_extract(images_tar, extract_dir)
        img_root = extract_dir / 'Images'

    # split
    split_dataset(img_root, out)


if __name__ == '__main__':
    main()
