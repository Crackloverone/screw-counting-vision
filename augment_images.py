# -*- coding: utf-8 -*-
"""
图片数据增强脚本 - 只处理原始capture图片
"""
import cv2
import numpy as np
from pathlib import Path
import random

SCRIPT_DIR = Path(__file__).parent
SCREENSHOTS_DIR = SCRIPT_DIR / 'screenshots'


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(image, matrix, (new_w, new_h))


def flip_horizontal(image):
    return cv2.flip(image, 1)


def flip_vertical(image):
    return cv2.flip(image, 0)


def adjust_brightness(image, factor=None):
    if factor is None:
        factor = random.uniform(0.7, 1.3)
    return np.clip(image * factor, 0, 255).astype(np.uint8)


def adjust_contrast(image, factor=None):
    if factor is None:
        factor = random.uniform(0.7, 1.3)
    mean = image.mean()
    return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)


def gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def add_noise(image):
    noise = np.random.normal(0, 15, image.shape).astype(np.uint8)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


AUGMENTATION_COMBOS = [
    ('flip_h', flip_horizontal),
    ('flip_v', flip_vertical),
    ('br', lambda img: adjust_brightness(img)),
    ('ct', lambda img: adjust_contrast(img)),
    ('blur', gaussian_blur),
    ('noise', add_noise),
    ('r90', lambda img: rotate_image(img, 90)),
    ('r180', lambda img: rotate_image(img, 180)),
    ('r270', lambda img: rotate_image(img, 270)),
]


def process_folder(folder_path, suffix_to_skip=['_br', '_ct', '_blur', '_noise', '_r90', '_r180', '_r270', '_flip', '_hflip', '_vflip']):
    folder_path = Path(folder_path)
    
    # 只处理原始capture图片（以capture_开头且没有增强后缀的）
    all_images = list(folder_path.glob('*.jpg'))
    original_images = [img for img in all_images if 'capture_' in img.name and not any(s in img.name for s in suffix_to_skip)]
    
    print(f"处理 {folder_path.name}: {len(original_images)} 张原图")
    total = 0
    
    for img_path in original_images:
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            name = img_path.stem
            random.shuffle(AUGMENTATION_COMBOS)
            
            # 每张图生成3种增强
            for suffix, func in AUGMENTATION_COMBOS[:3]:
                aug_image = func(image)
                aug_path = folder_path / f"{name}_{suffix}.jpg"
                cv2.imwrite(str(aug_path), aug_image)
                total += 1
                
        except Exception as e:
            print(f"错误: {e}")
    
    print(f"  生成 {total} 张增强图")
    return total


def main():
    total = 0
    for folder in ['train', 'val', 'test']:
        path = SCREENSHOTS_DIR / folder
        if path.exists():
            total += process_folder(path)
    
    print(f"\n总计生成 {total} 张增强图")


if __name__ == '__main__':
    main()