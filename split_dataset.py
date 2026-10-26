import os
import shutil
import random

input_dir = "dataset_converted"
output_dir = "dataset_split"

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

for cls in os.listdir(input_dir):
    cls_path = os.path.join(input_dir, cls)
    
    if not os.path.isdir(cls_path):
        continue
    
    images = os.listdir(cls_path)
    random.shuffle(images)
    
    train_end = int(len(images) * train_ratio)
    val_end = int(len(images) * (train_ratio + val_ratio))
    
    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }
    
    for split, imgs in splits.items():
        split_cls_path = os.path.join(output_dir, split, cls)
        os.makedirs(split_cls_path, exist_ok=True)
        
        for img in imgs:
            src = os.path.join(cls_path, img)
            dst = os.path.join(split_cls_path, img)
            shutil.copy(src, dst)

print("Dataset ayrıldı 🚀")