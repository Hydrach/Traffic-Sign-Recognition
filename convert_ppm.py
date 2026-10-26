import os
from PIL import Image

input_folder = "GTSRB/Final_Training/Images"
output_folder = "dataset_converted"

os.makedirs(output_folder, exist_ok=True)

for class_folder in os.listdir(input_folder):
    class_path = os.path.join(input_folder, class_folder)
    
    if not os.path.isdir(class_path):
        continue

    save_class_path = os.path.join(output_folder, class_folder)
    os.makedirs(save_class_path, exist_ok=True)

    for file in os.listdir(class_path):
        if file.endswith(".ppm"):
            img_path = os.path.join(class_path, file)
            img = Image.open(img_path)

            new_name = file.replace(".ppm", ".jpg")
            img.save(os.path.join(save_class_path, new_name))

print("Dönüştürme tamamlandı 🚀")