!pip install ultralytics -q
import os, glob, zipfile, random, shutil, yaml
from ultralytics import YOLO

# 1. Automatic Dataset Location Finder
input_dir = '/kaggle/input'
extract_path = '/kaggle/working/dataset_raw'

# Find all zip files in Kaggle input
zip_files = glob.glob(f"{input_dir}/**/*.zip", recursive=True)

if zip_files:
    print(f"Extracting zip dataset: {zip_files[0]}")
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    search_base = extract_path
else:
    print("Zip not found, searching direct input directory...")
    search_base = input_dir

# Find images and labels directories dynamically
images_dirs = glob.glob(f"{search_base}/**/images", recursive=True)
labels_dirs = glob.glob(f"{search_base}/**/labels", recursive=True)

if not images_dirs or not labels_dirs:
    raise ValueError("Could not find 'images' or 'labels' folder.")

images_dir = images_dirs[0]
labels_dir = labels_dirs[0]

print(f"Found Images Path: {images_dir}")
print(f"Found Labels Path: {labels_dir}")

# 2. Train-Val Split Setup (80-20 Split)
final_dataset_path = '/kaggle/working/dataset'
for folder in ['train/images', 'train/labels', 'val/images', 'val/labels']:
    os.makedirs(os.path.join(final_dataset_path, folder), exist_ok=True)

all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
random.seed(42)
random.shuffle(all_images)

split_index = int(len(all_images) * 0.8)
train_images, val_images = all_images[:split_index], all_images[split_index:]

def move_files(files, split_name):
    for filename in files:
        shutil.copy(os.path.join(images_dir, filename), os.path.join(final_dataset_path, split_name, 'images', filename))
        label_filename = os.path.splitext(filename)[0] + '.txt'
        if os.path.exists(os.path.join(labels_dir, label_filename)):
            shutil.copy(os.path.join(labels_dir, label_filename), os.path.join(final_dataset_path, split_name, 'labels', label_filename))

move_files(train_images, 'train')
move_files(val_images, 'val')

# 3. Create CORRECTED data.yaml matching classes.txt order
yaml_content = {
    'path': '/kaggle/working/dataset',
    'train': 'train/images',
    'val': 'val/images',
    'nc': 4,
    'names': ['debris', 'landslide', 'structures', 'uprooted_tree']
}
with open('/kaggle/working/dataset/data.yaml', 'w') as f:
    yaml.dump(yaml_content, f, default_flow_style=False)

print("Dataset prepared successfully!")
