import os
import random
import shutil

# Path Setting
base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "images/train")
label_dir = os.path.join(base_dir, "labels/train")
val_name = "val"

val_image_dir = os.path.join(image_dir, f'../{val_name}')
val_label_dir = os.path.join(label_dir, f'../{val_name}')
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)



# Get all image file names (.jpg or .png)
image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]

# Shuffle randomly (fixed random.seed ensures reproducibility)
random.seed(42)  # Random seed to ensure consistent splits each time
random.shuffle(image_files)



# Calculate split index (90:10)
split_idx = int(0.9 * len(image_files))
train_files = image_files[:split_idx]  # First 90% for training set
val_files = image_files[split_idx:]    # Last 10% for validation set



# # Write to train.txt
# with open(os.path.join(output_dir, "train.txt"), "w") as f:
#     for img in train_files:
#         f.write(f"{image_dir}/{img}\n")  # Write the full path

# # Write to val.txt
# with open(os.path.join(output_dir, "val.txt"), "w") as f:
#     for img in val_files:
#         f.write(f"{image_dir}/{img}\n")

# # Print statistics
# print(f"Total number of images: {len(image_files)}")
# print(f"Training set: {len(train_files)} images")
# print(f"Validation set: {len(val_files)} images")
# print(f"Split completed! Files saved in: {output_dir}")



for img in val_files:
    src_path = os.path.join(image_dir, img)
    dst_path = os.path.join(val_image_dir, img)
    shutil.move(src_path, dst_path)

    name = os.path.splitext(img)[0]
    label = f'{name}.txt'
    src_path = os.path.join(label_dir, label)
    dst_path = os.path.join(val_label_dir, label)
    shutil.move(src_path, dst_path)
