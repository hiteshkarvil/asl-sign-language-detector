import os
import shutil

keep = {'A', 'NOTHING'}  # folders to keep
base_dir = 'mini_train'

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path) and folder not in keep:
        print(f"🗑 Deleting {folder_path}")
        shutil.rmtree(folder_path)
