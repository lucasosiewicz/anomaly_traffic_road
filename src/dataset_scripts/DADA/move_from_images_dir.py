from pathlib import Path
from tqdm import tqdm
import time
import shutil

"""
This script takes images from DADA dataset and organizes them into a single directory called 'images'.
"""

DATASET_PATH = r"D:\MAGISTERKA\anomaly_traffic_road\datasets\DADA\DADA2000\images"

def get_directories_list(path):
    return [d for d in Path(path).iterdir() if d.is_dir()]

def get_files_list(path):
    return [f for f in Path(path).iterdir() if f.is_file()]


def main():

    head_dirs = get_directories_list(DATASET_PATH)

    for directory in tqdm(head_dirs):
        print(f"Processing directory: {directory}")
        subpath = directory / 'images'
        images = get_files_list(subpath)
        for image in images:
            shutil.move(image, directory)
        subpath.rmdir()
            
                

if __name__ == '__main__':
    main()