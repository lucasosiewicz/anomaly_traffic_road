from pathlib import Path
from tqdm import tqdm
import shutil

"""
This script takes images from DADA dataset and organizes them into a single directory called 'images'.
"""

DATASET_PATH = r"D:\MAGISTERKA\anomaly_traffic_road\datasets\DADA\DADA2000"

def get_directories_list(path):
    return [d for d in Path(path).iterdir() if d.is_dir()]


def main():

    i = 0

    # Create new directory
    new_dir = Path(f"{DATASET_PATH}/images")
    new_dir.mkdir(exist_ok=True)


    # Get list of all directories in DADA dataset
    head_dirs = get_directories_list(DATASET_PATH)


    for directory in tqdm(head_dirs):
        print(f"Processing directory: {directory}")
        subdirs = get_directories_list(directory)
        for subdir in subdirs:
            print(f"Processing subdirectory: {subdir}")
            subdir.rename(new_dir / f"{i}")
            i += 1
            
                    



if __name__ == '__main__':
    main()