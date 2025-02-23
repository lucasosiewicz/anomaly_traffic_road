from pathlib import Path
from tqdm import tqdm
import shutil

"""
This script removes unnecessary directories from DADA dataset and keeps only images.
"""

DATASET_PATH = r"D:\MAGISTERKA\anomaly_traffic_road\datasets\DADA\DADA2000"
DIR_NAMES = ['fixation', 'maps', 'seg', 'semantic']


def get_directories_list(path):
    return [d for d in Path(path).iterdir() if d.is_dir()]


def main():
    head_dirs = get_directories_list(DATASET_PATH)

    for directory in tqdm(head_dirs):
        print(f"Processing directory: {directory}")
        subdirs = get_directories_list(directory)
        for subdir in subdirs:
            print(f"Processing subdirectory: {subdir}")
            data = get_directories_list(subdir)
            for d in data:
                if d.name in DIR_NAMES:
                    shutil.rmtree(d)



if __name__ == '__main__':
    main()