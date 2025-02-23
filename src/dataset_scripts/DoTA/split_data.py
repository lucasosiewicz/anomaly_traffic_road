from pathlib import Path
import shutil

DATASET_PATH = r"D:\MAGISTERKA\anomaly_traffic_road\datasets\DoTA"

def get_directories_list(path):
    return [d for d in path.iterdir() if d.is_dir()]

def get_files_list(path):
    return [f for f in path.iterdir() if f.is_file()]

def create_set_dirs(path, set_dirs):
    for dir in set_dirs:
        Path(path / dir).mkdir(parents=True, exist_ok=True)

def load_split_instructions(train_path, val_path):
    # train split
    with open(train_path, 'r') as f:
        train_split = f.readlines()
        train_split = list(map(lambda x: x.strip(), train_split))

    # val split
    with open(val_path, 'r') as f:
        val_split = f.readlines()
        val_split = list(map(lambda x: x.strip(), val_split))
    
    return train_split, val_split

def split_data(train_split, val_split, dir_list, annotations_list):

    # split frames
    for dir in dir_list:
        if dir.name in train_split:
            dir.rename(Path(DATASET_PATH) / 'frames' / 'train' / dir.name)
        elif dir.name in val_split:
            dir.rename(Path(DATASET_PATH) / 'frames' / 'val' / dir.name)
        else:
            dir.rmtree()

    # split annotations
    for annotation in annotations_list:
        if annotation.name.replace('.json', '') in train_split:
            annotation.rename(Path(DATASET_PATH) / 'annotations' / 'train' / annotation.name)
        elif annotation.name.replace('.json', '') in val_split:
            annotation.rename(Path(DATASET_PATH) / 'annotations' / 'val' / annotation.name)


def main():
    
    set_dirs = ['train', 'val', 'test']
    dir_list = get_directories_list(Path(DATASET_PATH) / 'frames')
    annotations_list = get_files_list(Path(DATASET_PATH) / 'annotations')
    print(len(annotations_list))

    for dir in ['frames', 'annotations']:
        create_set_dirs(Path(DATASET_PATH) / dir, set_dirs)

    train_split, val_split = load_split_instructions('src/dataset_scripts/DoTA/train_split.txt', 
                                                     'src/dataset_scripts/DoTA/val_split.txt')
    

    split_data(train_split, val_split, dir_list, annotations_list)




if __name__ == '__main__':
    main()