import cv2
import json
import torch
from pathlib import Path
from kornia.geometry import resize
from torch.utils.data import Dataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Dataset(Dataset):
    def __init__(self, path_to_data: str, which_set: str, resize_target: tuple = (227, 227)):

        assert which_set in ['train', 'val', 'test'], f"which_set must be one of ['train', 'val', 'test'], got {which_set}"

        self.path_to_data = Path(path_to_data) # datasets\DoTA
        self.data, self.labels = self._load_data(which_set) # dataset\DoTA\frames
        self.resize_target = resize_target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.data[idx]), cv2.IMREAD_GRAYSCALE)
        if len(img.shape) == 2:
            img = torch.tensor(img).unsqueeze(2)
        img = torch.tensor(img).permute(2, 0, 1).float().to(DEVICE) / 255
        img = resize(img, self.resize_target)

        return img, self.labels[idx]
        

    def _load_data(self, which_set: str):
        video_paths = sorted([directory 
                              for directory 
                              in (self.path_to_data / 'frames' / which_set).iterdir() 
                              if directory.is_dir()])
        
        label_paths = sorted([label 
                              for label 
                              in (self.path_to_data / 'annotations' / which_set).rglob('*.json')])
        
        data = []
        targets = None

        for video_path, label_path in zip(video_paths, label_paths):
            video_name = video_path.name
            label_name = label_path.name.replace('.json', '')
            assert video_name == label_name, f"Video name {video_name} does not match label name {label_name}"

            # Load the labels
            with open(label_path, 'r') as f:
                labels = json.load(f)
                start = labels['anomaly_start']
                end = labels['anomaly_end']

            # Filter the frames
            video_path = video_path / 'images'
            frames = [frame for frame in video_path.rglob('*.jpg')]
            labels = torch.cat([torch.zeros(start), torch.ones(end - start), torch.zeros(len(frames) - end)])
            if which_set in ['train', 'val']:
                frames = list(set(frames) - set(frames[start:end]))
                labels = torch.zeros(len(frames))

            data.extend(frames)
            if targets is None:
                targets = labels
            else:
                targets = torch.cat([targets, labels])

        return data, targets
    


if __name__ == '__main__':
    path_to_data = r"D:\MAGISTERKA\anomaly_traffic_road\datasets\DoTA"
    which_set = 'train'
    dataset = Dataset(path_to_data, which_set)