import cv2
import json
import torch
from pathlib import Path
from kornia.geometry import resize
from torch.utils.data import Dataset

class SupervisedDataset(Dataset):
    def __init__(self, path_to_data: str, which_set: str, resize_target: tuple = (227, 227)):

        assert which_set in ['train', 'val', 'test'], f"which_set must be one of ['train', 'val', 'test'], got {which_set}"

        self.path_to_data = Path(path_to_data)
        self.images, self.labels = self._load_data(which_set)
        self.resize_target = resize_target

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(cv2.imread(str(self.images[idx]), cv2.IMREAD_GRAYSCALE))
        if len(img.shape) == 2:
            img = img.unsqueeze(2)
        img = img.permute(2, 0, 1).float() / 255.
        img = resize(img, self.resize_target)

        return img, self.labels[idx]

    def _load_data(self, which_set: str):
        label_paths = [label for label in (self.path_to_data / 'annotations' / which_set).rglob('*.json')]
        images = []
        targets = []
        for label_path in label_paths:
            with open(label_path, 'r', encoding='utf-8') as f:
                labels_data = json.load(f)
                if "labels" not in labels_data:
                    print(f"Warning: Key 'labels' not found in file {label_path}. Skipping file.")
                    continue
                labels = labels_data["labels"]

                try:
                    labels.sort(key=lambda x: x['image_path'])
                except KeyError:
                    print(f"Warning: Key 'image_path' not found in one of the labels in file {label_path}. Sorting skipped.")
                except Exception as e:
                    print(f"Error while sorting labels in file {label_path}: {e}. Sorting skipped.")

                temp_images_for_file = []
                temp_targets_for_file = []
                seen_anomaly_in_file = False

                for label in labels:
                    accident_id = label.get('accident_id', -1)
                    image_path_str = label.get('image_path')

                    if image_path_str is None:
                        print(f"Warning: Key 'image_path' not found in label in file {label_path}. Skipping label.")
                        continue

                    image_path = self.path_to_data / "frames" / which_set / image_path_str.replace("frames/", "")

                    if not image_path.exists():
                        continue

                    if accident_id != 0:
                        seen_anomaly_in_file = True
                        temp_images_for_file.append(image_path)
                        temp_targets_for_file.append(accident_id)
                    else:
                        if seen_anomaly_in_file:
                            break
                        else:
                            temp_images_for_file.append(image_path)
                            temp_targets_for_file.append(0)

            images.extend(temp_images_for_file)
            targets.extend(temp_targets_for_file)

        if not images:
            raise ValueError(f"No valid image files found in the set {which_set}")

        return images, torch.tensor(targets)

if __name__ == "__main__":
    dataset = SupervisedDataset(path_to_data='datasets/DoTA', which_set='train')
    print(len(dataset))
    print(dataset[0], dataset[1], dataset[2])
