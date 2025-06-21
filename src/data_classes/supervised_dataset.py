import cv2
import json
import torch
from pathlib import Path
from kornia.geometry import resize
from torch.utils.data import Dataset
import ast

class SupervisedDataset(Dataset):
    def __init__(self, path_to_data: str, which_set: str, resize_target: tuple = (227, 227), target_class: str | int = 'all', crop_type: str = None, dataset_name: str = 'DoTA'):

        assert which_set in ['train', 'val', 'test'], f"which_set must be one of ['train', 'val', 'test'], got {which_set}"

        self.path_to_data = Path(path_to_data)
        self.resize_target = resize_target
        self.target_class = target_class
        self.crop_type = crop_type
        self.which_set = which_set
        self.dataset_name = dataset_name
        self.images, self.labels = self.load_data(dataset_name, which_set)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = str(self.images[idx])
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # type: ignore
        
        if img is None:
            raise ValueError(f"Could not load image from path: {image_path}")
        
        img = torch.tensor(img)
        if self.crop_type:
            if self.crop_type == 'manual':
                img = img[300:-100]
        if len(img.shape) == 2:
            img = img.unsqueeze(2)
        img = img.permute(2, 0, 1).float() / 255.
        img = resize(img, self.resize_target)

        return img, self.labels[idx]

    def _load_dota(self, which_set: str):
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
                    image_path_str = label.get('image_path')

                    if image_path_str is None:
                        print(f"Warning: Key 'image_path' not found in label in file {label_path}. Skipping label.")
                        continue

                    image_path = self.path_to_data / "frames" / which_set / image_path_str.replace("frames/", "")

                    if not image_path.exists():
                        continue

                    original_accident_id = label.get('accident_id', -1)
                    
                    if original_accident_id == 0:
                        mapped_id = 0
                    elif original_accident_id in [1, 2, 3, 4, 5]:
                        mapped_id = 1
                    elif original_accident_id == 6:
                        mapped_id = 2
                    elif original_accident_id == 7:
                        mapped_id = 3
                    elif original_accident_id in [8, 9]:
                        mapped_id = 4
                    elif original_accident_id == 10:
                        mapped_id = 5
                    else:
                         continue

                    # Filter based on target_class parameter
                    if self.target_class == 'all':
                        # Include all classes
                        should_include = True
                    elif isinstance(self.target_class, int):
                        # Include only class 0 and the specified target_class
                        should_include = mapped_id == 0 or mapped_id == self.target_class
                    else:
                        # Invalid target_class value, skip
                        should_include = False

                    if not should_include:
                        continue

                    if mapped_id != 0:
                        seen_anomaly_in_file = True
                        temp_images_for_file.append(image_path)
                        temp_targets_for_file.append(mapped_id)
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
    
    def _load_carcrash(self, which_set: str):
        # Read labels from labels.txt
        labels_file = self.path_to_data / 'labels.txt'
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        all_data = []
        
        with open(labels_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse the line: directory_name,[list_of_targets],other_params...
                first_comma_idx = line.find(',')
                if first_comma_idx == -1:
                    continue
                
                directory_name = line[:first_comma_idx]
                
                # Extract the list of targets - find the part between [ and ]
                start_idx = line.find('[')
                end_idx = line.find(']')
                
                if start_idx == -1 or end_idx == -1:
                    continue
                
                # Parse the list of targets
                targets_str = line[start_idx:end_idx+1]
                try:
                    targets = ast.literal_eval(targets_str)
                except:
                    continue
                
                # Check if directory exists
                frames_dir = self.path_to_data / 'frames' / 'test' / directory_name
                if not frames_dir.exists():
                    continue
                
                # Get all image files in the directory
                image_files = sorted(list(frames_dir.glob('frame_*.jpg')))
                if len(image_files) == 0:
                    continue
                
                # Each entry contains: (image_path, label)
                for i, image_file in enumerate(image_files):
                    if i < len(targets):
                        label = targets[i]
                        all_data.append((image_file, label))
        
        if not all_data:
            raise ValueError(f"No valid data found for CarCrash dataset")
        
        # Split data into train/val/test (70:15:15) without random shuffling
        total_samples = len(all_data)
        train_size = int(0.7 * total_samples)
        val_size = int(0.15 * total_samples)
        
        if which_set == 'train':
            selected_data = all_data[:train_size]
        elif which_set == 'val':
            selected_data = all_data[train_size:train_size + val_size]
        elif which_set == 'test':
            selected_data = all_data[train_size + val_size:]
        else:
            raise ValueError(f"Invalid which_set: {which_set}")
        
        if not selected_data:
            raise ValueError(f"No data found for set: {which_set}")
        
        # Separate images and labels
        images = [item[0] for item in selected_data]
        labels = [item[1] for item in selected_data]
        
        return images, torch.tensor(labels)

    
    def load_data(self, dataset_name: str, which_set: str):
        if dataset_name == 'DoTA':
            return self._load_dota(which_set)
        elif dataset_name == 'CarCrash':
            return self._load_carcrash(which_set)
        else:
            raise ValueError(f"Dataset {dataset_name} not found")
