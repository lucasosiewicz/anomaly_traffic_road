import ast
import cv2
import json
import torch
from pathlib import Path
from kornia.geometry import resize
from torch.utils.data import Dataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UnsupervisedDataset(Dataset):
    def __init__(self, path_to_data: str, which_set: str, resize_target: tuple = (224, 224), crop_type: str = None, color_space: str = 'gray', dataset_name: str = 'DoTA', ego_involved = None):

        assert which_set in ['train', 'val'], f"which_set must be one of ['train', 'val'], got {which_set}"

        self.path_to_data = Path(path_to_data)
        self.resize_target = resize_target
        self.crop_type = crop_type
        self.which_set = which_set
        self.dataset_name = dataset_name
        self.ego_involved = ego_involved
        self.color_space = color_space
        self.images, self.labels = self.load_data(dataset_name, which_set)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.color_space == 'gray':
            img = torch.tensor(cv2.imread(str(self.images[idx]), cv2.IMREAD_GRAYSCALE)) # type: ignore
        elif self.color_space == 'rgb':
            img = torch.tensor(cv2.cvtColor(cv2.imread(str(self.images[idx])), cv2.COLOR_BGR2RGB)) # type: ignore
        else:
            raise ValueError(f"Invalid color space: {self.color_space}")
        
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
            temp_images_for_file = []
            temp_targets_for_file = []

            with open(label_path, 'r', encoding='utf-8') as f:
                try:
                    labels_data = json.load(f)
                    # None - accept all
                    # False - accept only non-ego involve
                    # True - accept only ego involve
                    if labels_data['ignore'] or (self.ego_involved is not None and self.ego_involved != labels_data['ego_involve']):
                        continue
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from file {label_path}. Skipping file.")
                    continue

                if "labels" not in labels_data:
                    print(f"Warning: Key 'labels' not found in file {label_path}. Skipping file.")
                    continue
                
                labels = labels_data["labels"]

                try:
                    labels.sort(key=lambda x: x['image_path'])
                except KeyError:
                    print(f"Warning: Key 'image_path' not found in one of the labels in file {label_path}. Skipping file.")
                    continue
                except Exception as e:
                     print(f"Error while sorting labels in file {label_path}: {e}. Skipping file.")
                     continue

                seen_anomaly_in_file = False

                for label in labels:
                    original_accident_id = label.get('accident_id', -1)
                    image_path_str = label.get('image_path')

                    if image_path_str is None:
                        print(f"Warning: Key 'image_path' not found in label in file {label_path}. Skipping label.")
                        continue

                    image_path = self.path_to_data / "frames" / which_set / image_path_str.lstrip("frames/")

                    if not image_path.exists():
                        continue
                    
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

                    if mapped_id != 0:
                        break 
                    else:
                        temp_images_for_file.append(image_path)
                        temp_targets_for_file.append(0)

            images.extend(temp_images_for_file)
            targets.extend(temp_targets_for_file)

        if not images:
            raise ValueError(f"No valid image files found in the set {which_set}")

        return images, torch.tensor(targets)
    
    def load_carcrash(self, which_set: str):
        if which_set == 'train':
            # For train and val sets, get all available frames with label 0
            frames_dir = self.path_to_data / 'frames' / which_set
            images = []
            targets = []
            
            # Iterate through all video directories
            for video_dir in frames_dir.iterdir():
                if video_dir.is_dir():
                    # Get all frame files in the video directory (50 frames per video)
                    for i in range(50):  # Each video has frames 000000 to 000049
                        frame_file = video_dir / f'frame_{i:06d}.jpg'
                        if frame_file.exists():
                            images.append(frame_file)
                            targets.append(0)  # All frames in train/val have label 0
                        
        elif which_set == 'val':
            # For test set, load labels from labels.txt file
            labels_file = self.path_to_data / 'labels.txt'
            frames_dir = self.path_to_data / 'frames' / which_set
            
            images = []
            targets = []
            
            print(f"DEBUG: Loading CarCrash test set from {labels_file}")
            
            # Parse labels.txt to get video IDs and their corresponding labels
            video_labels = {}
            successful_parses = 0
            
            with open(labels_file, 'r') as f:
                for line in f:
                    if line.strip():
                        # Split only on the first comma to separate video_id from the rest
                        comma_idx = line.find(',')
                        if comma_idx == -1:
                            continue
                            
                        video_id = line[:comma_idx].strip()
                        rest_of_line = line[comma_idx+1:].strip()
                        
                        # Find the labels list (starts with [ and ends with ])
                        start_bracket = rest_of_line.find('[')
                        end_bracket = rest_of_line.find(']')
                        
                        if start_bracket != -1 and end_bracket != -1 and end_bracket > start_bracket:
                            labels_str = rest_of_line[start_bracket:end_bracket+1]
                            try:
                                frame_labels = ast.literal_eval(labels_str)
                                video_labels[video_id] = frame_labels
                                successful_parses += 1
                            except (ValueError, SyntaxError):
                                continue
            
            print(f"DEBUG: Successfully parsed {successful_parses} video labels")
            
            # Load frames from test directory that have corresponding labels
            frames_added = 0
            for video_dir in frames_dir.iterdir():
                if video_dir.is_dir():
                    video_id = video_dir.name
                    if video_id in video_labels:
                        frame_labels = video_labels[video_id]
                        # Each video should have 50 frames matching the labels
                        num_frames = min(50, len(frame_labels))
                        for i in range(num_frames):
                            frame_file = video_dir / f'frame_{i:06d}.jpg'
                            if frame_file.exists():
                                images.append(frame_file)
                                targets.append(frame_labels[i])
                                frames_added += 1
            
            print(f"DEBUG: Added {frames_added} frames from test set")
        
        if len(images) == 0:
            raise ValueError(f"No valid image files found in the set {which_set}")
        
        return images, torch.tensor(targets)

    def load_data(self, dataset_name: str, which_set: str):
        if dataset_name == 'DoTA':
            return self._load_dota(which_set)
        elif dataset_name == 'CarCrash':
            return self.load_carcrash(which_set)
        else:
            raise ValueError(f"Dataset {dataset_name} not found")
