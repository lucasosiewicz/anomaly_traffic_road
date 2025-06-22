import json
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image


class VideoDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        sequence_length: int = 16,
        stride: int = 4,
        transform = None,
        target_transform = None
    ):
        """
        Inicjalizacja datasetu do obsługi sekwencji wideo.
        
        Args:
            root_dir (str): Ścieżka do głównego katalogu datasetu
            split (str): Zbiór danych ('train', 'val', 'test')
            sequence_length (int): Długość sekwencji klatek
            stride (int): Krok przesuwania okna sekwencji
            transform: Transformacje do zastosowania na klatkach
            target_transform: Transformacje do zastosowania na etykietach
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        self.target_transform = target_transform

        if self.transform is None:
            self.transform = T.ToTensor()

        # Cache na ostatnio wczytane wideo
        self._cache = {
            'name': None,
            'frames': None,
            'labels': None
        }

        # Pobieranie listy sekwencji z folderu frames
        frames_dir = self.root_dir / 'frames' / split
        self.sequences = [seq.name for seq in frames_dir.iterdir() if seq.is_dir()]
        
        # Przechowywanie informacji o wszystkich możliwych sekwencjach
        self.all_sequences = []
        for seq_name in self.sequences:
            # Sprawdzamy tylko liczbę klatek bez ich wczytywania
            frames_dir = self.root_dir / 'frames' / self.split / seq_name / 'images'
            num_frames = len(list(frames_dir.glob('*.jpg')))
            
            num_sequences = (num_frames - self.sequence_length) // self.stride + 1
            for seq_idx in range(num_sequences):
                self.all_sequences.append((seq_name, seq_idx * self.stride))

    def __len__(self) -> int:
        """Zwraca całkowitą liczbę możliwych sekwencji."""
        return len(self.all_sequences)
    
    def load_sequence(self, sequence_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Wczytuje sekwencję klatek i jej etykiety.
        
        Args:
            sequence_name (str): Nazwa sekwencji
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sekwencja klatek i etykiety
        """
        # Sprawdzenie czy sekwencja jest w cache
        if self._cache['name'] == sequence_name:
            return self._cache['frames'], self._cache['labels']
            
        frames_dir = self.root_dir / 'frames' / self.split / sequence_name / 'images'
        annotations_dir = self.root_dir / 'annotations' / self.split / f'{sequence_name}.json'
        
        # Wczytanie klatek
        frame_paths = sorted(list(frames_dir.glob('*.jpg')))
        if not frame_paths:
            raise ValueError(f"Nie znaleziono klatek w katalogu {frames_dir}")
            
        frames = []
        for frame_path in frame_paths:
            frame = Image.open(frame_path)
            if frame is None:
                raise ValueError(f"Nie udało się wczytać klatki {frame_path}")
            frames.append(frame)
            
        # Wczytanie adnotacji
        with open(annotations_dir, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
            start = annotations['anomaly_start']
            end = annotations['anomaly_end']
            
        labels = torch.zeros(len(frames))
        labels[start:end] = 1
        
        # Aktualizacja cache
        self._cache['name'] = sequence_name
        self._cache['frames'] = frames
        self._cache['labels'] = labels
          
        return frames, labels

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Zwraca konkretną sekwencję na podstawie indeksu.
        
        Args:
            idx (int): Indeks sekwencji
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sekwencja klatek i etykieta z ostatniej klatki
        """
        seq_name, start_idx = self.all_sequences[idx]
        frames, labels = self.load_sequence(seq_name)
        
        sequence_frames = frames[start_idx:start_idx + self.sequence_length]
        
        if self.transform:
            sequence = torch.stack([self.transform(frame) for frame in sequence_frames])
        else:
            sequence = torch.stack(sequence_frames)

        # Bierzemy tylko etykietę z ostatniej klatki
        sequence_label = labels[start_idx + self.sequence_length - 1]
        
        if self.target_transform:
            sequence_label = self.target_transform(sequence_label)
        
        return sequence, sequence_label 
