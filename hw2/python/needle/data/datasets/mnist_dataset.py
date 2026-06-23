from typing import List, Optional
from ..data_basic import Dataset
import gzip
import struct
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)

        with gzip.open(label_filename, 'rb') as f:
            magic, n = struct.unpack('>II', f.read(8))
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)

        with gzip.open(image_filename, 'rb') as f:
            magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
            self.images = np.frombuffer(f.read(), dtype=np.uint8)
            self.images = self.images.reshape(n, rows * cols).astype(np.float32) / 255.0

        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        images = self.images[index]
        labels = self.labels[index]

        if isinstance(index, (int, np.integer)):
            images = self.apply_transforms(images)
        else:
            images = np.array([self.apply_transforms(image) for image in images])

        return images, labels
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.labels)
        ### END YOUR SOLUTION
