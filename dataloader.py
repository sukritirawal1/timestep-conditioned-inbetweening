from torch.utils.data import Dataset
from torchvision.io import decode_image, ImageReadMode
import os
import math
import torch
from torchvision import transforms
from torchvision.transforms import v2


class AnitaDataset(Dataset):
    def __init__(self, root_dir, between_frames=3, image_shape=(240, 426)):
        self.root_dir = root_dir
        self.between_frames = between_frames
        self.step = max(1, math.ceil(between_frames / 2))

        self.frame_sets = self._get_frame_sets()
        self.transform = transforms.Compose(
            [
                # transforms.v2.ToDtype(torch.float32, scale=True),
                transforms.Resize(image_shape, antialias=True),
            ]
        )

    def _get_frame_sets(self):
        frame_sets = []
        for animation_dir in os.listdir(self.root_dir):
            animation_dir_path = os.path.join(self.root_dir, animation_dir)
            if not os.path.isdir(animation_dir_path):
                continue
            for type_dir in os.listdir(animation_dir_path):
                type_dir_path = os.path.join(animation_dir_path, type_dir)
                if not os.path.isdir(type_dir_path):
                    continue
                for scene_dir in os.listdir(type_dir_path):
                    scene_dir_path = os.path.join(type_dir_path, scene_dir)
                    if not os.path.isdir(scene_dir_path):
                        continue
                    frame_paths = os.listdir(scene_dir_path)
                    frame_paths.sort()
                    frame_paths = [
                        os.path.join(scene_dir_path, frame_path)
                        for frame_path in frame_paths
                    ]
                    # group frame paths by between_frames with overlap

                    for i in range(0, len(frame_paths), self.step):
                        if i + self.between_frames + 2 < len(frame_paths):
                            frame_set = tuple(
                                frame_paths[i : i + self.between_frames + 2]
                            )
                            frame_sets.append(frame_set)
        return frame_sets

    def __len__(self):
        return len(self.frame_sets)

    def __getitem__(self, idx):
        frame_set = self.frame_sets[idx]
        images = [
            decode_image(frame_path, mode=ImageReadMode.RGB) for frame_path in frame_set
        ]
        images = [
            v2.functional.to_dtype(image, torch.float32, scale=True) for image in images
        ]
        images = [self.transform(image) for image in images]
        images = [image.clamp(0.0, 1.0) for image in images]
        return {
            "anchor_start": images[0],
            "anchor_end": images[-1],
            "targets": torch.stack(images[1:-1]),
        }
