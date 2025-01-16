import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
def tan_fi(data):
    d = data.clone()
    d *= 2
    d -= 1
    return d

def tan_fi(data):
    # Transformation assumes data is in range [0, 1]. If not, normalization is required.
    d = data.clone()
    d *= 2
    d -= 1
    
    return d

def get_training_augmentation():
    train_transform = [
        # Geometric transformations that preserve noise characteristics
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # # Small color adjustments (subtle to not interfere with noise patterns)
        # A.ColorJitter(
        #     brightness=0.1,
        #     contrast=0.1,
        #     saturation=0.1,
        #     hue=0.1,
        #     p=0.3
        # ),
        
        # # Subtle lighting adjustments
        # A.RandomBrightnessContrast(
        #     brightness_limit=0.1,
        #     contrast_limit=0.1,
        #     p=0.3
        # ),
        
        # # Channel shuffling can help with robustness
        # A.ChannelShuffle(p=0.1),
        
    ]
    
    return A.Compose(
        train_transform,
        additional_targets={'image1': 'image'}
    )


class CBSD68Dataset(Dataset):
    def __init__(self, root_dir, noise_level=25, crop_size=256, num_crops=32, normalize=True, tanfi=True):
        self.root_dir = root_dir
        self.noise_level = f"noisy_{noise_level}"
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.normalize = normalize
        self.tanfi = tanfi

        self.original_dir = os.path.join(root_dir, "original_png")
        self.noisy_dir = os.path.join(root_dir, "CBSD_"+self.noise_level)

        self.image_paths = [fname for fname in os.listdir(self.original_dir) if fname.endswith('.png')]

        self.image_pairs = []
        for img_name in self.image_paths:
            clean_path = os.path.join(self.original_dir, img_name)
            noisy_path = os.path.join(self.noisy_dir, img_name)

            clean_image = Image.open(clean_path).convert("RGB")
            noisy_image = Image.open(noisy_path).convert("RGB")

            clean_np = np.array(clean_image).astype(np.float32)
            noisy_np = np.array(noisy_image).astype(np.float32)

            if self.normalize:
                clean_np /= 255.0
                noisy_np /= 255.0

            h, w, _ = clean_np.shape

            for _ in range(self.num_crops):
                top = random.randint(0, h - self.crop_size)
                left = random.randint(0, w - self.crop_size)

                clean_crop = clean_np[top:top + self.crop_size, left:left + self.crop_size]
                noisy_crop = noisy_np[top:top + self.crop_size, left:left + self.crop_size]

                clean_crop = torch.from_numpy(clean_crop).permute(2, 0, 1)
                noisy_crop = torch.from_numpy(noisy_crop).permute(2, 0, 1)

                if self.tanfi:
                    clean_crop = tan_fi(clean_crop)

                self.image_pairs.append((noisy_crop, clean_crop))

    def __len__(self):
        # return 50
        return len(self.image_pairs)

    def __getitem__(self, idx):
        # return 4
        noisy, clean = self.image_pairs[idx]
        return noisy, clean


    def visualize(self, idx):
        import matplotlib.pyplot as plt

        noisy_crop, clean_crop = self.image_pairs[idx]

        noisy_image = noisy_crop.permute(1, 2, 0).numpy()
        clean_image = clean_crop.permute(1, 2, 0).numpy()

        if self.normalize:
            noisy_image = (noisy_image * 255).astype(np.uint8)
            clean_image = (clean_image * 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(clean_image)
        axes[0].set_title("Clean Crop")
        axes[0].axis("off")

        axes[1].imshow(noisy_image)
        axes[1].set_title("Noisy Crop")
        axes[1].axis("off")

        plt.show()
        
class BSD400(Dataset):
    def __init__(self, root_dir, noise_level=25, crop_size=256, num_crops=32, normalize=True, tanfi=True):
        self.root_dir = root_dir
        self.noise_level = f"noisy{noise_level}"
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.normalize = normalize
        self.tanfi = tanfi

        self.original_dir = os.path.join(root_dir, "BSD400_noisy_0")
        self.noisy_dir = os.path.join(root_dir, "BSD_400_"+self.noise_level)

        self.image_paths = [fname for fname in os.listdir(self.original_dir) if fname.endswith('.png')]

        self.image_pairs = []
        for img_name in self.image_paths:
            clean_path = os.path.join(self.original_dir, img_name)
            noisy_path = os.path.join(self.noisy_dir, img_name)

            clean_image = Image.open(clean_path).convert("RGB")
            noisy_image = Image.open(noisy_path).convert("RGB")

            clean_np = np.array(clean_image).astype(np.float32)
            noisy_np = np.array(noisy_image).astype(np.float32)

            if self.normalize:
                clean_np /= 255.0
                noisy_np /= 255.0

            h, w, _ = clean_np.shape

            for _ in range(self.num_crops):
                top = random.randint(0, h - self.crop_size)
                left = random.randint(0, w - self.crop_size)

                clean_crop = clean_np[top:top + self.crop_size, left:left + self.crop_size]
                noisy_crop = noisy_np[top:top + self.crop_size, left:left + self.crop_size]

                clean_crop = torch.from_numpy(clean_crop).permute(2, 0, 1)
                noisy_crop = torch.from_numpy(noisy_crop).permute(2, 0, 1)

                if self.tanfi:
                    clean_crop = tan_fi(clean_crop)

                self.image_pairs.append((noisy_crop, clean_crop))

    def __len__(self):
        # return 30
        return len(self.image_pairs)

    def __getitem__(self, idx):
        # return 4
        noisy, clean = self.image_pairs[idx]
        return noisy, clean


    def visualize(self, idx):
        import matplotlib.pyplot as plt

        noisy_crop, clean_crop = self.image_pairs[idx]

        noisy_image = noisy_crop.permute(1, 2, 0).numpy()
        clean_image = clean_crop.permute(1, 2, 0).numpy()

        if self.normalize:
            noisy_image = (noisy_image * 255).astype(np.uint8)
            clean_image = (clean_image * 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(clean_image)
        axes[0].set_title("Clean Crop")
        axes[0].axis("off")

        axes[1].imshow(noisy_image)
        axes[1].set_title("Noisy Crop")
        axes[1].axis("off")

        plt.show()
        

class Waterloo(Dataset):
    def __init__(self, root_dir, noise_level=25, crop_size=256, num_crops=32, normalize=True, tanfi=True, augment=True):
        self.root_dir = root_dir
        self.noise_level = f"noisy_{noise_level}"
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.normalize = normalize
        self.tanfi = tanfi
        self.augment = augment

        self.original_dir = os.path.join(root_dir, "WaterlooED_noisy_0")
        self.noisy_dir = os.path.join(root_dir, "WaterlooED_"+self.noise_level)

        self.image_paths = [fname for fname in os.listdir(self.original_dir) if fname.endswith('.bmp')]
        
        # Define augmentation pipeline based on augment parameter
        if self.augment:
            self.transform = A.Compose([
                A.OneOf([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=[90, 90], p=0.5),
                    A.Rotate(limit=[180, 180], p=0.5),
                    A.Rotate(limit=[270, 270], p=0.5),
                ], p=1.0),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                ToTensorV2()
            ])

        self.image_pairs = []
        for img_name in self.image_paths:
            clean_path = os.path.join(self.original_dir, img_name)
            noisy_path = os.path.join(self.noisy_dir, img_name)

            clean_image = Image.open(clean_path).convert("RGB")
            noisy_image = Image.open(noisy_path).convert("RGB")

            clean_np = np.array(clean_image).astype(np.float32)
            noisy_np = np.array(noisy_image).astype(np.float32)

            if self.normalize:
                clean_np /= 255.0
                noisy_np /= 255.0

            h, w, _ = clean_np.shape

            for _ in range(self.num_crops):
                top = random.randint(0, h - self.crop_size)
                left = random.randint(0, w - self.crop_size)

                clean_crop = clean_np[top:top + self.crop_size, left:left + self.crop_size]
                noisy_crop = noisy_np[top:top + self.crop_size, left:left + self.crop_size]

                # Apply same augmentation to both clean and noisy images
                seed = random.randint(0, 2**32 - 1)
                
                random.seed(seed)
                transformed_clean = self.transform(image=clean_crop)['image']
                
                random.seed(seed)
                transformed_noisy = self.transform(image=noisy_crop)['image']

                if self.tanfi:
                    transformed_clean = tan_fi(transformed_clean)

                self.image_pairs.append((transformed_noisy, transformed_clean))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        noisy, clean = self.image_pairs[idx]
        return noisy, clean

    def visualize(self, idx):
        import matplotlib.pyplot as plt

        noisy_crop, clean_crop = self.image_pairs[idx]
        
        # Convert back to numpy for visualization
        noisy_image = noisy_crop.permute(1, 2, 0).numpy()
        clean_image = clean_crop.permute(1, 2, 0).numpy()

        if self.normalize:
            noisy_image = (noisy_image * 255).astype(np.uint8)
            clean_image = (clean_image * 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(clean_image)
        axes[0].set_title("Clean Crop")
        axes[0].axis("off")

        axes[1].imshow(noisy_image)
        axes[1].set_title("Noisy Crop")
        axes[1].axis("off")

        plt.show()


class Waterloo(Dataset):
    def __init__(self, root_dir, noise_level=25, crop_size=256, num_crops=32, normalize=True, tanfi=True, augment=True):
        self.root_dir = root_dir
        self.noise_level = f"noisy_{noise_level}"
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.normalize = normalize
        self.tanfi = tanfi
        self.augment = augment

        self.original_dir = os.path.join(root_dir, "WaterlooED_noisy_0")
        self.noisy_dir = os.path.join(root_dir, "WaterlooED_"+self.noise_level)

        self.image_paths = [fname for fname in os.listdir(self.original_dir) if fname.endswith('.bmp')]
        
        # Define augmentation pipeline based on augment parameter
        if self.augment:
            self.transform = A.Compose([
                A.OneOf([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=[90, 90], p=0.5),
                    A.Rotate(limit=[180, 180], p=0.5),
                    A.Rotate(limit=[270, 270], p=0.5),
                ], p=1.0),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                ToTensorV2()
            ])

        self.image_pairs = []
        for img_name in self.image_paths:
            clean_path = os.path.join(self.original_dir, img_name)
            noisy_path = os.path.join(self.noisy_dir, img_name)

            clean_image = Image.open(clean_path).convert("RGB")
            noisy_image = Image.open(noisy_path).convert("RGB")

            clean_np = np.array(clean_image).astype(np.float32)
            noisy_np = np.array(noisy_image).astype(np.float32)

            if self.normalize:
                clean_np /= 255.0
                noisy_np /= 255.0

            h, w, _ = clean_np.shape

            for _ in range(self.num_crops):
                top = random.randint(0, h - self.crop_size)
                left = random.randint(0, w - self.crop_size)

                clean_crop = clean_np[top:top + self.crop_size, left:left + self.crop_size]
                noisy_crop = noisy_np[top:top + self.crop_size, left:left + self.crop_size]

                # Apply same augmentation to both clean and noisy images
                seed = 42
                
                random.seed(seed)
                transformed_clean = self.transform(image=clean_crop)['image']
                
                random.seed(seed)
                transformed_noisy = self.transform(image=noisy_crop)['image']

                if self.tanfi:
                    transformed_clean = tan_fi(transformed_clean)

                self.image_pairs.append((transformed_noisy, transformed_clean))

    def __len__(self):
        # return 100
        return len(self.image_pairs)

    def __getitem__(self, idx):
        noisy, clean = self.image_pairs[idx]
        return noisy, clean

    def visualize(self, idx):
        import matplotlib.pyplot as plt

        noisy_crop, clean_crop = self.image_pairs[idx]
        
        # Convert back to numpy for visualization
        noisy_image = noisy_crop.permute(1, 2, 0).numpy()
        clean_image = clean_crop.permute(1, 2, 0).numpy()

        if self.normalize:
            noisy_image = (noisy_image * 255).astype(np.uint8)
            clean_image = (clean_image * 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(clean_image)
        axes[0].set_title("Clean Crop")
        axes[0].axis("off")

        axes[1].imshow(noisy_image)
        axes[1].set_title("Noisy Crop")
        axes[1].axis("off")

        plt.show()