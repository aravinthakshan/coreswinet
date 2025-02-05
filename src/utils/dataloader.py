import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
import albumentations as albu
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as BaseDataset
import cv2
import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm as tqdm
from torch.utils.data import DataLoader, Subset
import random
from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data import Dataset as BaseDataset
import cv2
import os
import numpy as np
import torch
import pandas as pd

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
    def __init__(self, root_dir, noise_level=25, crop_size=256, num_crops=32, normalize=True, tanfi=True,augmentation=None):
        self.root_dir = root_dir
        self.noise_level = f"noisy_{noise_level}"
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.normalize = normalize
        self.tanfi = tanfi

        self.original_dir = os.path.join(root_dir, "BSD400_noisy_0")
        self.noisy_dir = os.path.join(root_dir, "BSD400_"+self.noise_level)

        self.image_paths = [fname for fname in os.listdir(self.original_dir) if fname.endswith('.jpg')]

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
    def __init__(self, root_dir, noise_level=25, crop_size=256, num_crops=32, normalize=True, tanfi=True, augmentation=None):
        self.root_dir = root_dir
        self.noise_level = f"noisy_{noise_level}"
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.normalize = normalize
        self.tanfi = tanfi
        self.augmentation = augmentation

        self.original_dir = os.path.join(root_dir, "WaterlooED_noisy_0")
        self.noisy_dir = os.path.join(root_dir, "WaterlooED_"+self.noise_level)

        self.image_paths = [fname for fname in os.listdir(self.original_dir) if fname.endswith('.bmp')]

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

                # Apply augmentations if specified
                if self.augmentation:
                    augmented = self.augmentation(image=noisy_crop, image1=clean_crop)
                    noisy_crop, clean_crop = augmented['image'], augmented['image1']

                clean_crop = torch.from_numpy(clean_crop).permute(2, 0, 1)
                noisy_crop = torch.from_numpy(noisy_crop).permute(2, 0, 1)

                if self.tanfi:
                    clean_crop = tan_fi(clean_crop)

                self.image_pairs.append((noisy_crop, clean_crop))
                
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

# Define the augmentation functions
def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.Rotate(limit=[90, 90], p=0.5),
        albu.Rotate(limit=[270, 270], p=0.5)
    ]
    return albu.Compose(train_transform, additional_targets={'image1': 'image'})
class DIV2K(Dataset):
    def __init__(self, root_dir, noise_level=25, crop_size=256, num_crops=32, normalize=True, tanfi=True, augmentation=None):
        self.root_dir = root_dir
        self.noise_level = f"{noise_level}"
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.normalize = normalize
        self.tanfi = tanfi
        self.augmentation = augmentation

        self.original_dir = os.path.join(root_dir, "DIV2K_noisy_0")
        self.noisy_dir = os.path.join(root_dir, "DIV2K_noisy_" + self.noise_level)

        # Store only paths instead of loading all images
        self.image_paths = [fname for fname in os.listdir(self.original_dir) if fname.endswith('.png')]

    def __len__(self):
        return len(self.image_paths) * self.num_crops

    def __getitem__(self, idx):
        # Determine which image and crop to load
        img_idx = idx // self.num_crops
        crop_idx = idx % self.num_crops

        img_name = self.image_paths[img_idx]
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
        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)

        clean_crop = clean_np[top:top + self.crop_size, left:left + self.crop_size]
        noisy_crop = noisy_np[top:top + self.crop_size, left:left + self.crop_size]

        if self.augmentation:
            augmented = self.augmentation(image=noisy_crop, image1=clean_crop)
            noisy_crop, clean_crop = augmented['image'], augmented['image1']

        clean_crop = torch.from_numpy(clean_crop).permute(2, 0, 1)
        noisy_crop = torch.from_numpy(noisy_crop).permute(2, 0, 1)

        if self.tanfi:
            clean_crop = tan_fi(clean_crop)

        return noisy_crop, clean_crop
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy.io


class SIDD(Dataset):
    def __init__(self, data_dir, transform=None, size=256, normalize=True, standardize=False, mode="train",tanfi=True):
        self.mode = mode
        self.transform = transform
        self.size = size
        self.normalize = normalize
        self.standardize = standardize
        self.tanfi=tanfi

        if mode == "train":
            self.data_dir = data_dir
            self.image_pairs = []
            for root, _, files in os.walk(data_dir):
                gt_file = next((f for f in files if "GT_SRGB" in f), None)
                noisy_file = next((f for f in files if "NOISY_SRGB" in f), None)
                if gt_file and noisy_file:
                    self.image_pairs.append((
                        os.path.join(root, noisy_file),
                        os.path.join(root, gt_file)
                    ))
        
        elif mode == "val":
            self.noisy_data = scipy.io.loadmat(os.path.join(data_dir, 'ValidationNoisyBlocksSrgb.mat'))['ValidationNoisyBlocksSrgb']
            self.gt_data = scipy.io.loadmat(os.path.join(data_dir, 'ValidationGtBlocksSrgb.mat'))['ValidationGtBlocksSrgb']
        
        elif mode == "test":
            self.data_dir = data_dir
            file_path = os.path.join(data_dir, f'SIDD_Benchmark_Code_v1.2/SIDD_Benchmark_Code_v1.2/BenchmarkBlocks32.mat')
            data = scipy.io.loadmat(file_path)
            
            if 'BenchmarkBlocks32' in data:
                self.crop_coordinates = data['BenchmarkBlocks32']
            else:
                raise ValueError("Dataset key 'BenchmarkBlocks32' not found in the .mat file.")

            self.image_pairs = []
            benchmark_data_path = os.path.join(data_dir, "SIDD_Benchmark_Data", "SIDD_Benchmark_Data")
            if not os.path.exists(benchmark_data_path):
                raise ValueError(f"SIDD_Benchmark_Data directory not found in {data_dir}")
            
            for scene_folder in os.listdir(benchmark_data_path):
                scene_path = os.path.join(benchmark_data_path, scene_folder)
                if os.path.isdir(scene_path):
                    scene_folder = scene_folder.split("_")
                    noisy_path = os.path.join(scene_path, f"{scene_folder[0]}_NOISY_SRGB_010.PNG")
                    if os.path.exists(noisy_path):
                        self.image_pairs.append(noisy_path)
            
            if len(self.image_pairs) == 0:
                raise ValueError(f"No valid image files found in {data_dir}")
        
        else:
            raise ValueError("Mode must be 'train', 'val', or 'test'")

    def __len__(self):
        if self.mode == "train":
            return len(self.image_pairs)
        elif self.mode == "val":
            return self.noisy_data.shape[0] * self.noisy_data.shape[1]
        elif self.mode == "test":
            return len(self.image_pairs) * 32  # 32 crops per image

    def __getitem__(self, idx):
        if self.mode == "train":
            noisy_path, gt_path = self.image_pairs[idx]
            noisy = cv2.imread(noisy_path)
            noiseless = cv2.imread(gt_path)
            noisy_cropped, noiseless_cropped = self.get_cropped_images(noisy, noiseless, self.size)
            noisy_img, noiseless_img = self.rotate_images(noisy_cropped, noiseless_cropped)
            
        elif self.mode == "val":
            block_idx = idx // self.noisy_data.shape[1]
            patch_idx = idx % self.noisy_data.shape[1]
            noisy_img = self.noisy_data[block_idx, patch_idx]
            noiseless_img = self.gt_data[block_idx, patch_idx]
        
        elif self.mode == "test":
            image_idx = idx // 32
            crop_idx = idx % 32
            noisy_path = self.image_pairs[image_idx]
            noisy = cv2.imread(noisy_path)
            noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
            crop_coords = self.crop_coordinates[crop_idx]
            noisy_img = self.crop_image(noisy, crop_coords)
            noiseless_img = None  # No ground truth for test mode

        # Convert to torch tensor and change dimension order from HWC to CHW
        noisy_img = torch.tensor(noisy_img, dtype=torch.float32).permute(2, 0, 1)
        if noiseless_img is not None:
            noiseless_img = torch.tensor(noiseless_img, dtype=torch.float32).permute(2, 0, 1)
        
        if self.normalize:
            noisy_img = self.normalize_image(noisy_img)
            if noiseless_img is not None:
                noiseless_img = self.normalize_image(noiseless_img)
        if self.tanfi:
            noiseless_img=tan_fi(noiseless_img)
        
        if self.standardize:
            noisy_img = self.standardize_image(noisy_img)
            if noiseless_img is not None:
                noiseless_img = self.standardize_image(noiseless_img)
        return noisy_img, noiseless_img

    def get_cropped_images(self, img1, img2, size=256):
        #img1 = img1.astype(np.uint8)
        #img2 = img2.astype(np.uint8)

        # Apply Albumentations
        augment = albu.Compose([
            albu.RandomCrop(width=size, height=size),
        ], additional_targets={'image1': 'image'})
        
        augmented = augment(image=img1, image1=img2)
        img1_cropped = augmented['image']
        img2_cropped = augmented['image1']
        
        return img1_cropped, img2_cropped

    def rotate_images(self, img1, img2):
        angles = [0, 90, 180, 270]
        angle = random.choice(angles)
        img1_rotated = Image.fromarray(img1).rotate(angle, expand=True)
        img2_rotated = Image.fromarray(img2).rotate(angle, expand=True)
        return np.array(img1_rotated), np.array(img2_rotated)

    def normalize_image(self, img):
        return img.float() / 255.0

    def standardize_image(self, img):
        normalize = transforms.Normalize(mean=[0.2400, 0.2518, 0.3112],
                                         std=[0.0892, 0.0785, 0.1011])
        
        standardized_img = normalize(img)
        
        return standardized_img
    
    def crop_image(self, image, crop_coords):
        y, x, h, w = crop_coords
        return image[y:y+h, x:x+w]

class McMasterDataset(Dataset):
    def __init__(self, root_dir, noise_level=25, crop_size=256, num_crops=32, normalize=True, tanfi=True):
        self.root_dir = root_dir
        self.noise_level = f"noisy_{noise_level}"
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.normalize = normalize
        self.tanfi = tanfi

        self.original_dir = os.path.join(root_dir, "McMaster_noisy_0")
        self.noisy_dir = os.path.join(root_dir, "McMaster_"+self.noise_level)

        self.image_paths = [fname for fname in os.listdir(self.original_dir) if fname.endswith('.tif')]

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
# # Usage example
# if __name__ == "__main__":
#     # Paths to your data
#     train_data_dir = "/path/to/train/data"
#     val_data_dir = "/path/to/val/data"
#     test_data_dir = "/path/to/test/data"
    

    
#     # Create datasets
#     train_dataset = SIDD(train_data_dir, mode="train")
#     val_dataset = SIDD(val_data_dir, mode="val")
#     test_dataset = SIDD(test_data_dir, mode="test", crop_coordinates=crop_coords_list)
    
#     # Create dataloaders
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
#     # Visualize individual samples from train dataset
#     visualize_sidd(train_dataset, num_samples=4)
    
#     # Visualize a batch from train dataloader
#     visualize_sidd(train_loader, batch=True)
    
#     # Visualize specific samples from val dataset
#     visualize_sidd(val_dataset, indices=[0, 10, 20, 30])
    
#     # Visualize a batch from val dataloader
#     visualize_sidd(val_loader, batch=True)
    
#     # Visualize a batch from test dataloader
#     visualize_sidd(test_loader, batch=True)
class kodak(Dataset):
    def __init__(self, root_dir, noise_level=25, crop_size=256, num_crops=32, normalize=True, tanfi=True):
        self.root_dir = root_dir
        self.noise_level = f"noisy_{noise_level}"
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.normalize = normalize
        self.tanfi = tanfi

        self.original_dir = os.path.join(root_dir, "kodak_noisy_0")
        self.noisy_dir = os.path.join(root_dir, "kodak_"+self.noise_level)

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


class SICEGradTrain(BaseDataset):
    def __init__(self, root_dir, transform=None, augmentation=None):
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, 'SICE_Grad')
        self.label_dir = os.path.join(root_dir, 'SICE_Reshape')
        self.file_names = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.jpg')]
        self.transform = transform
        self.augmentation = augmentation
        self.file_names = self.file_names[:480]

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input_name = os.path.join(self.input_dir, self.file_names[idx])
        base_name = os.path.splitext(self.file_names[idx])[0]
        
        label_name_jpg_upper = os.path.join(self.label_dir, base_name + '.JPG')
        label_name_jpg_lower = os.path.join(self.label_dir, base_name + '.jpg')
        label_name_png = os.path.join(self.label_dir, base_name + '.PNG')
        
        if os.path.exists(label_name_jpg_upper):
            label_name = label_name_jpg_upper
        elif os.path.exists(label_name_jpg_lower):
            label_name = label_name_jpg_lower
        elif os.path.exists(label_name_png):
            label_name = label_name_png
        else:
            raise FileNotFoundError(f"No corresponding label image found for {input_name}")
        
        #READING IMAGE
        label_image = cv2.imread(label_name)
        input_image = cv2.imread(input_name)
        
        #CHANGING COLORSPACE
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        #ROTATING PORTRAIT IMAGES
        if label_image.shape[0] > label_image.shape[1]:
            label_image = cv2.rotate(label_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            input_image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        #RESIZE + FLIP AUGMENTATION
        if self.augmentation:
            augmented = self.augmentation(image1=label_image, image=input_image)
            label_image, input_image = augmented['image1'], augmented['image']
        
        #NORMALISATION
        input_image = input_image / 255.0
        label_image = label_image / 255.0
        
        #STANDARDISATION
        # if self.transform:
        #     input_image = self.transform(image=input_image)['image']
        
        #CHANGING DIMSPACE
        input_image = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1)
        label_image = torch.tensor(label_image, dtype=torch.float32).permute(2, 0, 1)
        
        return input_image, label_image
    
class SICEGradVal(BaseDataset):
    def __init__(self, root_dir, transform=None, augmentation=None):
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, 'SICE_Grad')
        self.label_dir = os.path.join(root_dir, 'SICE_Reshape')
        self.file_names = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.jpg')]
        self.transform = transform
        self.augmentation = augmentation
        self.file_names = self.file_names[480:530]

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input_name = os.path.join(self.input_dir, self.file_names[idx])
        base_name = os.path.splitext(self.file_names[idx])[0]
        
        label_name_jpg_upper = os.path.join(self.label_dir, base_name + '.JPG')
        label_name_jpg_lower = os.path.join(self.label_dir, base_name + '.jpg')
        label_name_png = os.path.join(self.label_dir, base_name + '.PNG')
        
        if os.path.exists(label_name_jpg_upper):
            label_name = label_name_jpg_upper
        elif os.path.exists(label_name_jpg_lower):
            label_name = label_name_jpg_lower
        elif os.path.exists(label_name_png):
            label_name = label_name_png
        else:
            raise FileNotFoundError(f"No corresponding label image found for {input_name}")
        
        #READING IMAGE
        label_image = cv2.imread(label_name)
        input_image = cv2.imread(input_name)
        
        #CHANGING COLORSPACE
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        #ROTATING PORTRAIT IMAGES
        if label_image.shape[0] > label_image.shape[1]:
            label_image = cv2.rotate(label_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            input_image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        #RESIZE + FLIP AUGMENTATION
        if self.augmentation:
            augmented = self.augmentation(image1=label_image, image=input_image)
            label_image, input_image = augmented['image1'], augmented['image']
        
        #NORMALISATION
        input_image = input_image / 255.0
        label_image = label_image / 255.0
        
        #STANDARDISATION
        # if self.transform:
        #     input_image = self.transform(image=input_image)['image']
        
        #CHANGING DIMSPACE
        input_image = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1)
        label_image = torch.tensor(label_image, dtype=torch.float32).permute(2, 0, 1)
        
        return input_image, label_image
    
class SICEGradTest(BaseDataset):
    def __init__(self, root_dir, transform=None, augmentation=None):
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, 'SICE_Grad')
        self.label_dir = os.path.join(root_dir, 'SICE_Reshape')
        self.file_names = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.jpg')]
        self.transform = transform
        self.augmentation = augmentation
        self.file_names = self.file_names[530:]

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input_name = os.path.join(self.input_dir, self.file_names[idx])
        base_name = os.path.splitext(self.file_names[idx])[0]
        
        label_name_jpg_upper = os.path.join(self.label_dir, base_name + '.JPG')
        label_name_jpg_lower = os.path.join(self.label_dir, base_name + '.jpg')
        label_name_png = os.path.join(self.label_dir, base_name + '.PNG')
        
        if os.path.exists(label_name_jpg_upper):
            label_name = label_name_jpg_upper
        elif os.path.exists(label_name_jpg_lower):
            label_name = label_name_jpg_lower
        elif os.path.exists(label_name_png):
            label_name = label_name_png
        else:
            raise FileNotFoundError(f"No corresponding label image found for {input_name}")
        
        #READING IMAGE
        label_image = cv2.imread(label_name)
        input_image = cv2.imread(input_name)
        
        #CHANGING COLORSPACE
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        #ROTATING PORTRAIT IMAGES
        if label_image.shape[0] > label_image.shape[1]:
            label_image = cv2.rotate(label_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            input_image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        #RESIZE + FLIP AUGMENTATION
        if self.augmentation:
            augmented = self.augmentation(image1=label_image, image=input_image)
            label_image, input_image = augmented['image1'], augmented['image']
        
        #NORMALISATION
        input_image = input_image / 255.0
        label_image = label_image / 255.0
        
        #STANDARDISATION
        # if self.transform:
        #     input_image = self.transform(image=input_image)['image']
        
        #CHANGING DIMSPACE
        input_image = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1)
        label_image = torch.tensor(label_image, dtype=torch.float32).permute(2, 0, 1)
        
        return input_image, label_image
    
class SICEMixTrain(BaseDataset):
    def __init__(self, root_dir, transform=None, augmentation=None):
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, 'SICE_Mix')
        self.label_dir = os.path.join(root_dir, 'SICE_Reshape')
        self.file_names = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.jpg')]
        self.transform = transform
        self.augmentation = augmentation
        self.file_names = self.file_names[:480]

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input_name = os.path.join(self.input_dir, self.file_names[idx])
        base_name = os.path.splitext(self.file_names[idx])[0]
        
        label_name_jpg_upper = os.path.join(self.label_dir, base_name + '.JPG')
        label_name_jpg_lower = os.path.join(self.label_dir, base_name + '.jpg')
        label_name_png = os.path.join(self.label_dir, base_name + '.PNG')
        
        if os.path.exists(label_name_jpg_upper):
            label_name = label_name_jpg_upper
        elif os.path.exists(label_name_jpg_lower):
            label_name = label_name_jpg_lower
        elif os.path.exists(label_name_png):
            label_name = label_name_png
        else:
            raise FileNotFoundError(f"No corresponding label image found for {input_name}")
        
        #READING IMAGE
        label_image = cv2.imread(label_name)
        input_image = cv2.imread(input_name)
        
        #CHANGING COLORSPACE
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        #ROTATING PORTRAIT IMAGES
        if label_image.shape[0] > label_image.shape[1]:
            label_image = cv2.rotate(label_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            input_image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        #RESIZE + FLIP AUGMENTATION
        if self.augmentation:
            augmented = self.augmentation(image1=label_image, image=input_image)
            label_image, input_image = augmented['image1'], augmented['image']
        
        #NORMALISATION
        input_image = input_image / 255.0
        label_image = label_image / 255.0
        
        #STANDARDISATION
        # if self.transform:
        #     input_image = self.transform(image=input_image)['image']
        
        #CHANGING DIMSPACE
        input_image = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1)
        label_image = torch.tensor(label_image, dtype=torch.float32).permute(2, 0, 1)
        
        return input_image, label_image
    
class SICEMixVal(BaseDataset):
    def __init__(self, root_dir, transform=None, augmentation=None):
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, 'SICE_Mix')
        self.label_dir = os.path.join(root_dir, 'SICE_Reshape')
        self.file_names = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.jpg')]
        self.transform = transform
        self.augmentation = augmentation
        self.file_names = self.file_names[480:530]

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input_name = os.path.join(self.input_dir, self.file_names[idx])
        base_name = os.path.splitext(self.file_names[idx])[0]
        
        label_name_jpg_upper = os.path.join(self.label_dir, base_name + '.JPG')
        label_name_jpg_lower = os.path.join(self.label_dir, base_name + '.jpg')
        label_name_png = os.path.join(self.label_dir, base_name + '.PNG')
        
        if os.path.exists(label_name_jpg_upper):
            label_name = label_name_jpg_upper
        elif os.path.exists(label_name_jpg_lower):
            label_name = label_name_jpg_lower
        elif os.path.exists(label_name_png):
            label_name = label_name_png
        else:
            raise FileNotFoundError(f"No corresponding label image found for {input_name}")
        
        #READING IMAGE
        label_image = cv2.imread(label_name)
        input_image = cv2.imread(input_name)
        
        #CHANGING COLORSPACE
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        #ROTATING PORTRAIT IMAGES
        if label_image.shape[0] > label_image.shape[1]:
            label_image = cv2.rotate(label_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            input_image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        #RESIZE + FLIP AUGMENTATION
        if self.augmentation:
            augmented = self.augmentation(image1=label_image, image=input_image)
            label_image, input_image = augmented['image1'], augmented['image']
        
        #NORMALISATION
        input_image = input_image / 255.0
        label_image = label_image / 255.0
        
        #STANDARDISATION
        # if self.transform:
        #     input_image = self.transform(image=input_image)['image']
        
        #CHANGING DIMSPACE
        input_image = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1)
        label_image = torch.tensor(label_image, dtype=torch.float32).permute(2, 0, 1)
        
        return input_image, label_image
    
class SICEMixTest(BaseDataset):
    def __init__(self, root_dir, transform=None, augmentation=None):
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, 'SICE_Mix')
        self.label_dir = os.path.join(root_dir, 'SICE_Reshape')
        self.file_names = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.jpg')]
        self.transform = transform
        self.augmentation = augmentation
        self.file_names = self.file_names[530:]

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input_name = os.path.join(self.input_dir, self.file_names[idx])
        base_name = os.path.splitext(self.file_names[idx])[0]
        
        label_name_jpg_upper = os.path.join(self.label_dir, base_name + '.JPG')
        label_name_jpg_lower = os.path.join(self.label_dir, base_name + '.jpg')
        label_name_png = os.path.join(self.label_dir, base_name + '.PNG')
        
        if os.path.exists(label_name_jpg_upper):
            label_name = label_name_jpg_upper
        elif os.path.exists(label_name_jpg_lower):
            label_name = label_name_jpg_lower
        elif os.path.exists(label_name_png):
            label_name = label_name_png
        else:
            raise FileNotFoundError(f"No corresponding label image found for {input_name}")
        
        #READING IMAGE
        label_image = cv2.imread(label_name)
        input_image = cv2.imread(input_name)
        
        #CHANGING COLORSPACE
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        #ROTATING PORTRAIT IMAGES
        if label_image.shape[0] > label_image.shape[1]:
            label_image = cv2.rotate(label_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            input_image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        #RESIZE + FLIP AUGMENTATION
        if self.augmentation:
            augmented = self.augmentation(image1=label_image, image=input_image)
            label_image, input_image = augmented['image1'], augmented['image']
        
        #NORMALISATION
        input_image = input_image / 255.0
        label_image = label_image / 255.0
        
        #STANDARDISATION
        # if self.transform:
        #     input_image = self.transform(image=input_image)['image']
        
        #CHANGING DIMSPACE
        input_image = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1)
        label_image = torch.tensor(label_image, dtype=torch.float32).permute(2, 0, 1)
        
        return input_image, label_image
    
class LOLTrain(BaseDataset):
    def __init__(self, high_res_folder, low_res_folder, flag,  transform=None, augmentation=None):
        self.image_pairs = list_image_paths(high_res_folder, low_res_folder, flag)
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        high_res_path, low_res_path = self.image_pairs[idx]

        high_res_image = cv2.imread(high_res_path)
        low_res_image = cv2.imread(low_res_path)

        # totensor = ToTensorV2()
        
        if self.augmentation:
            augmented = self.augmentation(image=high_res_image, image1=low_res_image)
            high_res_image, low_res_image = augmented['image'], augmented['image1']

        low_res_image = low_res_image / 255.0
        high_res_image = high_res_image / 255.0

        low_res_image = torch.tensor(low_res_image, dtype=torch.float32).permute(2, 0, 1)
        high_res_image = torch.tensor(high_res_image, dtype=torch.float32).permute(2, 0, 1)
            
        return low_res_image.float(),high_res_image.float()
    
def get_training_augmentation():
    train_transform = [
        albu.Resize(608, 896, interpolation=cv2.INTER_LINEAR, always_apply=True),
        albu.VerticalFlip(p=0.5),
    ]
    return albu.Compose(train_transform, additional_targets={'image1':'image'}, is_check_shapes=False)
    
def get_validation_augmentation():
    test_transform = [
        albu.Resize(608, 896, interpolation=cv2.INTER_LINEAR, always_apply=True),
    ]   
    return albu.Compose(test_transform, additional_targets={'image1': 'image'}, is_check_shapes=False)

def get_transform(dataset):
        if dataset == 'grad':   
            mean = [0.41441402, 0.41269127, 0.37940571]
            std = [0.33492465, 0.33443474, 0.33518072]
        if dataset == 'mix':
            mean = [0.41268688, 0.41124236, 0.37886961]
            std = [0.33789958, 0.33786919, 0.33946865]
        return albu.Compose([
            albu.Normalize(mean=mean, std=std),
    ])