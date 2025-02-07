import os
import torch
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
from torchvision import transforms
from utils.misc import get_metrics
from visualizer import load_models, get_metrics, get_statistics

def un_tan_fi(data):
    d = data.clone()
    d += 1
    d /= 2
    return d

def tan_fi(data):
    d = data.clone()
    d *= 2
    d -= 1
    return d

class RainInferenceDataset(Dataset):
    def __init__(
        self, 
        root_dir: str,
        dataset_name: str,
        normalize: bool = True,
        tanfi: bool = True
    ):
        self.root_dir = Path(root_dir) / dataset_name
        self.normalize = normalize
        self.tanfi = tanfi
        
        self.input_dir = self.root_dir / "input"
        self.target_dir = self.root_dir / "target"
        
        if not (self.input_dir.exists() and self.target_dir.exists()):
            raise ValueError(f"Invalid directory structure for {dataset_name}")
            
        self.image_pairs = self._get_image_pairs()
        
        # Define transforms for resizing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), antialias=True),
            transforms.ToTensor()
        ])
        
    def _get_image_pairs(self) -> List[Tuple[str, str]]:
        input_images = sorted(self.input_dir.glob("*.jpg"))
        pairs = []
        
        for input_path in input_images:
            target_path = self.target_dir / input_path.name
            if target_path.exists():
                pairs.append((str(input_path), str(target_path)))
            else:
                print(f"Warning: No matching target found for {input_path.name}")
                
        return pairs
    
    def __len__(self) -> int:
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        input_path, target_path = self.image_pairs[idx]
        
        # Load images
        input_img = Image.open(input_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")
        
        # Apply resize transform
        input_tensor = self.transform(input_img)
        target_tensor = self.transform(target_img)
        
        # Handle normalization if needed (transform already scales to [0,1])
        if not self.normalize:
            input_tensor *= 255.0
            target_tensor *= 255.0
            
        # Apply tanfi transform if requested
        if self.tanfi:
            target_tensor = tan_fi(target_tensor)
            
        return input_tensor, target_tensor, os.path.basename(input_path)

def evaluate_model(model, dataloader, device):
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)
    
    total_psnr = 0
    total_ssim = 0
    num_batches = 0
    
    model.eval()
    
    with tqdm(dataloader, desc="Testing Progress") as loader:
        for input_batch, target_batch, filenames in loader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            with torch.no_grad():
                clean_ref = un_tan_fi(target_batch)
                output, _, _ = model(input_batch, clean_ref)
                
                psnr = psnr_metric(output, target_batch)
                ssim = ssim_metric(output, target_batch)
                
                total_psnr += psnr.item()
                total_ssim += ssim.item()
                num_batches += 1
                
                loader.set_postfix(psnr=psnr.item(), ssim=ssim.item())
    
    return total_psnr / num_batches, total_ssim / num_batches

def run_full_evaluation(
    model,
    root_dir: str,
    test_datasets: List[str] = ['Rain100H', 'Rain100L', 'Test100', 'Test1200', 'Test2800'],
    batch_size: int = 1,
    num_workers: int = 4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    results = {}
    
    for dataset_name in test_datasets:
        try:
            dataset = RainInferenceDataset(
                root_dir=root_dir,
                dataset_name=dataset_name
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                pin_memory=True
            )
            
            print(f"\nEvaluating {dataset_name} ({len(dataset)} images)")
            avg_psnr, avg_ssim = evaluate_model(model, dataloader, device)
            
            results[dataset_name] = {
                'psnr': avg_psnr,
                'ssim': avg_ssim
            }
            
            print(f"{dataset_name} Results:")
            print(f"Average PSNR: {avg_psnr:.4f}")
            print(f"Average SSIM: {avg_ssim:.4f}")
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            continue
            
    return results

# Example usage:
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load your model
    model = load_models('/kaggle/input/rain/pytorch/default/1/best_model.pth', device)
    model.to(device).eval()
    model.bypass = True
    
    # Run evaluation
    root_dir = "/kaggle/input/rain13k/test/test"
    results = run_full_evaluation(model, root_dir, device=device)