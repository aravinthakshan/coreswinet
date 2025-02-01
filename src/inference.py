import torch
import torch.nn as nn
import argparse
from utils.model.coreswinet import Model
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from utils.dataloader import CBSD68Dataset,McMasterDataset
from torch.utils.data import DataLoader, Subset
import torchmetrics
from tqdm import tqdm
from utils.misc import get_metrics
from visualizer import load_models, get_metrics, get_statistics
import wandb 

def un_tan_fi(data):
    d = data.clone()
    d += 1
    d /= 2
    return d

def test(
    batch_size,
    test_dir,
    use_wandb=True,
    device='cuda',
    noise_level=25,
    test_dataset='CBSD68',
    crop_size=256, 
    num_crops=34,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if use_wandb:
        wandb.init(project="DeFInet", config={
            "noise_level": noise_level,
            "crop_size": crop_size,
            "num_crops": num_crops
        })
    
    main_model = load_models(
        '/kaggle/input/waterlooidk/final_model (7).pth',
        device
    )

    main_model.to(device).eval()
    # n2n_model.to(device).eval()
    
    main_model.bypass = True 
    print("Main Model Bypass!")
    if test_dataset=='CBSD68':
        dataset = CBSD68Dataset(
            root_dir='/kaggle/input/cbsd68/CBSD68', 
            noise_level=25,
            crop_size=256,
            num_crops=34,
            normalize=True,
            tanfi=True 
        )
    elif test_dataset=='mcmaster':
        dataset = McMasterDataset(
            root_dir='/kaggle/input/mcmaster/McMaster', 
            noise_level=25,
            crop_size=256,
            num_crops=34,
            normalize=True,
            tanfi=True 
        )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)
    
    # Reset metrics at start
    psnr_metric.reset()
    ssim_metric.reset()
    
    # Initialize running averages
    total_psnr = 0
    total_ssim = 0
    num_batches = 0
    
    with tqdm(dataloader, desc="Testing Progress") as loader:
        for noise, clean in loader:
            noise, clean = noise.to(device), clean.to(device)
            
            with torch.no_grad():
                output_n2n = un_tan_fi(clean)
                output_main, _, _ = main_model(noise, output_n2n)
            
            # Calculate metrics for main model
            psnr_main, ssim_main = get_metrics(clean, output_main, psnr_metric, ssim_metric)
            
            # Update running averages (per batch, not per image)
            total_psnr += psnr_main
            total_ssim += ssim_main
            num_batches += 1
            
            # Update progress bar
            loader.set_postfix(psnr=psnr_main, ssim=ssim_main)
    
    # Calculate final averages
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    
    print(f"\nFinal Results:")
    print(f"Average PSNR TEST: {avg_psnr:.4f}")
    print(f"Average SSIM TEST: {avg_ssim:.4f}")
    
    if use_wandb:
        wandb.log({
            "avg_psnr_test": avg_psnr,
            "avg_ssim_test": avg_ssim
        })
        wandb.finish()

def test_model(config):
    test(
        config['batch_size'],
        config['test_dir'],
        config['wandb'],
        config['device'], 
        config['noise_level'],
        config['test_dataset']
    )
    
    
# def preprocess_image(img_path, device):
#     # Open the image and convert it to RGB
#     image = Image.open(img_path).convert('RGB')
#     image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
#     image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(device)  # To tensor, add batch dimension
#     image = tan_fi(image)  # Apply tanh-like transformation
#     return image


# def save_output(output_image, output_path):
#     # Remove batch dimension and convert to HxWxC
#     if output_image.ndim == 3 and output_image.shape[0] == 3:
#         output_image = np.transpose(output_image, (1, 2, 0))  # Convert to HxWxC

#     # Convert the output to [0, 255] and save as image
#     output_image = np.clip(output_image, 0, 255).astype(np.uint8)  # Ensure values are in the range [0, 255]
#     plt.imshow(output_image)
#     plt.axis('off')  # Hide axes
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     plt.savefig(output_path, bbox_inches='tight', pad_inches=0)


# def main(args):
#     # Load the model
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     p = torchmetrics.image.PeakSignalNoiseRatio().to(device)
#     z = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)
#     model = Model()
    
#     model.load_state_dict(torch.load(args.weights_path, map_location=device))
#     model.to(device)
#     dataset = CBSD68Dataset(root_dir=args.train_dir, noise_level=25, crop_size=256, num_crops=34, normalize=True)
#     test_loader=DataLoader(dataset,batch_size=1,shuffle=False)
#     model.eval()
#     itr=0
#     with tqdm(test_loader, desc="Testing Progress") as loader:
#         for _, batch_data in enumerate(loader):
#             noise, clean = [x.to(device) for x in batch_data]
#             output = model(noise)
#             psnr_train_itr, ssim_train_itr = get_metrics(clean,output,p,z,Standardize=True)
#             itr+=1
            
#             psnr_train += psnr_train_itr
#             ssim_train += ssim_train_itr
    
#     psnr_train /= (itr + 1)
#     ssim_train /= (itr + 1)

#     # Reverse the tanh-like transformation and scale back to [0, 1]
#     print("Image before untanifying:", output)
#     output = un_tan_fi(output)
#     print("Image after untanifying:", output)

#     # Remove batch dimension and convert to numpy array
#     output = output.squeeze(0).cpu().detach().numpy()

#     # Clip the values to [0, 1] and scale to [0, 255]
#     output = np.clip(output, 0, 1)
#     output = (output * 255).astype(np.uint8)
#     print("Image after un-normalizing:", output)

#     # Save the output image
#     output_folder = "/kaggle/working"
#     output_file = os.path.join(output_folder, "output_image.png")
#     save_output(output, output_file)
#     print(f"Output saved at {output_file}")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights_path', type=str, required=True, help="Path to the model weights")
#     parser.add_argument('--train_dir', type=str, required=True, help="Path to the input image")
#     arguments = parser.parse_args()
#     main(arguments)