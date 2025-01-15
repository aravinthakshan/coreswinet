import torch
import torch.nn as nn
import argparse
from utils.model.plsworkmodel import Model
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from utils.dataloader import CBSD68Dataset
from torch.utils.data import DataLoader, Subset
import torchmetrics
from tqdm import tqdm
from utils.misc import get_metrics, replace_activations_with_relu
from visualizer import load_models
import wandb 

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

def test(
    batch_size,
    test_dir,
    wandb_debug,
    device='cuda',
    use_wandb = True
      # New parameter to control when to enable bypass
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if use_wandb:
        wandb.init(project="image-denoising", config={
            "noise_level": noise_level,
            "crop_size": crop_size,
            "num_crops": num_crops
        })
    
    main_model, n2n_model = load_models(
    './main_model/best_model.pth', 
    './n2n_model/best_model_n2n.pth', 
    device
)

    main_model.to(device).eval()
    n2n_model.to(device).eval()
    
    main_model.bypass = True 
    print("Main Model Bypass ! ")
    
    dataset = CBSD68Dataset(
        root_dir=val_dir, 
        noise_level=noise_level, 
        crop_size=crop_size, 
        num_crops=num_crops,
        normalize=True,
        tanfi=True 
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)
    
    selected_indices = [10, 20]  
    
    all_stats = []
    for i, (noise, clean) in enumerate(dataloader):
        if i not in selected_indices:
            continue
        
        noise, clean = noise.to(device), clean.to(device)
        
        with torch.no_grad():
        #    output_n2n = n2n_model.denoise(noise)
            output_n2n = un_tan_fi(clean)
            output_main, _, _ = main_model(noise, output_n2n)
        
        # Get metrics for both models
        psnr_main, ssim_main = get_metrics(clean, output_main, psnr_metric, ssim_metric)
        # psnr_n2n, ssim_n2n = get_metrics(clean, output_n2n, psnr_metric, ssim_metric,n2n=True)
        
        print(f"\nImage {i}:")
        print(f"Main Model - PSNR: {psnr_main:.4f}, SSIM: {ssim_main:.4f}")
        # print(f"N2N Model  - PSNR: {psnr_n2n:.4f}, SSIM: {ssim_n2n:.4f}")
        
        # Get statistics for both models with different suffixes
        stats_main = get_statistics(noise[0], clean[0], output_main[0], i, suffix='_main', wb=use_wandb)
        # stats_n2n = get_statistics(noise[0], clean[0], output_n2n[0], i, suffix='_n2n', wb=use_wandb,n2n=True)
        
        if use_wandb:
            wandb.log({
                f"image_{i}/main_psnr": psnr_main,
                f"image_{i}/main_ssim": ssim_main,
                # f"image_{i}/n2n_psnr": psnr_n2n,
                # f"image_{i}/n2n_ssim": ssim_n2n
            })
        
        all_stats.append({
            "main_model": stats_main,
            # "n2n_model": stats_n2n
        })
    
    if use_wandb:
        wandb.finish()

    
def test_model(config):
    test(
        config['batch_size'],
        config['test_dir'],
        config['wandb'],
        config['device'],
        config['lr'],
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
