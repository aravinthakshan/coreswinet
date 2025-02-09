import torch
import torch.nn as nn
import argparse
from utils.model.coreswinet import Model
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from utils.dataloader import CBSD68Dataset,McMasterDataset,kodak
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

def test_single_configuration(
    main_model,
    dataset_name,
    noise_level,
    batch_size,
    crop_size=256,
    num_crops=34,
    device='cuda',
    use_wandb=True
):
    """
    Run testing for a single dataset and noise level configuration.
    
    Args:
        main_model: The model to test
        dataset_name: Name of the dataset ('CBSD68', 'mcmaster', or 'kodak')
        noise_level: Noise level to test with
        batch_size: Batch size for testing
        crop_size: Size of image crops
        num_crops: Number of crops per image
        device: Computing device
        use_wandb: Whether to log to wandb
    
    Returns:
        tuple: (avg_psnr, avg_ssim) for this configuration
    """
        # Convert numeric parameters to correct types
    noise_level = int(noise_level)
    batch_size = int(batch_size)
    crop_size = int(crop_size)
    # num_crops = int(num_crops)
    # Dataset paths mapping
    dataset_paths = {
        'CBSD68': '/kaggle/input/d/aryamangupta04/cbsd68/CBSD68',
        'mcmaster': '/kaggle/input/mcmaster-proper/McMaster',
        'kodak': '/kaggle/input/kodak-test/kodak_test'
    }
    
    # Dataset class mapping
    dataset_classes = {
        'CBSD68': CBSD68Dataset,
        'mcmaster': McMasterDataset,
        'kodak': kodak
    }
    
    # Initialize dataset
    dataset_class = dataset_classes[dataset_name]
    dataset = dataset_class(
        root_dir=dataset_paths[dataset_name],
        noise_level=noise_level,
        crop_size=256,
        num_crops=34,
        normalize=True,
        tanfi=True
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize metrics
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)
    psnr_metric.reset()
    ssim_metric.reset()
    
    total_psnr = 0
    total_ssim = 0
    num_batches = 0
    
    # Testing loop
    with tqdm(dataloader, desc=f"Testing {dataset_name} - Noise {noise_level}") as loader:
        for noise, clean in loader:
            noise, clean = noise.to(device), clean.to(device)

            # clean = un_tan_fi(clean) ## ---> remove this later
            
            with torch.no_grad():
                output_n2n = un_tan_fi(un_tan_fi(clean))
                output_main, _, _ = main_model(noise, output_n2n)
            
            psnr_main, ssim_main = get_metrics(clean, output_main, psnr_metric, ssim_metric)
            
            total_psnr += psnr_main
            total_ssim += ssim_main
            num_batches += 1
            
            loader.set_postfix(psnr=psnr_main, ssim=ssim_main)
    
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    
    return avg_psnr, avg_ssim

def test(
    batch_size=16,
    test_dir='/kaggle/input/cbsd68/CBSD68',
    use_wandb=True,
    device='cuda',
    crop_size=256,
    num_crops=34,
):
    """
    Main testing function that runs tests across multiple datasets and noise levels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb if requested
    if use_wandb:
        wandb.init(project="DeFInet", config={
            "crop_size": crop_size,
            "num_crops": 34
        })
    
    # Load model
    main_model = load_models(
        './main_model/final_model.pth',
        device
    )
    main_model.to(device).eval()
    main_model.bypass = True
    print("Main Model Bypass!")
    
    # Define test configurations
    datasets = ['CBSD68', 'mcmaster','kodak'] 
    noise_levels = [15, 25, 50]
    
    # Store results
    all_results = {}
    
    # Iterate through all configurations
    for dataset_name in datasets:
        dataset_results = {}
        for noise_level in noise_levels:
            print(f"\nTesting {dataset_name} with noise level {noise_level}")
            
            avg_psnr, avg_ssim = test_single_configuration(
                main_model=main_model,
                dataset_name=dataset_name,
                noise_level=noise_level,
                batch_size=batch_size,
                crop_size=crop_size,
                num_crops=num_crops,
                device=device,
                use_wandb=use_wandb
            )
            
            # Store results
            dataset_results[noise_level] = {
                'psnr': avg_psnr,
                'ssim': avg_ssim
            }
            
            # Log to wandb if enabled
            if use_wandb:
                wandb.log({
                    f"{dataset_name}_noise_{noise_level}_psnr": avg_psnr,
                    f"{dataset_name}_noise_{noise_level}_ssim": avg_ssim
                })
        
        all_results[dataset_name] = dataset_results
    
    # Print final results
    print("\nFinal Results:")
    for dataset_name, dataset_results in all_results.items():
        print(f"\n{dataset_name} Results:")
        for noise_level, metrics in dataset_results.items():
            print(f"Noise Level {noise_level}:")
            print(f"  PSNR: {metrics['psnr']:.4f}")
            print(f"  SSIM: {metrics['ssim']:.4f}")
    
    if use_wandb:
        wandb.finish()
    
    return all_results

def test_model():
    test()

if __name__ == '__main__': 
    test_model()
    
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