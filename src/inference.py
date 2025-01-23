import torch
import torch.nn as nn
import argparse
from utils.model.coreswinet import Model
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from utils.dataloader import CBSD68Dataset
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

def test_model(
    batch_size,
    test_dir,
    noise_level=25,
    crop_size=256, 
    num_crops=34,
    device='cuda',
    use_wandb=True,
):
    # Load models
    main_model_snap, _ = load_models(
        './snap_model/snapshot_model.pth', 
        './n2n_model/best_model_n2n.pth', 
        device
    )
    main_model, _ = load_models(
        './main_model/best_model.pth', 
        './n2n_model/best_model_n2n.pth', 
        device
    )

    # Set model states
    for model in [main_model_snap, main_model]:
        model.to(device)
        model.eval()

    # Set bypass states after loading
    main_model_snap.bypass_second = True 
    main_model.bypass_first = True
    print("Models loaded and bypass states set!")
    
    # Initialize dataset and metrics
    dataset = CBSD68Dataset(
        root_dir=test_dir, 
        noise_level=noise_level,
        crop_size=crop_size,
        num_crops=num_crops,
        normalize=True,
        tanfi=True 
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)
    
    total_psnr = total_ssim = 0
    num_batches = 0
    
    with torch.inference_mode():  # More efficient than no_grad
        with tqdm(dataloader, desc="Testing Progress") as loader:
            for noise, clean in loader:
                noise, clean = noise.to(device), clean.to(device)
                
                gt = un_tan_fi(clean)
                main_model_snap_out, _, _ = main_model_snap(noise, gt)
                x = un_tan_fi(main_model_snap_out)
                output_main, _, _ = main_model(noise, x)
                
                psnr_main, ssim_main = get_metrics(clean, output_main, psnr_metric, ssim_metric)
                
                total_psnr += psnr_main
                total_ssim += ssim_main
                num_batches += 1
                
                loader.set_postfix(psnr=psnr_main, ssim=ssim_main)
    
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
