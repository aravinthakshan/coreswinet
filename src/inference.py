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


def preprocess_image(img_path, device):
    # Open the image and convert it to RGB
    image = Image.open(img_path).convert('RGB')
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(device)  # To tensor, add batch dimension
    image = tan_fi(image)  # Apply tanh-like transformation
    return image


def save_output(output_image, output_path):
    # Remove batch dimension and convert to HxWxC
    if output_image.ndim == 3 and output_image.shape[0] == 3:
        output_image = np.transpose(output_image, (1, 2, 0))  # Convert to HxWxC

    # Convert the output to [0, 255] and save as image
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)  # Ensure values are in the range [0, 255]
    plt.imshow(output_image)
    plt.axis('off')  # Hide axes
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)


def main(args):
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p = torchmetrics.image.PeakSignalNoiseRatio().to(device)
    z = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)
    model = Model()
    
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.to(device)
    dataset = CBSD68Dataset(root_dir=args.train_dir, noise_level=25, crop_size=256, num_crops=34, normalize=True)
    test_loader=DataLoader(dataset,batch_size=1,shuffle=False)
    model.eval()
    itr=0
    with tqdm(test_loader, desc="Testing Progress") as loader:
        for _, batch_data in enumerate(loader):
            noise, clean = [x.to(device) for x in batch_data]
            output = model(noise)
            psnr_train_itr, ssim_train_itr = get_metrics(clean,output,p,z,Standardize=True)
            itr+=1
            
            psnr_train += psnr_train_itr
            ssim_train += ssim_train_itr
    
    psnr_train /= (itr + 1)
    ssim_train /= (itr + 1)

    # Reverse the tanh-like transformation and scale back to [0, 1]
    print("Image before untanifying:", output)
    output = un_tan_fi(output)
    print("Image after untanifying:", output)

    # Remove batch dimension and convert to numpy array
    output = output.squeeze(0).cpu().detach().numpy()

    # Clip the values to [0, 1] and scale to [0, 255]
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)
    print("Image after un-normalizing:", output)

    # Save the output image
    output_folder = "/kaggle/working"
    output_file = os.path.join(output_folder, "output_image.png")
    save_output(output, output_file)
    print(f"Output saved at {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str, required=True, help="Path to the model weights")
    parser.add_argument('--train_dir', type=str, required=True, help="Path to the input image")
    arguments = parser.parse_args()
    main(arguments)
