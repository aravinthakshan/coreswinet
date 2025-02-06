import wandb
from torch.utils.data import DataLoader
from utils.misc import get_metrics, visualize_epoch, un_tan_fi
from utils.model.coreswinet import Model, replace_decoder_convs
# from utils.model.newmodel import Model
from utils.dataloader import CBSD68Dataset, Waterloo,DIV2K,BSD400,SIDD,get_training_augmentation
from tqdm import tqdm
import torch
import torch.nn as nn
import torchmetrics
import torchmetrics.image
from visualizer import main_vis
from utils.soap_optimizer import SOAP
from utils.model.archs.ZSN2N import train_n2n, N2NNetwork
from utils.loss import ContrastiveLoss, TextureLoss, PSNRLoss
import os 

class PReLUBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
    
    def forward(self, x):
        return self.block(x)

class SimpleResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.prelu(self.bn1(self.conv1(x)))
        out += self.shortcut(x)
        return out

class StudentModel(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.channels = [64, 128, 256, 512]
        
        # Initial conv
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Encoder blocks with residual connections
        self.encoder_blocks = nn.ModuleList([
            SimpleResBlock(self.channels[i], self.channels[i+1], stride=2)
            for i in range(len(self.channels)-1)
        ])
        self.encoder_blocks.insert(0, SimpleResBlock(64, 64))
        
        # PReLU blocks after each encoder level
        self.prelu_blocks = nn.ModuleList([
            PReLUBlock(ch) for ch in self.channels
        ])
        
        # Decoder blocks
        decoder_channels = self.channels[::-1]  # Reverse for decoder
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(len(decoder_channels)-1):
            block = nn.Sequential(
                nn.ConvTranspose2d(decoder_channels[i], decoder_channels[i+1], 2, stride=2),
                nn.Conv2d(decoder_channels[i+1] * 2, decoder_channels[i+1], 3, padding=1),
                nn.BatchNorm2d(decoder_channels[i+1]),
                nn.PReLU()
            )
            self.decoder_blocks.append(block)
            
        # Final processing
        self.final = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, in_channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Initial processing
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # Encoder
        features = []
        for enc_block, prelu_block in zip(self.encoder_blocks, self.prelu_blocks):
            x = enc_block(x)
            x = prelu_block(x)
            features.append(x)
            
        # Decoder
        for i, dec_block in enumerate(self.decoder_blocks):
            x = dec_block[0](x)  # Upsample
            skip = features[-(i+2)]  # Skip connection
            x = torch.cat([x, skip], dim=1)
            x = dec_block[1:](x)  # Rest of conv block
            
        return self.final(x)

def train_student(
    teacher_model,
    train_dir,
    test_dir,
    wandb_debug=True,
    dataset_name='BSD',
    noise_level=25,
    device='cuda',
    epochs=100,
    batch_size=32,
    lr=3e-3,
):
    # Dataset and dataloaders (using your existing dataset code)
    if dataset_name=='Waterloo':
        dataset = Waterloo(root_dir=train_dir, noise_level=noise_level, crop_size=256, num_crops=2, normalize=True, augmentation=get_training_augmentation())
    elif dataset_name=='BSD':
        dataset = BSD400(root_dir=train_dir, noise_level=noise_level, crop_size=256, num_crops=20, normalize=True, augmentation=get_training_augmentation())
    elif dataset_name=='DIV2K':
        dataset = DIV2K(root_dir=train_dir, noise_level=noise_level, crop_size=256, num_crops=2, normalize=True, augmentation=get_training_augmentation())
    elif dataset_name=='SIDD':
        train_dataset = SIDD(data_dir=train_dir, normalize=True, standardize=False, mode='train')
        val_dataset = SIDD(data_dir='/kaggle/input/sidd-val', normalize=True, standardize=False, mode='val')

    if dataset_name != 'SIDD':
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Initialize student model
    student_model = StudentModel().to(device)
    
    # Set teacher model to eval mode
    teacher_model.eval()
    teacher_model.bypass = True  # Always use bypass mode for teacher
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Optimizer
    optimizer = SOAP(
        student_model.parameters(),
        lr=lr,
        betas=(0.95, 0.95),
        weight_decay=0.01,
        precondition_frequency=10,
        merge_dims=True,
        normalize_grads=True
    )

    # Loss and metrics
    mse_criterion = nn.MSELoss()
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)

    # Tracking
    max_ssim = 0
    max_psnr = 0
    logger = {
        'epoch': 0,
        'train_loss': 0,
        'train_psnr': 0,
        'train_ssim': 0,
        'val_psnr': 0,
        'val_ssim': 0,
        'best_epoch': 0,
        'max_psnr': 0,
        'max_ssim': 0,
    }

    os.makedirs('./student_model', exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        student_model.train()
        total_loss = []
        psnr_train, ssim_train = 0, 0
        
        psnr_metric.reset()
        ssim_metric.reset()

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training") as loader:
            for itr, batch_data in enumerate(loader):
                noise, clean = [x.to(device) for x in batch_data]
                
                # Get teacher output
                with torch.no_grad():
                    n2n_output = un_tan_fi(clean)  # Using clean as n2n output like in original
                    teacher_output, _ = teacher_model(noise, n2n_output)
                
                # Get student output
                optimizer.zero_grad()
                student_output = student_model(noise)

                # Loss computation
                loss = mse_criterion(student_output, teacher_output)
                loss.backward()
                optimizer.step()

                # Calculate metrics
                psnr_train_itr, ssim_train_itr = get_metrics(clean, student_output, psnr_metric, ssim_metric)
                
                total_loss.append(loss.item())
                psnr_train += psnr_train_itr
                ssim_train += ssim_train_itr
                
                loader.set_postfix(loss=loss.item(), psnr=psnr_train_itr, ssim=ssim_train_itr)

        # Validation loop
        student_model.eval()
        with tqdm(val_loader, desc="Validation") as loader:
            psnr_val, ssim_val = 0, 0
            with torch.no_grad():
                for batch_data in loader:
                    noise, clean = [x.to(device) for x in batch_data]
                    student_output = student_model(noise)
                    psnr_val_itr, ssim_val_itr = get_metrics(clean, student_output, psnr_metric, ssim_metric)
                    psnr_val += psnr_val_itr
                    ssim_val += ssim_val_itr

            psnr_val /= len(val_loader)
            ssim_val /= len(val_loader)

            # Update logger and save best model
            if psnr_val > max_psnr:
                max_psnr = psnr_val
                max_ssim = ssim_val
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': student_model.state_dict(),
                    'max_ssim': max_ssim,
                    'max_psnr': max_psnr,
                }, './student_model/best_model.pth')
                print(f"Saved best student model at epoch {epoch}")

            if wandb_debug:
                logger.update({
                    'epoch': epoch + 1,
                    'val_psnr': psnr_val,
                    'val_ssim': ssim_val,
                    'max_psnr': max_psnr,
                    'max_ssim': max_ssim
                })
                wandb.log(logger)

    # Save final model
    torch.save({
        'epoch': epochs-1,
        'model_state_dict': student_model.state_dict(),
        'max_ssim': max_ssim,
        'max_psnr': max_psnr,
    }, './student_model/final_model.pth')

    if wandb_debug:
        artifact = wandb.Artifact(
            name='final_student_model',
            type='model',
            description='Final state of student model'
        )
        artifact.add_file('./student_model/final_model.pth')
        wandb.log_artifact(artifact)

    return student_model