import wandb
from torch.utils.data import DataLoader
from utils.misc import get_metrics
from utils.model.plsworkmodel import Model
from utils.dataloader import CBSD68Dataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torchmetrics
import torchmetrics.image
from visualizer import main_vis
from utils.soap_optimizer import SOAP
from utils.model.archs.ZSN2N import train_n2n, N2NNetwork
from utils.loss import ContrastiveLoss, TextureLoss

def train(
    epochs,
    batch_size,
    # dataset_name,
    train_dir,
    val_dir,
    wandb_debug,
    device='cuda',
    lr=3e-3,
    n2n_epochs=1000, #### CHANGE THIS BACK TO 1000
    contrastive_temperature=0.5
):


    # Dataset and dataloaders
    dataset = CBSD68Dataset(root_dir=train_dir, noise_level=25, crop_size=256, num_crops=34, normalize=True)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,drop_last=True )
    
    print(f"Images per epoch: {len(train_loader) * train_loader.batch_size}")

    # Train N2N model
    print("Training N2N model...")
    # Initialize model
    model = N2NNetwork()
    
    n2n_model = train_n2n(epochs=n2n_epochs, model=model, dataloader=train_loader)

    n2n_model.eval()  # Set N2N model to evaluation mode

    # Initialize main model
    model = Model(in_channels=3, contrastive=True).to(device)
    
    # Optimizer
    optimizer = SOAP(
        model.parameters(),
        lr=lr,
        betas=(0.95, 0.95),
        weight_decay=0.01,
        precondition_frequency=10,
        merge_dims=True,
        normalize_grads=True
    )
    
    # Loss functions
    mse_criterion = nn.MSELoss()
    contrastive_loss_fn = ContrastiveLoss(batch_size=batch_size, temperature=contrastive_temperature)
    texture_loss_fn = TextureLoss()
    
    # Metrics
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)
    
    # Logger and tracking
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
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = []
        psnr_train, ssim_train = 0, 0
        
        psnr_metric.reset()
        ssim_metric.reset()
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training Progress") as loader:
            for itr, batch_data in enumerate(loader):
                noise, clean = [x.to(device) for x in batch_data]
                
                # Get N2N denoised output
                with torch.no_grad():
                    n2n_output = n2n_model.denoise(noise)
                
                optimizer.zero_grad()
                
                # Forward pass with both noisy and N2N denoised input
                output, f1, f2 = model(noise, n2n_output)
                
                # Calculate losses
                mse_loss = mse_criterion(output, clean)
                contrastive_loss = contrastive_loss_fn(f1, f2)
                texture_LOSS = texture_loss_fn(output,clean)
                
                # Combined loss
                loss = mse_loss + texture_LOSS + contrastive_loss 
                
                loss.backward()
                optimizer.step()
                
                # Calculate metrics
                psnr_train_itr, ssim_train_itr = get_metrics(clean, output, psnr_metric, ssim_metric)
                
                total_loss.append(loss.item())
                psnr_train += psnr_train_itr
                ssim_train += ssim_train_itr
                
                loader.set_postfix(loss=loss.item(), psnr=psnr_train_itr, ssim=ssim_train_itr)
            
            # Average metrics
            psnr_train /= (itr + 1)
            ssim_train /= (itr + 1)
            avg_loss = sum(total_loss) / len(total_loss)
            
            # Update logger
            logger['train_loss'] = avg_loss
            logger['train_psnr'] = psnr_train
            logger['train_ssim'] = ssim_train
            
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print(f'TRAIN Loss: {avg_loss:.4f}')
            print(f'TRAIN PSNR: {psnr_train:.4f}')
            print(f'TRAIN SSIM: {ssim_train:.4f}')
            
            psnr_metric.reset()
            ssim_metric.reset()
        
        # Validation loop
        model.eval()
        with tqdm(val_loader, desc="Validation Progress") as loader:
            psnr_val, ssim_val = 0, 0
            with torch.no_grad():
                for batch_data in loader:
                    noise, clean = [x.to(device) for x in batch_data]
                    n2n_output = n2n_model.denoise(noise)
                    output, _, _ = model(noise, n2n_output)
                    psnr_val_itr, ssim_val_itr = get_metrics(clean, output, psnr_metric, ssim_metric)
                    psnr_val += psnr_val_itr
                    ssim_val += ssim_val_itr
            
            psnr_val /= len(val_loader)
            ssim_val /= len(val_loader)
            
            logger['val_psnr'] = psnr_val
            logger['val_ssim'] = ssim_val
            logger['epoch'] = epoch + 1
            
            if max_ssim <= ssim_val:
                max_ssim = ssim_val
                max_psnr = psnr_val
                logger['max_ssim'] = max_ssim
                logger['max_psnr'] = max_psnr
                logger['best_epoch'] = epoch + 1
                # Save both models
                torch.save({
                    'main_model': model.state_dict(),
                    'n2n_model': n2n_model.state_dict()
                }, './best_models.pth')
                print(f"Saved Models at epoch {epoch}.")
                
            print(f"\nVal PSNR: {psnr_val:.4f}")
            print(f"Val SSIM: {ssim_val:.4f}")
            
            if wandb_debug:
                wandb.log(logger)
                
    main_vis(val_dir)


def train_model(config):
    train(
        config['epochs'],
        config['batch_size'],
        config['train_dir'],
        config['val_dir'],
        config['wandb'],
        config['device'],
        config['lr'],
    )
    
# def test(test_dir, model_path, device='cuda'):
#     """Test the model on a test dataset"""
#     test_dataset = CBSD68Dataset(root_dir=test_dir, noise_level=25, crop_size=256, num_crops=1, normalize=True)
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
#     main_model = Model(in_channels=3, contrastive=False).to(device)
#     n2n_model = N2NNetwork().to(device)
    
#     # Load saved weights
#     checkpoint = torch.load(model_path)
#     main_model.load_state_dict(checkpoint['main_model'])
#     n2n_model.load_state_dict(checkpoint['n2n_model'])
    
#     main_model.eval()
#     n2n_model.eval()
    
#     # Initialize metrics
#     p = torchmetrics.image.PeakSignalNoiseRatio().to(device)
#     z = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)
    
#     total_psnr = 0
#     total_ssim = 0
    
#     with torch.no_grad():
#         for batch_data in tqdm(test_loader, desc="Testing Progress"):
#             noise, clean = [x.to(device) for x in batch_data]
#             n2n_output = n2n_model.denoise(noise)
#             output, _, _ = main_model(noise, n2n_output)
            
#             psnr_val, ssim_val = get_metrics(clean, output, p, z)
#             total_psnr += psnr_val
#             total_ssim += ssim_val
    
#     avg_psnr = total_psnr / len(test_loader)
#     avg_ssim = total_ssim / len(test_loader)
    
#     print(f"Test Results:")
#     print(f"Average PSNR: {avg_psnr:.4f}")
#     print(f"Average SSIM: {avg_ssim:.4f}")
    
#     return avg_psnr, avg_ssim