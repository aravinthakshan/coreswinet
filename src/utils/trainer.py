import wandb
from torch.utils.data import DataLoader
from utils.misc import get_metrics, visualize_epoch, un_tan_fi
from utils.model.coreswinet import Model
from utils.dataloader import CBSD68Dataset, Waterloo
from tqdm import tqdm
import torch
import torch.nn as nn
import torchmetrics
import torchmetrics.image
from visualizer import main_vis
from utils.soap_optimizer import SOAP
from utils.model.archs.ZSN2N import train_n2n, N2NNetwork
from utils.model.archs import ConditionalDiscriminator
from utils.loss import ContrastiveLoss, PSNRLoss, GANLoss
import os


def train(
    epochs,
    batch_size,
    train_dir,
    test_dir,
    wandb_debug,
    device='cuda',
    lr=3e-3,
    n2n_epochs=10,
    contrastive_temperature=0.5,
):
    # Dataset and dataloaders setup
    dataset = Waterloo(root_dir=train_dir, noise_level=25, crop_size=256, num_crops=2, normalize=True, augment=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    bypass_epoch = 20
    
    print(f"Images per epoch Train: {len(train_loader) * train_loader.batch_size}")
    print(f"Images per epoch Val: {len(val_loader) * val_loader.batch_size}")
    
    # Initialize models
    print("Training N2N model...")
    n2n_model = N2NNetwork().to(device)
    n2n_model, psnr_threshold = train_n2n(epochs=n2n_epochs, model=n2n_model, dataloader=train_loader)
    n2n_model.eval()

    # Initialize generator and conditional discriminator
    generator = Model(in_channels=3, contrastive=True, bypass=False).to(device)
    discriminator = ConditionalDiscriminator(in_channels=3).to(device)
    
    # Initialize GAN loss
    gan_criterion = GANLoss(gan_type='standard', device=device)
    
    # Optimizers
    optimizer_G = SOAP(
        generator.parameters(),
        lr=lr,
        betas=(0.95, 0.95),
        weight_decay=0.01,
        precondition_frequency=10,
        merge_dims=True,
        normalize_grads=True
    )
    
    optimizer_D = SOAP(
        discriminator.parameters(),
        lr=lr * 0.5,
        betas=(0.95, 0.95),
        weight_decay=0.01,
        precondition_frequency=10,
        merge_dims=True,
        normalize_grads=True
    )
    
    # Loss functions
    mse_criterion = nn.MSELoss()
    contrastive_loss_fn = ContrastiveLoss(batch_size=batch_size, temperature=contrastive_temperature)
    psrn_loss_fn = PSNRLoss()
    
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
        'g_loss': 0,
        'd_loss': 0,
        'n2n_condition_quality': 0
    }
    
    # Create directories for saving models
    os.makedirs('./main_model', exist_ok=True)
    os.makedirs('./n2n_model', exist_ok=True)
    os.makedirs('./discriminator_model', exist_ok=True)
    
    # Training loop
    for epoch in range(epochs):
        if epoch >= bypass_epoch:
            generator.bypass = True
            print(f"\nEpoch {epoch + 1}: Enabling encoder bypass and disabling contrastive loss")
        else:
            generator.bypass = False

        # Training phase
        generator.train()
        discriminator.train()
        n2n_model.eval()
        
        total_loss = []
        total_g_loss = []
        total_d_loss = []
        psnr_train, ssim_train = 0, 0
        
        psnr_metric.reset()
        ssim_metric.reset()
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training Progress") as loader:
            for itr, batch_data in enumerate(loader):
                noise, clean = [x.to(device) for x in batch_data]
                
                # Get N2N noise estimation as condition
                with torch.no_grad():
                    noise_estimation = n2n_model(noise)
                    condition = noise - noise_estimation
                
                # Train Discriminator
                optimizer_D.zero_grad()
                
                fake_images, f1, f2 = generator(noise, condition)
                
                real_pred = discriminator(clean, condition)
                d_real_loss = gan_criterion(real_pred, True, is_disc=True)
                
                fake_pred = discriminator(fake_images.detach(), condition)
                d_fake_loss = gan_criterion(fake_pred, False, is_disc=True)
                
                d_loss = (d_real_loss + d_fake_loss) * 0.5
                d_loss.backward()
                optimizer_D.step()
                
                # Train Generator
                optimizer_G.zero_grad()
                
                fake_images, f1, f2 = generator(noise, condition)
                fake_pred = discriminator(fake_images, condition)
                
                mse_loss = mse_criterion(fake_images, clean)
                psnr_loss = psrn_loss_fn(fake_images, clean)
                g_loss = gan_criterion(fake_pred, True, is_disc=False)
                
                if epoch < bypass_epoch:
                    contrastive_loss = contrastive_loss_fn(f1, f2)
                    loss = mse_loss + 0.05 * contrastive_loss + psnr_loss * 0.01 + 0.1 * g_loss
                else:
                    loss = mse_loss + psnr_loss * 0.01 + 0.1 * g_loss
                
                loss.backward()
                optimizer_G.step()
                
                # Calculate metrics
                psnr_train_itr, ssim_train_itr = get_metrics(clean, fake_images, psnr_metric, ssim_metric)
                n2n_psnr, _ = get_metrics(clean, noise_estimation, psnr_metric, ssim_metric)
                
                total_loss.append(loss.item())
                total_g_loss.append(g_loss.item())
                total_d_loss.append(d_loss.item())
                psnr_train += psnr_train_itr
                ssim_train += ssim_train_itr
                
                loader.set_postfix(
                    loss=loss.item(),
                    g_loss=g_loss.item(),
                    d_loss=d_loss.item(),
                    psnr=psnr_train_itr,
                    ssim=ssim_train_itr,
                    n2n_psnr=n2n_psnr
                )
            
            # Average training metrics
            psnr_train /= (itr + 1)
            ssim_train /= (itr + 1)
            avg_loss = sum(total_loss) / len(total_loss)
            avg_g_loss = sum(total_g_loss) / len(total_g_loss)
            avg_d_loss = sum(total_d_loss) / len(total_d_loss)
            
            logger.update({
                'train_loss': avg_loss,
                'train_psnr': psnr_train,
                'train_ssim': ssim_train,
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss,
            })
            
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print(f'TRAIN Loss: {avg_loss:.4f}')
            print(f'TRAIN PSNR: {psnr_train:.4f}')
            print(f'TRAIN SSIM: {ssim_train:.4f}')
        
        # Validation phase
        generator.eval()
        discriminator.eval()
        
        psnr_metric.reset()
        ssim_metric.reset()
        
        if epoch >= bypass_epoch:
            generator.bypass = True
            max_psnr = 0
            max_ssim = 0
        else:
            generator.bypass = False
            
        with tqdm(val_loader, desc="Validation Progress") as loader:
            psnr_val, ssim_val = 0, 0
            val_g_loss, val_d_loss = 0, 0
            
            with torch.no_grad():
                for batch_data in loader:
                    noise, clean = [x.to(device) for x in batch_data]
                    
                    # Get N2N noise estimation
                    noise_estimation = n2n_model(noise)
                    condition = noise - noise_estimation
                    
                    # Generate fake images
                    fake_images, _, _ = generator(noise, condition)
                    
                    # Calculate GAN losses
                    real_pred = discriminator(clean, condition)
                    fake_pred = discriminator(fake_images, condition)
                    
                    d_real_loss = gan_criterion(real_pred, True, is_disc=True)
                    d_fake_loss = gan_criterion(fake_pred, False, is_disc=True)
                    g_loss = gan_criterion(fake_pred, True, is_disc=False)
                    
                    # Calculate metrics
                    psnr_val_itr, ssim_val_itr = get_metrics(clean, fake_images, psnr_metric, ssim_metric)
                    
                    psnr_val += psnr_val_itr
                    ssim_val += ssim_val_itr
                    val_g_loss += g_loss.item()
                    val_d_loss += (d_real_loss + d_fake_loss).item() * 0.5
            
            # Average validation metrics
            psnr_val /= len(val_loader)
            ssim_val /= len(val_loader)
            val_g_loss /= len(val_loader)
            val_d_loss /= len(val_loader)
            
            logger.update({
                'val_psnr': psnr_val,
                'val_ssim': ssim_val,
                'val_g_loss': val_g_loss,
                'val_d_loss': val_d_loss,
                'epoch': epoch + 1
            })

            # Save best models
            if max_psnr <= psnr_val:
                max_ssim = ssim_val
                max_psnr = psnr_val
                logger.update({
                    'max_ssim': max_ssim,
                    'max_psnr': max_psnr,
                    'best_epoch': epoch + 1
                })
                
                # Save generator (main model)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': generator.state_dict(),
                    'optimizer_state_dict': optimizer_G.state_dict(),
                    'max_ssim': max_ssim,
                    'max_psnr': max_psnr,
                }, './main_model/best_model.pth')
                
                # Save discriminator
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': discriminator.state_dict(),
                    'optimizer_state_dict': optimizer_D.state_dict(),
                }, './discriminator_model/best_discriminator.pth')
                
                # Save N2N model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': n2n_model.state_dict(),
                    'max_ssim': max_ssim,
                    'max_psnr': max_psnr,
                }, './n2n_model/best_model_n2n.pth')
                
                print(f"Saved all models at epoch {epoch + 1}")
            
            print(f"\nVal PSNR: {psnr_val:.4f}")
            print(f"Val SSIM: {ssim_val:.4f}")
            print(f"Val G_Loss: {val_g_loss:.4f}")
            print(f"Val D_Loss: {val_d_loss:.4f}")
            
            if wandb_debug:
                wandb.log(logger)
    
    main_vis(test_dir)

def train_model(config):
    train(
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        train_dir=config['train_dir'],
        test_dir=config['test_dir'],  
        wandb_debug=config['wandb'], 
        device=config['device'],
        lr=config['lr'],
    )