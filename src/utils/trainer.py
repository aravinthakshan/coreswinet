import wandb
from torch.utils.data import DataLoader
from utils.misc import get_metrics, visualize_epoch, un_tan_fi
from utils.model.coreswinet import Model
from utils.dataloader import CBSD68Dataset, Waterloo, get_training_augmentation
from tqdm import tqdm
import torch
import torch.nn as nn
import torchmetrics
import torchmetrics.image
from visualizer import main_vis
from utils.soap_optimizer import SOAP
from utils.model.archs.ZSN2N import train_n2n, N2NNetwork
from utils.model.archs.Discriminator import Discriminator 
from utils.loss import ContrastiveLoss, TextureLoss, PSNRLoss, GANLoss
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
    # Dataset and dataloaders setup (unchanged)
    dataset = Waterloo(root_dir=train_dir, noise_level=25, crop_size=256, num_crops=2, normalize=True, augment=get_training_augmentation())
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    bypass_epoch = 20
    
    # Initialize N2N model (unchanged)
    print("Training N2N model...")
    model = N2NNetwork()
    n2n_model, psnr_threshold = train_n2n(epochs=n2n_epochs, model=model, dataloader=train_loader)
    n2n_model.eval()

    # Initialize main model and discriminator
    generator = Model(in_channels=3, contrastive=True, bypass=False).to(device)
    discriminator = Discriminator(in_channels=3).to(device)
    
    # Initialize GAN loss
    gan_criterion = GANLoss(gan_type='lsgan', device=device)
    
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
        lr=lr * 0.5,  # Generally discriminator learns faster, so lower learning rate
        betas=(0.95, 0.95),
        weight_decay=0.01,
        precondition_frequency=10,
        merge_dims=True,
        normalize_grads=True
    )
    
    # Loss functions
    mse_criterion = nn.MSELoss()
    contrastive_loss_fn = ContrastiveLoss(batch_size=batch_size, temperature=contrastive_temperature)
    # psrn_loss_fn = PSNRLoss()
    
    # Metrics
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)
    
    # Logger initialization (unchanged)
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
    }
    
    os.makedirs('./main_model', exist_ok=True)
    os.makedirs('./n2n_model', exist_ok=True)
    
    # Training loop
    for epoch in range(epochs):
        if epoch >= bypass_epoch:
            generator.bypass = True
            print(f"\nEpoch {epoch + 1}: Enabling encoder bypass and disabling contrastive loss")
        else:
            generator.bypass = False

        generator.train()
        discriminator.train()
        total_loss = []
        total_g_loss = []
        total_d_loss = []
        psnr_train, ssim_train = 0, 0
        
        psnr_metric.reset()
        ssim_metric.reset()
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training Progress") as loader:
            for itr, batch_data in enumerate(loader):
                noise, clean = [x.to(device) for x in batch_data]
                n2n_output = un_tan_fi(clean)
                
                # Train Discriminator
                optimizer_D.zero_grad()
                
                # Generate fake images
                fake_images, _, _ = generator(noise, n2n_output)
                
                # Real images loss
                real_pred = discriminator(clean)
                d_real_loss = gan_criterion(real_pred, True, is_disc=True)
                
                # Fake images loss
                fake_pred = discriminator(fake_images.detach())
                d_fake_loss = gan_criterion(fake_pred, False, is_disc=True)
                
                # Combined discriminator loss
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                optimizer_D.step()
                
                # Train Generator
                optimizer_G.zero_grad()
                
                # Generate images again for generator update
                fake_images, f1, f2 = generator(noise, n2n_output)
                fake_pred = discriminator(fake_images)
                
                # Calculate losses
                mse_loss = mse_criterion(fake_images, clean)
                # psnr_loss = psrn_loss_fn(fake_images, clean)
                g_loss = gan_criterion(fake_pred, True, is_disc=False)
                
                # Combine losses
                if epoch < bypass_epoch:
                    contrastive_loss = contrastive_loss_fn(f1, f2)
                    loss = 2000 * mse_loss + g_loss + 0.01 * contrastive_loss
                else:
                    loss = 2000 * mse_loss + g_loss
                
                loss.backward()
                optimizer_G.step()
                
                # Calculate metrics
                psnr_train_itr, ssim_train_itr = get_metrics(clean, fake_images, psnr_metric, ssim_metric)
                
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
                    ssim=ssim_train_itr
                )
            
            # Average metrics
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
            
                    # Validation loop
        model.eval()
        
        if epoch >= bypass_epoch:
            model.bypass = True
            max_psnr = 0
            max_ssim = 0
        else:
            model.bypass = False
            
        with tqdm(val_loader, desc="Validation Progress") as loader:
            psnr_val, ssim_val = 0, 0
            with torch.no_grad():
                for batch_data in loader:
                    noise, clean = [x.to(device) for x in batch_data]        

                    # if use_n2n:
                    #     n2n_output = n2n_model.denoise(noise)
                    # else:
                    #     n2n_output = noise
                    n2n_output = un_tan_fi(clean) ##note
                    output, _, _ = model(noise, n2n_output)
                    psnr_val_itr, ssim_val_itr = get_metrics(clean, output, psnr_metric, ssim_metric)
                    psnr_val += psnr_val_itr
                    ssim_val += ssim_val_itr
            
            psnr_val /= len(val_loader)
            ssim_val /= len(val_loader)
            
            logger['val_psnr'] = psnr_val
            logger['val_ssim'] = ssim_val
            logger['epoch'] = epoch + 1

            if max_psnr <= psnr_val :
                max_ssim = ssim_val
                max_psnr = psnr_val
                logger['max_ssim'] = max_ssim
                logger['max_psnr'] = max_psnr
                logger['best_epoch'] = epoch + 1
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'max_ssim': max_ssim,
                    'max_psnr': max_psnr,
                }, './main_model/best_model.pth')
                print(f"Saved main model at epoch {epoch}.")
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': n2n_model.state_dict(),
                    'max_ssim': max_ssim,
                    'max_psnr': max_psnr,
                }, './n2n_model/best_model_n2n.pth')
                
                print(f"Saved n2n model at epoch {epoch}.")
                print(f"Saved Models at epoch {epoch}.")
                
            print(f"\nVal PSNR: {psnr_val:.4f}")
            print(f"Val SSIM: {ssim_val:.4f}")
            
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
