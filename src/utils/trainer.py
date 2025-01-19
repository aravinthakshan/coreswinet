import wandb
from torch.utils.data import DataLoader
from utils.misc import get_metrics, un_tan_fi
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
from utils.loss import ContrastiveLoss, TextureLoss, PSNRLoss
import os 
import copy

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
    # Dataset and dataloaders
    dataset = Waterloo(root_dir=train_dir, noise_level=25, crop_size=256, num_crops=2, normalize=True,augmentation=get_training_augmentation())
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    bypass_epoch = 30
    target_psnr = 30.9
    model2_saved = False
    
    print(f"Images per epoch Train: {len(train_loader) * train_loader.batch_size}")
    print(f"Images per epoch Val: {len(val_loader) * val_loader.batch_size}")
    
    # Train N2N model
    print("Training N2N model...")
    model = N2NNetwork()
    n2n_model, psnr_threshold = train_n2n(epochs=n2n_epochs, model=model, dataloader=train_loader)
    print("PSNR THRESHOLD:", psnr_threshold)
    n2n_model.eval()

    # Initialize main model
    model = Model(in_channels=3, contrastive=True, bypass=False).to(device)
    
    # Create directories
    os.makedirs('./main_model', exist_ok=True)
    os.makedirs('./n2n_model', exist_ok=True)
    os.makedirs('./model2', exist_ok=True)
    
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
    psnr_loss_func = PSNRLoss()
    
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
    
    # Model2 
    model2 = None
    
    # Training loop
    for epoch in range(epochs):
        # Check if we should enable bypass
        if epoch >= bypass_epoch:
            model.bypass = True
            print(f"\nEpoch {epoch + 1}: Enabling encoder bypass and disabling contrastive loss")
        else:
            model.bypass = False

        model.train()
        total_loss = []
        psnr_train, ssim_train = 0, 0
        
        psnr_metric.reset()
        ssim_metric.reset()
        
        # Training phase
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training Progress") as loader:
            for itr, batch_data in enumerate(loader):
                noise, clean = [x.to(device) for x in batch_data]
                
                # If model2 exists, use its output for the second encoder
                if model2 is not None:
                    with torch.no_grad():
                        model2.eval()
                        model2.bypass_first = True
                        n2n_output, _, _ = model2(noise, un_tan_fi(clean))
                else:
                    n2n_output = un_tan_fi(clean) # ouput not n2n
                
                optimizer.zero_grad()
                
                # Forward pass
                output, f1, f2 = model(noise, n2n_output)
                
                # Calculate losses
                mse_loss = mse_criterion(output, clean)
                psnr_loss = psnr_loss_func(output, clean)
                
                if epoch < bypass_epoch:
                    contrastive_loss = contrastive_loss_fn(f1, f2)
                    loss = mse_loss + 0.01 * contrastive_loss + 0.1 * psnr_loss
                else:
                    loss = mse_loss + 0.01 * psnr_loss 
                
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
            
            logger['train_loss'] = avg_loss
            logger['train_psnr'] = psnr_train
            logger['train_ssim'] = ssim_train
            
            print(f'\nEpoch {epoch + 1}/{epochs}')
            print(f'TRAIN Loss: {avg_loss:.4f}')
            print(f'TRAIN PSNR: {psnr_train:.4f}')
            print(f'TRAIN SSIM: {ssim_train:.4f}')
        
        # Validation phase
        model.eval()
        psnr_metric.reset()
        ssim_metric.reset()
        
        with tqdm(val_loader, desc="Validation Progress") as loader:
            psnr_val, ssim_val = 0, 0
            with torch.no_grad():
                for batch_data in loader:
                    noise, clean = [x.to(device) for x in batch_data]
                    
                    # Use model2's output if it exists
                    if model2 is not None:
                        model2.eval()
                        model2.bypass_first = True
                        n2n_output, _, _ = model2(noise, un_tan_fi(clean))
                    else:
                        n2n_output = un_tan_fi(clean)
                    
                    output, _, _ = model(noise, n2n_output)
                    psnr_val_itr, ssim_val_itr = get_metrics(clean, output, psnr_metric, ssim_metric)
                    psnr_val += psnr_val_itr
                    ssim_val += ssim_val_itr
            
            psnr_val /= len(val_loader)
            ssim_val /= len(val_loader)
            
            logger['val_psnr'] = psnr_val
            logger['val_ssim'] = ssim_val
            logger['epoch'] = epoch + 1


            if psnr_val >= target_psnr and not model2_saved and epoch >= bypass_epoch:
                print(f"\nReached target PSNR of {target_psnr}. Saving model2 and starting new fine-tuning phase.")
                model2 = copy.deepcopy(model)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model2.state_dict(),
                    'psnr': psnr_val,
                    'ssim': ssim_val,
                }, './model2/best_model2.pth')
                model2_saved = True
                # Reset
                max_psnr = 0
                max_ssim = 0
                print("Model2 saved. Starting fine-tuning with model2's output.")
                continue

            if max_psnr <= psnr_val:
                max_ssim = ssim_val
                max_psnr = psnr_val
                logger['max_ssim'] = max_ssim
                logger['max_psnr'] = max_psnr
                logger['best_epoch'] = epoch + 1
                
                # Save main model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'max_ssim': max_ssim,
                    'max_psnr': max_psnr,
                }, './main_model/best_model.pth')
                
                # Save N2N model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': n2n_model.state_dict(),
                    'max_ssim': max_ssim,
                    'max_psnr': max_psnr,
                }, './n2n_model/best_model_n2n.pth')
                
                print(f"\nSaved models at epoch {epoch}.")
            
            print(f"\nVal PSNR: {psnr_val:.4f}")
            print(f"Val SSIM: {ssim_val:.4f}")
            
            if wandb_debug:
                wandb.log(logger)
    
  
    if model2 is not None:
        print("\nSaving final state of model2...")
        torch.save({
            'epoch': epochs-1,
            'model_state_dict': model2.state_dict(),
            'final_psnr': psnr_val,
            'final_ssim': ssim_val,
        }, './model2/final_model2.pth')
        print("Final model2 saved.")
        
    else:
        print("\nNo model2 was created during training (target PSNR was not reached).")
        print("Saving current model as final model2...")
        model2 = copy.deepcopy(model)
        torch.save({
            'epoch': epochs-1,
            'model_state_dict': model2.state_dict(),
            'final_psnr': psnr_val,
            'final_ssim': ssim_val,
        }, './model2/final_model2.pth')
        print("Final model2 saved.")

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
