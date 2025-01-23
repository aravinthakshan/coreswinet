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
from utils.loss import ContrastiveLoss, TextureLoss, PSNRLoss
import os 

def train(
    epochs,
    batch_size,
    train_dir,
    # test_dir,
    wandb_debug,
    device='cuda',
    lr=3e-3,
    n2n_epochs=10,
    contrastive_temperature=0.5,
      # New parameter to control when to enable bypass
):
    # Dataset and dataloaders
    dataset = Waterloo(root_dir=train_dir, noise_level=25, crop_size=256, num_crops=2, normalize=True,augmentation=get_training_augmentation())
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    bypass_epoch = 20
    bypass_epoch_second = 50
    
    print(f"Images per epoch Train: {len(train_loader) * train_loader.batch_size}")
    print(f"Images per epoch Val: {len(val_loader) * val_loader.batch_size}")
    # Train N2N model
    print("Training N2N model...")
    model = N2NNetwork()  ### N2N
    n2n_model, psnr_threshold = train_n2n(epochs=n2n_epochs, model=model, dataloader=train_loader)
    print("PSNR THRESHOLD:", psnr_threshold)
    n2n_model.eval()

    # Initialize main model with bypass parameter
    model = Model(in_channels=3, contrastive=True, bypass_first=False, bypass_second=False).to(device)
    modeltwo = Model(in_channels=3).to(device)
    
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
    
    # use_n2n = True
    os.makedirs('./main_model', exist_ok=True)
    os.makedirs('./n2n_model', exist_ok=True)
    os.makedirs('./snap_model', exist_ok=True)
    
    # Training loop
    for epoch in range(epochs):
        # Check if we should enable bypass and disable contrastive loss
        if epoch<bypass_epoch:
            model.bypass_second = False
            model.bypass_first = False

        elif epoch<bypass_epoch_second:
            model.bypass_first = False
            model.bypass_second = True
            print(f"\nEpoch {epoch + 1}: Disabling second encoder and contrastive loss")

        else:
            model.bypass_second = False
            model.bypass_first = True
            print(f"\nEpoch {epoch + 1}: Enabling second encoder, disabling first")

        model.train()
        modeltwo.eval()
        total_loss = []
        psnr_train, ssim_train = 0, 0
        
        psnr_metric.reset()
        ssim_metric.reset()
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training Progress") as loader:
            for itr, batch_data in enumerate(loader):
                noise, clean = [x.to(device) for x in batch_data]
                gt = un_tan_fi(clean)

                if model.bypass_first and modeltwo.bypass_second:
                    mtwo_output, _, _ = modeltwo(noise,gt)
                    gt = un_tan_fi(mtwo_output)

                optimizer.zero_grad()
                output, f1, f2 = model(noise, gt)
                
                # Calculate losses
                mse_loss = mse_criterion(output, clean)
                psnr_loss = psnr_loss_func(output,clean)
                # Only apply contrastive loss before bypass_epoch
                if epoch < bypass_epoch:
                    contrastive_loss = contrastive_loss_fn(f1, f2)
                    loss = mse_loss + 0.01 * contrastive_loss + 0.1 * psnr_loss
                else:
                    loss = mse_loss 
                
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
            
            psnr_metric.reset()
            ssim_metric.reset()
        
        # Validation loop
        model.eval()

        if epoch<bypass_epoch:
            model.bypass_second = False
            model.bypass_first = False
        elif epoch<bypass_epoch_second:
            model.bypass_first = False
            model.bypass_second = True
            max_psnr = 0
            max_ssim = 0
            print(f"\nEpoch {epoch + 1}: Disabling second encoder and contrastive loss")
        else:
            model.bypass_second = False
            model.bypass_first = True
            max_psnr = 0
            max_ssim = 0
            print(f"\nEpoch {epoch + 1}: Enabling second encoder, disabling first")
            
        with tqdm(val_loader, desc="Validation Progress") as loader:
            psnr_val, ssim_val = 0, 0
            with torch.no_grad():
                for batch_data in loader:
                    noise, clean = [x.to(device) for x in batch_data]        
                    gt = un_tan_fi(clean) ##note

                    if model.bypass_first and modeltwo.bypass_second:
                        mtwo_output, _, _ = modeltwo(noise,gt)
                        gt = un_tan_fi(mtwo_output)
                        
                    output, _, _ = model(noise, gt)
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
                if epoch == bypass_epoch_second - 1:
                    
                    snapshot_path = './snap_model/snapshot_model.pth'
                    torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                        }, snapshot_path)
                    print("Saved Snapshot weights")
                        
                    main_checkpoint = torch.load(snapshot_path, map_location=device)
                    modeltwo = Model().to(device)
                    modeltwo.bypass_second= True
                    modeltwo.load_state_dict(main_checkpoint['model_state_dict'])
                    modeltwo.eval() 
                    print("Loaded snapshot model for inference")
    
            print(f"\nVal PSNR: {psnr_val:.4f}")
            print(f"Val SSIM: {ssim_val:.4f}")
            
            if wandb_debug:
                wandb.log(logger)
                
        # After the main training loop ends
    print("\nTraining completed. Saving final models...")
    
    final_main_path = './main_model/final_model.pth'
    final_snap_path = './snap_model/final_snapshot.pth'
    
    torch.save({
        'epoch': epochs-1,
        'model_state_dict': model.state_dict(),
        'max_ssim': max_ssim,
        'max_psnr': max_psnr,
    }, final_main_path)
    
    torch.save({
        'epoch': epochs-1,
        'model_state_dict': modeltwo.state_dict(),
    }, final_snap_path)
    
    if True:
        # Create artifacts
        main_artifact = wandb.Artifact(
            name='final_main_model',
            type='model',
            description='Final state of main model'
        )
        snap_artifact = wandb.Artifact(
            name='final_snapshot_model',
            type='model',
            description='Final state of snapshot model'
        )
        
        main_artifact.add_file(final_main_path)
        snap_artifact.add_file(final_snap_path)
        
        # Log artifacts to wandb
        wandb.log_artifact(main_artifact)
        wandb.log_artifact(snap_artifact)
        
        print("Uploaded final models to wandb as artifacts")
    
def train_model(config):
    train(
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        train_dir=config['train_dir'],
        # test_dir=config['test_dir'],  
        wandb_debug=config['wandb'], 
        device=config['device'],
        lr=config['lr'],
    )