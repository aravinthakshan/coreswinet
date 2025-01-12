import wandb
import segmentation_models_pytorch as smp
from .train_utils import TrainEpoch, ValidEpoch
from .loss import custom_loss, custom_loss_val
from .dataloader import Dataset
from .transformations import get_training_augmentation, get_preprocessing,resize
from .model import Unet
from torch.utils.data import DataLoader
from .model import Discriminator
import torch

def train(epochs, 
          batch_size, 
          hr_dir, 
          tar_dir, 
          hr_val_dir, 
          tar_val_dir, 
          hr_test_dir,
          tar_test_dir,
          encoder='resnet34', 
          encoder_weights='imagenet', 
          device='cuda', 
          lr=1e-4,
          beta=1, 
          loss_weight=0.5,
          gan_type='standard'
          ):

    activation = 'tanh' 
    # create segmentation model with pretrained encoder
    model = Unet(
        encoder_name=encoder, 
        encoder_weights=encoder_weights, 
        encoder_depth = 5,
        classes=1, 
        activation=activation,
        fusion=True,
        contrastive=True,
    )

    disc = Discriminator().to(device)

    # preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    train_dataset = Dataset(
        hr_dir,
        tar_dir,
        augmentation=get_training_augmentation(), 
        preprocessing= True,
        resize = resize()
    )
    valid_dataset = Dataset(
        hr_val_dir,
        tar_val_dir,
        augmentation=None, 
        preprocessing=True,
        resize = resize()
    )
    test_dataset = Dataset(
        hr_test_dir,
        tar_test_dir,
        augmentation=None, 
        preprocessing=True,
        resize = resize()
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
    # loss = custom_loss(batch_size, beta=beta, loss_weight=loss_weight, gan_type=gan_type)
    # loss_val = custom_loss_val(loss_weight=loss_weight, gan_type=gan_type)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,250)
    train_epoch = TrainEpoch(
        beta=beta,
        model=model,
        discriminator=disc,
        loss_weight=loss_weight, 
        device=device,
        verbose=True,
        contrastive=True,
        gan_type=gan_type,
        batch_size=batch_size
    )
    valid_epoch = ValidEpoch(
        model=model,
        discriminator=disc, 
        loss_weight=loss_weight, 
        device=device,
        verbose=True,
        gan_type=gan_type,
        batch_size=batch_size
    )

    min_mse = 0
    min_mae = 0
    max_ssim = 0
    max_psnr = 0
    for i in range(0, epochs):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        print(train_logs)
        wandb.log({'epoch':i+1,
                    't_loss':train_logs['LOSS'],
                    # 't_gan_loss': train_logs['gan_loss'],
                    'v_loss':valid_logs['LOSS'],
                    't_ssim':train_logs['SSIM'],
                    'v_ssim':valid_logs['SSIM'],
                    't_psnr':train_logs['PSNR'],
                    'v_psnr':valid_logs['PSNR'],
                    't_mse':train_logs['MSE'],
                    'v_mse':valid_logs['MSE'],
                    't_mae':train_logs['MAE'],
                    'v_mae':valid_logs['MAE']
                    })
        #do something (save model, change lr, etc.)
        if min_mse <= valid_logs['MSE']:
            min_mse = valid_logs['MSE']
            min_mae = valid_logs['MAE']
            max_psnr = valid_logs['PSNR']
            max_ssim = valid_logs['SSIM']
            wandb.config.update({'min_mae':min_mae,'min_mse':min_mse, 'max_ssim':max_ssim, 'max_psnr':max_psnr}, allow_val_change=True)
            torch.save(model.state_dict(), './best_model.pth')
            print('Model saved!')
    print(f'max ssim: {max_ssim} max psnr: {max_psnr} min mse: {min_mse} min mae: {min_mae}')

def train_model(configs):
    train(configs['epochs'], configs['batch_size'], configs['hr_dir'],
         configs['tar_dir'], configs['hr_val_dir'],
         configs['tar_val_dir'], configs['hr_test_dir'],configs['tar_test_dir'], configs['encoder'],
         configs['encoder_weights'], configs['device'], configs['lr'], configs['beta'],
         configs['loss_weight'],  configs['gan_type'])