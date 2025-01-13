from utils.trainer import train_model
import argparse
import wandb

def main(args):
    config={
        'epochs' : args.epochs,
        'train_dir' : args.train_dir,
        'val_dir' : args.val_dir,
        'batch_size' : args.batch_size,
        'device' : args.device,
        'lr' : args.lr,
        'wandb': args.wandbd
    }
    if args.wandbd:
        wandb.login(key=args.key)
        wandb.init(
            project = "DeFInet",
            config = {
                "Epochs": args.epochs,
                "Dataset": args.dataset_name,
                "Batch Size": args.batch_size,
                "Learning Rate": args.lr
            }
        )
    train_model(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default=8)
    parser.add_argument('--epochs', type=int, required=False, default=10)
    parser.add_argument('--lr', type=int, required=False, default=1e-4)
    parser.add_argument('--device', type=str, required=False, default='cuda')
    
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    
    parser.add_argument('--wandbd', type=bool,default=True)
    
    # never set key in REPO
    parser.add_argument('--key', type = str, required=False, default = '9097b6348907fd8bad133bde5c71d9e0c08fde45')
    arguments=parser.parse_args()
    main(arguments)