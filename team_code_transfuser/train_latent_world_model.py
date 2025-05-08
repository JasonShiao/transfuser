import os
import sys
from data_latent_world import LatentWorldDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from world_model import TransformerWorldModel
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Latent World Model")
    parser.add_argument("--data_path", type=str, default="/home/ubuntu/transfuser/latent_pred_dataset/latent_pred_dataset.npy", help="Path to the dataset")
    parser.add_argument("--train_log_dir", type=str, default="./train_logs", help="Directory to save logs")
    args = parser.parse_args()
  
    data_npy_path = args.data_path
    
    os.makedirs(args.train_log_dir, exist_ok=True)
    writer = SummaryWriter(args.train_log_dir)
    
    dataset = LatentWorldDataset(data_npy_path)
    num_epochs = 10
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    world_model = TransformerWorldModel(latent_dim=512, seq_len=2).to(device) # prev + current
    summary(world_model, (2, 512), device=device)
    lr = 1e-4
    optimizer = optim.AdamW(world_model.parameters(), lr=lr)
    
    # Resume training if checkpoint exists
    latest_ckpt = os.path.join(args.train_log_dir, 'latest_ckpt.pth')
    if os.path.isfile(latest_ckpt):
        print(f"Resuming from checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        world_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed from epoch {start_epoch} with loss {checkpoint['loss']:.4f}")
    
    
    for epoch in tqdm(range(start_epoch, num_epochs)):
        total_loss = 0
        for data in tqdm(dataloader):
            optimizer.zero_grad()
            # Stack to get full batch: (batch_size, 2, 512)
            inputs = torch.stack([data['prev'], data['current']], dim=1).to(device)
            target = data['next'].to(device)

            pred = world_model(inputs)  # shape: (B, D)
            # print(f"pred shape: {pred.shape}")
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")
        # Save checkpoint
        checkpoint_path = os.path.join(args.train_log_dir, f"wm_ckpt_epoch_{epoch + 1}.pth")
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': world_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        torch.save(checkpoint, checkpoint_path)
        torch.save(checkpoint, os.path.join(args.train_log_dir, 'latest_ckpt.pth'))
        print(f"Checkpoint saved at {checkpoint_path}")
