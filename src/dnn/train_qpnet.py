import argparse
import os

import torch

from src.dnn.models import build_qpnet

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from src.dnn.data.q_patch_dataset import QPatchDataset
from src.dnn.trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #dataset
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--mode', type=str, default='l2')
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--data_name', type=str, default="DIV2K")

    #training & testing
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--num_workers', type=int, default=6)

    #dnn
    parser.add_argument('--output_dim', type=int, default=10)
    parser.add_argument('--num_channels', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    
    # sr model
    parser.add_argument('--sr_model_name', type=str, default='edsr')
    parser.add_argument('--sr_channels', type=int, default=32)
    
    # checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
    parser.add_argument('--profiles', type=str, default='./dataset/profile/')

    args = parser.parse_args()
    
    profile_path = os.path.join(args.profiles, f'size_qual_stats_{args.data_name}_{args.sr_model_name}_{args.sr_channels}_{args.patch_size}.csv')

    train_dataset = QPatchDataset(args, True)
    train_dataset.size_qual_df.to_csv(args.profiles, encoding='utf-8')
    test_dataset = QPatchDataset(args, False, train_dataset.threshold_list)
    
    # set torch device
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.device}'
    device = torch.device('cpu' if args.use_cpu else 'cuda')

    train_loader = data.DataLoader(train_dataset, args.batch_size, True, num_workers=args.num_workers, pin_memory=True)
    test_loader = data.DataLoader(test_dataset, 1, False, num_workers=args.num_workers, pin_memory=True)

    model = build_qpnet(args.num_layers, args.num_channels, args.output_dim, args.patch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.num_epochs//2, gamma=0.5)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, scheduler, device)
    trainer.train(args.num_epochs, args.checkpoint_dir)
