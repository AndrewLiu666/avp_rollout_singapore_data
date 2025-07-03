#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import argparse
import time

# Add the current directory to path to import ACT modules
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# Import ACT modules directly
from utils import load_data
from policy import ACTPolicy, CNNMLPPolicy
import torch
import pickle
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import wandb
import h5py

# Constants matching the ACT training setup
RECORD_DIR = Path("../data/records")  # Where your processed episodes are stored
LOG_DIR = Path("../data/logs")  # Where training logs and checkpoints are saved


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def compute_dict_mean(epoch_dicts):
    result = {}
    for key in epoch_dicts[0].keys():
        result[key] = torch.stack([d[key] for d in epoch_dicts]).mean()
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        print("policy config", policy_config)
        policy = ACTPolicy(policy_config)

    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def forward_pass(data, policy):
    if len(data) != 4:
        print(f"ERROR - Expected 4 elements, got {len(data)}: {data}")
        return None
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad)


def repeater(data_loader):
    """Infinite data loader that repeats the dataset indefinitely"""
    while True:
        for data in data_loader:
            yield data


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    min_val_loss = np.inf
    best_ckpt_info = None

    train_dataloader = repeater(train_dataloader)
    print("num_epochs", num_epochs)
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        if epoch % 50 == 0:  # Validate every 50 epochs instead of 500
            # validation
            with torch.inference_mode():
                policy.eval()
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    validation_dicts.append(forward_dict)
                    if batch_idx > 5:  # Reduced validation batches
                        break

                validation_summary = compute_dict_mean(validation_dicts)

                epoch_val_loss = validation_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))

            for k in list(validation_summary.keys()):
                validation_summary[f'val/{k}'] = validation_summary.pop(k)

            if not config.get('no_wandb', False):
                wandb.log(validation_summary, step=epoch)

            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in validation_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()

        data = next(train_dataloader)
        forward_dict = forward_pass(data, policy)
        if forward_dict is None:
            print("Skipping batch due to data error")
            continue
        # backward
        loss = forward_dict['loss']
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_summary = detach_dict(forward_dict)

        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if not config.get('no_wandb', False):
            wandb.log(epoch_summary, step=epoch)

        if epoch % 100 == 0 and epoch >= 100:  # Save checkpoints every 100 epochs
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    if best_ckpt_info is None:
        best_ckpt_info = (num_epochs - 1, epoch_train_loss, policy.state_dict())

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    return best_ckpt_info


def train_act_h1_policy(dataset_dir, experiment_name, **kwargs):
    """
    Train ACT policy for H1 robot manipulation.
    
    Args:
        dataset_dir: Directory containing processed HDF5 episode files
        experiment_name: Name for this training experiment
        **kwargs: Additional training parameters
    """

    # Default training configuration for H1 robot
    default_config = {
        'policy_class': 'ACT',
        'task_name': 'h1_manipulation',
        'batch_size': 8,
        'num_epochs': 2000,
        'lr': 1e-5,
        'chunk_size': 100,  # Action chunking size
        'hidden_dim': 512,
        'dim_feedforward': 3200,
        'kl_weight': 10,
        'temporal_agg': True,
        'seed': 0,
        'eval': False,
        'onscreen_render': False,
        'qpos_noise_std': 0.0,
        'no_wandb': True,
        'save_jit': False,
        'resumeid': None,
        'resume_ckpt': None,
    }

    # Update with user-provided parameters
    config = {**default_config, **kwargs}
    config['exptid'] = experiment_name
    config['taskid'] = config['task_name']  # taskid is required by ACT

    # Set up directories
    os.makedirs(RECORD_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create a task directory structure that ACT expects
    task_dir = RECORD_DIR / config['task_name']
    os.makedirs(task_dir, exist_ok=True)

    # # Create processed directory and copy/link the dataset
    # processed_dir = task_dir / 'processed'
    # if processed_dir.exists():
    #     # Remove existing symlink if it exists
    #     processed_dir.unlink()
    # processed_dir.symlink_to(Path(dataset_dir).absolute())

    # Create processed directory and copy/link the dataset
    processed_dir = task_dir / 'processed'
    if processed_dir.exists():
        processed_dir.unlink()
    # ✅ only create symlink if dataset_dir is directory
    if os.path.isdir(dataset_dir):
        processed_dir.symlink_to(Path(dataset_dir).absolute())
    else:
        processed_dir = Path(dataset_dir)  # just use the file path

    # Set up checkpoint directory
    ckpt_dir = LOG_DIR / config['task_name'] / experiment_name
    os.makedirs(ckpt_dir, exist_ok=True)
    config['ckpt_dir'] = str(ckpt_dir)

    print("=" * 60)
    print("TRAINING ACT POLICY FOR H1 ROBOT")
    print("=" * 60)
    print(f"Dataset directory: {dataset_dir}")
    print(f"Task name: {config['task_name']}")
    print(f"Experiment name: {experiment_name}")
    print(f"Policy class: {config['policy_class']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Chunk size: {config['chunk_size']}")
    print(f"Hidden dim: {config['hidden_dim']}")
    print(f"Number of epochs: {config['num_epochs']}")
    print(f"Seed: {config['seed']}")
    print(f"Checkpoint dir: {ckpt_dir}")
    print("=" * 60)

    # Initialize wandb if not disabled
    if not config.get('no_wandb', False):
        wandb.init(project="h1_act_training", name=experiment_name,
                   group=config['task_name'], mode="online", dir=str(LOG_DIR))
        wandb.config.update(config)
    else:
        # Initialize wandb in disabled mode
        wandb.init(mode="disabled")

    try:
        # Load data
        # camera_names = ['left', 'right']
        # camera_names = ['agentview', 'robot0_eye_in_hand', 'robot1_eye_in_hand']
        camera_names = ['agentview']

        # Get task parameters
        if dataset_dir.endswith('.hdf5'):
            with h5py.File(dataset_dir, 'r') as f:
                first_demo = sorted(f['data'].keys())[0]
                state_dim = f[f'data/{first_demo}/obs/joint_positions'].shape[1]
                action_dim = f[f'data/{first_demo}/actions'].shape[1]
        else:
            state_dim = 26
            action_dim = 28
        print(f"Auto-inferred dims: state_dim={state_dim}, action_dim={action_dim}")


        lr_backbone = 1e-5
        backbone = 'dino_v2'

        if config['policy_class'] == 'ACT':
            enc_layers = 4
            dec_layers = 7
            nheads = 8
            policy_config = {
                'lr': config['lr'],
                'num_queries': config['chunk_size'],
                'kl_weight': config['kl_weight'],
                'hidden_dim': config['hidden_dim'],
                'dim_feedforward': config['dim_feedforward'],
                'lr_backbone': lr_backbone,
                'backbone': backbone,
                'enc_layers': enc_layers,
                'dec_layers': dec_layers,
                'nheads': nheads,
                'camera_names': camera_names,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'qpos_noise_std': config['qpos_noise_std'],
                'policy_class': config['policy_class'],
                'seed': config['seed'],
                'task_id': config['task_name'],
                'exptid': config['exptid'],
            }
        elif config['policy_class'] == 'CNNMLP':
            policy_config = {
                'lr': config['lr'],
                'lr_backbone': lr_backbone,
                'backbone': backbone,
                'num_queries': 1,
                'camera_names': camera_names,
            }
        else:
            raise NotImplementedError

        config['policy_config'] = policy_config

        print("Loading data...")
        train_dataloader, val_dataloader, stats, _ = load_data(
            str(processed_dir), camera_names, config['batch_size'], config['batch_size']
        )

        # Save dataset stats
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        stats['config'] = policy_config
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)

        print("Starting training...")
        start_time = time.time()

        best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
        best_epoch, min_val_loss, best_state_dict = best_ckpt_info

        # Save best checkpoint
        ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
        torch.save(best_state_dict, ckpt_path)
        print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')

        training_time = time.time() - start_time
        print("=" * 60)
        print(f"TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Training time: {training_time / 3600:.2f} hours")
        print(f"Checkpoints saved to: {ckpt_dir}")
        print("=" * 60)

    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train ACT policy for H1 robot manipulation')

    # Required arguments
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Directory containing processed HDF5 episode files')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Name for this training experiment')

    # Training hyperparameters
    parser.add_argument('--policy_class', type=str, default='ACT', choices=['ACT', 'CNNMLP'],
                        help='Policy class to use (default: ACT)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size (default: 8)')
    parser.add_argument('--num_epochs', type=int, default=2000,
                        help='Number of training epochs (default: 2000)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate (default: 1e-5)')
    parser.add_argument('--chunk_size', type=int, default=100,
                        help='Action chunking size (default: 100)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension for transformer (default: 512)')
    parser.add_argument('--dim_feedforward', type=int, default=3200,
                        help='Feedforward dimension for transformer (default: 3200)')
    parser.add_argument('--kl_weight', type=float, default=10,
                        help='KL divergence weight (default: 10)')
    parser.add_argument('--qpos_noise_std', type=float, default=0.0,
                        help='Joint position noise standard deviation (default: 0.0)')

    # Training options
    parser.add_argument('--no_temporal_agg', action='store_true',
                        help='Disable temporal aggregation')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--onscreen_render', action='store_true',
                        help='Enable onscreen rendering during training')

    # Resume training
    parser.add_argument('--resumeid', type=str,
                        help='Experiment ID to resume from')
    parser.add_argument('--resume_ckpt', type=str,
                        help='Specific checkpoint to resume from')

    args = parser.parse_args()

    # Validate dataset directory
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory {args.dataset_dir} does not exist!")
        return

    # Check if there are any processed episode files
    # processed_files = [f for f in os.listdir(args.dataset_dir) if f.startswith('processed_episode_') and f.endswith('.hdf5')]
    # if not processed_files:
    #     print(f"Error: No processed episode files found in {args.dataset_dir}")
    #     print("Please run the conversion script first to convert your JSON data to HDF5 format")
    #     return
    #
    # print(f"Found {len(processed_files)} processed episode files")

    if not (args.dataset_dir.endswith('.hdf5') and os.path.isfile(args.dataset_dir)):
        processed_files = [f for f in os.listdir(args.dataset_dir) if
                           f.startswith('processed_episode_') and f.endswith('.hdf5')]
        if not processed_files:
            print(f"Error: No processed episode files found in {args.dataset_dir}")
            print("Please run the conversion script first to convert your JSON data to HDF5 format")
            return
        print(f"Found {len(processed_files)} processed episode files")

    # Convert args to dict and filter out None values
    training_config = {k: v for k, v in vars(args).items()
                       if v is not None and k not in ['dataset_dir', 'experiment_name']}

    # Convert no_temporal_agg to temporal_agg
    if 'no_temporal_agg' in training_config:
        training_config['temporal_agg'] = not training_config.pop('no_temporal_agg')

    # Start training
    train_act_h1_policy(
        dataset_dir=args.dataset_dir,
        experiment_name=args.experiment_name,
        **training_config
    )


if __name__ == "__main__":
    main()
