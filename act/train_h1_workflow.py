#!/usr/bin/env python3
"""
Complete workflow for training ACT policy on H1 robot manipulation data.

This script demonstrates the full pipeline:
1. Convert JSON trajectory data to HDF5 format
2. Set up training configuration
3. Train ACT policy
4. Evaluate trained policy

Usage:
    python train_h1_workflow.py --json_file ../teleop/sample_data.json --experiment_name h1_cup_picking
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import json

def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("SUCCESS!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False

def setup_directories():
    """Create necessary directories for ACT training."""
    directories = [
        "../data/records",
        "../data/logs", 
        "../data/processed_episodes"
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def validate_json_data(json_file):
    """Validate that the JSON data has the expected format."""
    print(f"Validating JSON data in {json_file}...")
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Check required fields
        required_fields = ['info', 'text', 'data']
        for field in required_fields:
            if field not in data:
                print(f"ERROR: Missing required field '{field}' in JSON data")
                return False
        
        # Check data structure
        if not data['data']:
            print("ERROR: No trajectory data found in JSON")
            return False
        
        # Check first timestep structure
        first_timestep = data['data'][0]
        required_timestep_fields = ['states', 'actions']
        for field in required_timestep_fields:
            if field not in first_timestep:
                print(f"ERROR: Missing field '{field}' in timestep data")
                return False
        
        # Check joint data structure
        required_joint_groups = ['left_arm', 'right_arm', 'left_hand', 'right_hand']
        for group in required_joint_groups:
            if group not in first_timestep['states']:
                print(f"ERROR: Missing joint group '{group}' in states")
                return False
            if 'qpos' not in first_timestep['states'][group]:
                print(f"ERROR: Missing 'qpos' in states.{group}")
                return False
        
        print(f"✓ JSON data validation passed!")
        print(f"✓ Found {len(data['data'])} timesteps")
        print(f"✓ Task: {data['text'].get('goal', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to validate JSON data: {e}")
        return False

def convert_data_to_hdf5(json_file, output_dir, episode_id=0, num_episodes=5):
    """Convert JSON data to HDF5 format for ACT training."""
    cmd = [
        sys.executable, "convert_json_to_hdf5.py",
        "--json_file", json_file,
        "--output_dir", output_dir,
        "--episode_id", str(episode_id),
        "--num_episodes", str(num_episodes)
    ]
    
    return run_command(cmd, f"Converting JSON data to HDF5 format ({num_episodes} episodes)")

def train_act_policy(dataset_dir, experiment_name, num_epochs=1000, batch_size=4, no_wandb=False):
    """Train the ACT policy on the converted data."""
    cmd = [
        sys.executable, "train_act_policy.py",
        "--dataset_dir", dataset_dir,
        "--experiment_name", experiment_name,
        "--num_epochs", str(num_epochs),
        "--batch_size", str(batch_size),
        "--lr", "1e-5",
        "--chunk_size", "50",  # Smaller chunk size for shorter trajectories
        "--hidden_dim", "256",  # Smaller model for faster training
        "--dim_feedforward", "1024"
    ]
    
    # Add no_wandb flag if requested
    if no_wandb:
        cmd.append("--no_wandb")
    
    return run_command(cmd, f"Training ACT policy '{experiment_name}'")

def main():
    parser = argparse.ArgumentParser(description='Complete H1 robot ACT training workflow')
    
    # Data and experiment configuration
    parser.add_argument('--json_file', type=str, required=True,
                       help='Path to JSON trajectory data file')
    parser.add_argument('--experiment_name', type=str, required=True,
                       help='Name for this training experiment')
    parser.add_argument('--output_dir', type=str, default='../data/processed_episodes',
                       help='Directory to save processed HDF5 files')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=1000,
                       help='Number of training epochs (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Training batch size (default: 4)')
    parser.add_argument('--episode_id', type=int, default=0,
                       help='Episode ID for HDF5 file naming (default: 0)')
    parser.add_argument('--num_episodes', type=int, default=10,
                       help='Number of episodes to create from JSON data (default: 10)')
    
    # Training options
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable wandb logging during training')
    
    # Workflow control
    parser.add_argument('--skip_conversion', action='store_true',
                       help='Skip data conversion step (if already done)')
    parser.add_argument('--convert_only', action='store_true',
                       help='Only convert data, do not train')
    
    args = parser.parse_args()
    
    print("="*80)
    print("H1 ROBOT ACT POLICY TRAINING WORKFLOW")
    print("="*80)
    print(f"JSON file: {args.json_file}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Training epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Wandb logging: {'Disabled' if args.no_wandb else 'Enabled'}")
    print("="*80)
    
    # Validate inputs
    if not os.path.exists(args.json_file):
        print(f"ERROR: JSON file {args.json_file} not found!")
        return 1
    
    # Step 1: Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()
    
    # Step 2: Validate JSON data
    print("\n✅ Validating JSON data...")
    if not validate_json_data(args.json_file):
        return 1
    
    # Step 3: Convert data to HDF5 format
    if not args.skip_conversion:
        print("\n🔄 Converting JSON to HDF5 format...")
        if not convert_data_to_hdf5(args.json_file, args.output_dir, args.episode_id, args.num_episodes):
            print("ERROR: Data conversion failed!")
            return 1
    else:
        print("\n⏭️  Skipping data conversion...")
    
    # Check if HDF5 files exist
    hdf5_files = [f for f in os.listdir(args.output_dir) 
                  if f.startswith('processed_episode_') and f.endswith('.hdf5')]
    if not hdf5_files:
        print(f"ERROR: No HDF5 files found in {args.output_dir}")
        return 1
    
    print(f"✓ Found {len(hdf5_files)} processed episode files")
    
    if args.convert_only:
        print("\n🎯 Conversion complete! Use --skip_conversion to train on this data.")
        return 0
    
    # Step 4: Train ACT policy
    print("\n🤖 Training ACT policy...")
    if not train_act_policy(args.output_dir, args.experiment_name, 
                           args.num_epochs, args.batch_size, args.no_wandb):
        print("ERROR: Training failed!")
        return 1
    
    # Step 5: Success message
    print("\n" + "="*80)
    print("🎉 TRAINING WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"✓ Data converted from: {args.json_file}")
    print(f"✓ HDF5 files saved to: {args.output_dir}")
    print(f"✓ Created {args.num_episodes} episodes")
    print(f"✓ Model trained for: {args.num_epochs} epochs")
    print(f"✓ Experiment name: {args.experiment_name}")
    print(f"✓ Checkpoints saved to: ../data/logs/h1_manipulation/{args.experiment_name}/")
    print("\nNext steps:")
    print("1. Check training logs in ../data/logs/")
    print("2. Evaluate the trained policy")
    print("3. Use the policy for robot control")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 