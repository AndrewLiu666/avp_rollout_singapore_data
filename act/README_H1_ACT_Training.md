# ACT Policy Training for H1 Robot with Inspire Hands

This guide explains how to train Action Chunking with Transformers (ACT) policies for the H1 humanoid robot using teleoperation demonstration data.

## Overview

The ACT (Action Chunking with Transformers) framework enables learning complex bimanual manipulation policies from demonstration data. This implementation is specifically configured for the H1 humanoid robot with Inspire hands.

## Files Description

- `convert_json_to_hdf5.py` - Converts JSON trajectory data to ACT's HDF5 format
- `train_act_policy.py` - Training script for ACT policies
- `train_h1_workflow.py` - Complete workflow from JSON data to trained policy
- `imitate_episodes.py` - Core ACT training implementation (from original ACT repo)
- `policy.py` - ACT policy models
- `utils.py` - Data loading and processing utilities

## Prerequisites

### Dependencies

```bash
# Install required packages
pip install torch torchvision
pip install h5py numpy matplotlib tqdm einops
pip install opencv-python
pip install wandb  # for experiment tracking (optional)
```

### Data Format

Your JSON trajectory data should follow this structure:

```json
{
    "info": {
        "version": "1.0.0",
        "joint_names": {
            "left_arm": ["joint1", "joint2", ...],
            "right_arm": ["joint1", "joint2", ...],
            "left_hand": ["joint1", "joint2", ...],
            "right_hand": ["joint1", "joint2", ...]
        }
    },
    "text": {
        "goal": "Pick up the red cup on the table",
        "desc": "Task description",
        "steps": "step1: ... step2: ..."
    },
    "data": [
        {
            "idx": 0,
            "states": {
                "left_arm": {"qpos": [7 joint positions]},
                "right_arm": {"qpos": [7 joint positions]},
                "left_hand": {"qpos": [6 joint positions]},
                "right_hand": {"qpos": [6 joint positions]}
            },
            "actions": {
                "left_arm": {"qpos": [7 joint positions]},
                "right_arm": {"qpos": [7 joint positions]},
                "left_hand": {"qpos": [6 joint positions]},
                "right_hand": {"qpos": [6 joint positions]}
            }
        }
    ]
}
```

## Quick Start

### Method 1: Complete Workflow (Recommended)

Use the all-in-one workflow script:

```bash
cd act

# Train with sample data
python train_h1_workflow.py \
    --json_file ../teleop/sample_data.json \
    --experiment_name h1_cup_picking \
    --num_epochs 500 \
    --batch_size 4

# Train with your own data
python train_h1_workflow.py \
    --json_file /path/to/your/data.json \
    --experiment_name my_experiment \
    --num_epochs 1000 \
    --batch_size 8
```

### Method 2: Step-by-Step Process

#### Step 1: Convert JSON to HDF5

```bash
# Convert trajectory data to HDF5 format
python convert_json_to_hdf5.py \
    --json_file ../teleop/sample_data.json \
    --output_dir ../data/processed_episodes \
    --episode_id 0

# If you have multiple trajectories
for i in {0..9}; do
    python convert_json_to_hdf5.py \
        --json_file /path/to/trajectory_${i}.json \
        --output_dir ../data/processed_episodes \
        --episode_id $i
done
```

#### Step 2: Train ACT Policy

```bash
python train_act_policy.py \
    --dataset_dir ../data/processed_episodes \
    --experiment_name h1_manipulation_v1 \
    --num_epochs 2000 \
    --batch_size 8 \
    --lr 1e-5 \
    --chunk_size 100
```

## Training Configuration

### Key Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `num_epochs` | Training epochs | 2000 | 500-5000 |
| `batch_size` | Batch size | 8 | 4-16 |
| `lr` | Learning rate | 1e-5 | 1e-6 to 1e-4 |
| `chunk_size` | Action chunking size | 100 | 50-200 |
| `hidden_dim` | Transformer hidden dim | 512 | 256-1024 |
| `kl_weight` | KL divergence weight | 10 | 1-100 |

### For Different Scenarios

**Small dataset (< 10 episodes):**
```bash
--num_epochs 1000 --batch_size 4 --lr 1e-4 --chunk_size 50
```

**Large dataset (> 100 episodes):**
```bash
--num_epochs 3000 --batch_size 16 --lr 1e-5 --chunk_size 100
```

**Fast prototyping:**
```bash
--num_epochs 200 --batch_size 2 --hidden_dim 256 --dim_feedforward 1024
```

## Data Requirements

### Minimum Data Requirements

- **Episodes**: At least 10-20 demonstration episodes
- **Episode length**: 50-500 timesteps per episode
- **Action diversity**: Demonstrations should cover the range of desired behaviors

### Data Quality Tips

1. **Consistent demonstrations**: Similar task execution across episodes
2. **Smooth trajectories**: Avoid jerky or erratic movements
3. **Complete episodes**: Each episode should show the full task execution
4. **Varied conditions**: Include slight variations in object positions, approaches

## Robot Configuration

### Joint Mapping

The system expects the following joint structure:

- **Left Arm**: 7 DOF (shoulder, elbow, wrist joints)
- **Right Arm**: 7 DOF (shoulder, elbow, wrist joints)  
- **Left Hand**: 6 DOF (Inspire hand joints)
- **Right Hand**: 6 DOF (Inspire hand joints)

**Total**: 26 state dimensions, 28 action dimensions (including gripper states)

### Action Space

Actions are in joint space:
- Position control for all joints
- Additional gripper control signals
- Normalization applied during training

## Training Monitoring

### Using Wandb (Recommended)

```bash
# Enable wandb logging
python train_act_policy.py \
    --dataset_dir ../data/processed_episodes \
    --experiment_name h1_test \
    # ... other args (wandb enabled by default)

# Disable wandb
python train_act_policy.py \
    --dataset_dir ../data/processed_episodes \
    --experiment_name h1_test \
    --no_wandb \
    # ... other args
```

### Local Monitoring

Training logs are saved to:
- Checkpoints: `../data/logs/h1_manipulation/{experiment_name}/`
- Dataset stats: `../data/logs/h1_manipulation/{experiment_name}/dataset_stats.pkl`

## Using Trained Policies

### Loading a Trained Policy

```python
import torch
from policy import ACTPolicy

# Load checkpoint
checkpoint_path = "../data/logs/h1_manipulation/my_experiment/policy_best.ckpt"
checkpoint = torch.load(checkpoint_path)

# Create and load policy
policy_config = {
    'lr': 1e-5,
    'num_queries': 100,
    'hidden_dim': 512,
    'camera_names': ['left', 'right'],
    'state_dim': 26,
    'action_dim': 28,
    # ... other config parameters
}

policy = ACTPolicy(policy_config)
policy.load_state_dict(checkpoint)
policy.eval()
```

### Running Inference

```python
# Prepare inputs
qpos = torch.tensor(current_joint_positions).float().unsqueeze(0)  # [1, 26]
images = torch.tensor(camera_images).float().unsqueeze(0)  # [1, 2, 3, H, W]

# Get action prediction
with torch.no_grad():
    actions = policy(qpos, images)  # [1, chunk_size, 28]

# Use first action
next_action = actions[0, 0].numpy()  # [28]
```

## Troubleshooting

### Common Issues

1. **Out of memory errors**
   - Reduce batch size: `--batch_size 2`
   - Reduce model size: `--hidden_dim 256 --dim_feedforward 1024`

2. **Training not converging**
   - Increase learning rate: `--lr 1e-4`
   - Reduce KL weight: `--kl_weight 1`
   - Check data quality and normalization

3. **Invalid joint positions**
   - Verify joint limits in your data
   - Check joint ordering matches robot configuration

4. **Missing dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Performance Tips

1. **Use GPU**: Training is much faster with CUDA
2. **Data preprocessing**: Convert all episodes at once before training
3. **Multiple workers**: Increase `num_workers` in dataloader for faster loading
4. **Checkpoint resuming**: Use `--resumeid` to continue interrupted training

## Advanced Usage

### Custom Data Loaders

You can modify `utils.py` to handle different data formats or add data augmentation.

### Policy Architecture

Modify `policy.py` to experiment with different transformer architectures or add custom modules.

### Multi-task Training

Train on multiple tasks by combining datasets:

```bash
# Convert multiple task datasets
python convert_json_to_hdf5.py --json_file task1_data.json --output_dir ../data/episodes --episode_id 0
python convert_json_to_hdf5.py --json_file task2_data.json --output_dir ../data/episodes --episode_id 1

# Train on combined data
python train_act_policy.py --dataset_dir ../data/episodes --experiment_name multi_task
```

## Example Commands

### Complete Training Pipeline

```bash
# 1. Convert demo data
python convert_json_to_hdf5.py \
    --json_file ../teleop/sample_data.json \
    --output_dir ../data/processed_episodes \
    --episode_id 0

# 2. Train policy
python train_act_policy.py \
    --dataset_dir ../data/processed_episodes \
    --experiment_name h1_cup_manipulation \
    --num_epochs 1000 \
    --batch_size 8 \
    --lr 1e-5 \
    --chunk_size 50

# 3. Resume training if needed
python train_act_policy.py \
    --dataset_dir ../data/processed_episodes \
    --experiment_name h1_cup_manipulation_v2 \
    --resumeid h1_cup_manipulation \
    --resume_ckpt policy_epoch_500_seed_0.ckpt
```

### Quick Testing

```bash
# Fast training for testing
python train_h1_workflow.py \
    --json_file ../teleop/sample_data.json \
    --experiment_name quick_test \
    --num_epochs 100 \
    --batch_size 2
```

## Next Steps

After training:

1. **Evaluate policy performance** in simulation
2. **Deploy to real robot** using the trained weights
3. **Collect more data** to improve performance
4. **Fine-tune** on specific tasks or conditions

For questions or issues, refer to the original ACT paper: [Learning fine-grained bimanual manipulation with low-cost hardware](https://arxiv.org/abs/2304.13705) 