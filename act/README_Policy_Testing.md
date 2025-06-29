# ACT Policy Testing in Isaac Gym

This document explains how to test your trained ACT policy in Isaac Gym simulation.

## Overview

The `test_policy_simulation.py` script allows you to deploy and test your trained ACT policy in a simulated environment that matches your training data. The robot will be controlled by the policy's predictions instead of replaying recorded data.

## Prerequisites

1. **Trained ACT Policy**: You need a trained policy checkpoint (`.ckpt` file)
2. **Dataset Statistics**: You need the normalization stats from training (`.pkl` file)
3. **Isaac Gym**: Properly installed and configured
4. **Robot Assets**: H1 Inspire robot URDF files in the correct location

## Usage

### Basic Usage

```bash
cd act
python test_policy_simulation.py \
    --policy_path ../data/logs/h1_manipulation/h1_cup_test/policy_best.ckpt \
    --stats_path ../data/logs/h1_manipulation/h1_cup_test/dataset_stats.pkl
```

### With Custom Parameters

```bash
python test_policy_simulation.py \
    --policy_path ../data/logs/h1_manipulation/h1_cup_test/policy_best.ckpt \
    --stats_path ../data/logs/h1_manipulation/h1_cup_test/dataset_stats.pkl \
    --max_steps 2000
```

## Arguments

- `--policy_path`: Path to the trained policy checkpoint (`.ckpt` file)
- `--stats_path`: Path to the dataset statistics file (`.pkl` file)  
- `--max_steps`: Maximum number of simulation steps (default: 1000)

## How It Works

### 1. Environment Setup
- Creates the same Isaac Gym environment as the replay script
- H1 robot with Inspire hands
- Table and red cup objects
- Same camera viewpoint and physics settings

### 2. Policy Loading
- Loads the trained ACT policy with the correct configuration
- Loads normalization statistics used during training
- Sets the policy to evaluation mode

### 3. Control Loop
The script runs a continuous control loop:

1. **State Observation**: Gets current robot joint positions (26 dims)
2. **Image Input**: Creates dummy camera images (can be replaced with real camera feeds)
3. **Action Prediction**: Uses the policy to predict a chunk of future actions
4. **Action Execution**: Applies the predicted actions to the robot joints
5. **Simulation Step**: Advances the physics simulation

### 4. Action Chunking
ACT uses action chunking, meaning it predicts multiple future actions at once:
- When the action queue is empty, the policy generates a new chunk of 50 actions
- Actions are executed sequentially from the chunk
- New chunks are generated as needed

## Expected Behavior

- The robot should attempt to perform the manipulation task it was trained on
- Joint movements should be smooth and coordinated
- The policy should generate reasonable trajectories even with dummy camera input

## Troubleshooting

### Common Issues

1. **File Not Found Errors**
   - Check that policy and stats file paths are correct
   - Ensure robot URDF files are in `../teleop/assets/h1_inspire/urdf/`

2. **CUDA Out of Memory**
   - The policy requires GPU memory for inference
   - Close other GPU-intensive applications

3. **Robot Behavior Issues**
   - The policy was trained on specific data - behavior depends on training quality
   - Dummy images may not provide meaningful visual information

### Debugging

The script provides detailed output:
- Policy loading confirmation
- Action chunk generation messages
- Step-by-step progress updates

## File Locations

After training with experiment name `h1_cup_test`, you'll find:

```
data/logs/h1_manipulation/h1_cup_test/
├── policy_best.ckpt          # Best policy checkpoint
├── policy_last.ckpt          # Final policy checkpoint  
├── policy_epoch_*_seed_*.ckpt # Intermediate checkpoints
└── dataset_stats.pkl         # Normalization statistics
```

## Extending the Script

### Adding Real Camera Feeds

Replace the `create_dummy_images()` method to capture real camera images:

```python
def create_dummy_images(self):
    # Replace with actual camera capture
    left_image = capture_left_camera()  # Your camera capture function
    right_image = capture_right_camera()
    
    # Process and format images as (num_cameras, 3, height, width)
    images = np.stack([left_image, right_image], axis=0)
    return torch.from_numpy(images).cuda()
```

### Modifying Robot Control

You can adjust the robot control by modifying:
- `convert_action_to_robot_qpos()`: How actions map to robot joints
- `step_simulation()`: How joint positions are applied
- Control timing and smoothing

### Performance Monitoring

Add performance metrics:
- Task success detection
- Trajectory smoothness analysis  
- Action prediction timing

## Next Steps

1. **Test with different checkpoints**: Try `policy_best.ckpt` vs `policy_last.ckpt`
2. **Analyze behavior**: Observe how well the policy generalizes
3. **Real robot deployment**: Adapt the script for real robot control
4. **Data collection**: Record successful runs for additional training data

This testing framework provides a bridge between simulation training and real-world deployment of your ACT policies. 