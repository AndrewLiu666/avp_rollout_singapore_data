#!/usr/bin/env python3

import json
import h5py
import numpy as np
import argparse
import os
from pathlib import Path
import cv2
from tqdm import tqdm

def convert_json_to_hdf5(json_file, output_dir, episode_id=0, cameras=['left', 'right'], num_episodes=5, image_dir=None):
    """
    Convert JSON trajectory data to HDF5 format for ACT training.
    Creates multiple episodes from the same trajectory data with slight variations.
    
    Args:
        json_file: Path to the JSON data file
        output_dir: Directory to save the processed HDF5 files
        episode_id: Starting episode ID number
        cameras: List of camera names
        num_episodes: Number of episodes to create (default: 5)
        image_dir: Directory containing real images (optional, uses dummy images if None)
    """
    
    # Load JSON data
    print(f"Loading data from {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    trajectory_data = data['data']
    num_timesteps = len(trajectory_data)
    
    print(f"Converting {num_timesteps} timesteps...")
    print(f"Creating {num_episodes} episodes for ACT training...")
    if image_dir:
        print(f"Using real images from: {image_dir}")
    else:
        print("Using dummy images")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Standard image dimensions for ACT (these will be resized by the model anyway)
    img_height = 224  # Standard size that works well with vision transformers
    img_width = 224
    
    for ep_idx in range(num_episodes):
        print(f"Creating episode {episode_id + ep_idx}...")
        
        # Extract trajectory information
        states = []
        actions = []
        # Create separate image arrays for each camera
        images_dict = {}
        for cam_name in cameras:
            images_dict[cam_name] = []
        
        for i, timestep in enumerate(tqdm(trajectory_data, desc=f"Processing episode {ep_idx}")):
            # Extract joint states and actions
            state_data = timestep['states']
            action_data = timestep['actions']
            
            # Combine all joint positions into state vector (26 dims total)
            state_vec = np.concatenate([
                state_data['left_arm']['qpos'],    # 7 dims
                state_data['left_hand']['qpos'],   # 6 dims  
                state_data['right_arm']['qpos'],   # 7 dims
                state_data['right_hand']['qpos']   # 6 dims
            ])
            
            # Add small noise for variation between episodes
            if ep_idx > 0:
                noise_scale = 0.01  # Small noise to create variation
                noise = np.random.normal(0, noise_scale, state_vec.shape)
                state_vec = state_vec + noise
            
            states.append(state_vec)
            
            # Combine all joint positions into action vector (26 dims + 2 gripper dims = 28 total)
            action_vec = np.concatenate([
                action_data['left_arm']['qpos'],   # 7 dims
                action_data['left_hand']['qpos'],  # 6 dims
                action_data['right_arm']['qpos'],  # 7 dims
                action_data['right_hand']['qpos']  # 6 dims
            ])
            
            # Add small noise for variation between episodes
            if ep_idx > 0:
                noise_scale = 0.01  # Small noise to create variation
                noise = np.random.normal(0, noise_scale, action_vec.shape)
                action_vec = action_vec + noise
            
            # Add dummy gripper states (2 dims) to reach 28 total action dims
            action_vec = np.concatenate([action_vec, [0.0, 0.0]])
            actions.append(action_vec)
            
            # Handle images - either real or dummy
            if image_dir:
                # Load real images
                colors = timestep.get('colors', {})
                for j, cam_name in enumerate(cameras):
                    # Check for color_j first, then fall back to color_0 if only one camera available
                    color_key = f'color_{j}'
                    fallback_key = 'color_0'  # Use color_0 for both cameras if only one available
                    
                    if color_key in colors:
                        img_path = os.path.join(image_dir, colors[color_key])
                    elif fallback_key in colors and j == 1:  # For right camera, use left camera image
                        img_path = os.path.join(image_dir, colors[fallback_key])
                        print(f"Using color_0 image for camera {cam_name} (right camera)")
                    else:
                        img_path = None
                    
                    if img_path and os.path.exists(img_path):
                        # Load and resize real image
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                        img = cv2.resize(img, (img_width, img_height))  # Resize to consistent size
                        
                        # Convert to float and normalize to [0, 1] range
                        img = img.astype(np.float32) / 255.0
                        
                        # Convert from HWC to CHW format (channels first) for ACT
                        img = np.transpose(img, (2, 0, 1))  # (3, height, width)
                        
                        images_dict[cam_name].append(img)
                        if j == 0:
                            print(f"Loaded real image for camera {cam_name}: {img_path}")
                        else:
                            print(f"Using same image for camera {cam_name}: {img_path}")
                    else:
                        if img_path:
                            print(f"Warning: Image {img_path} not found, using dummy image for camera {cam_name}")
                        else:
                            print(f"Warning: No image data for camera {cam_name} at timestep {i}, using dummy image")
                        # Fallback to dummy image
                        dummy_img = _create_dummy_image(ep_idx, j, i, img_height, img_width)
                        images_dict[cam_name].append(dummy_img)
            else:
                # Create dummy RGB images for each camera
                for cam_idx, cam_name in enumerate(cameras):
                    dummy_img = _create_dummy_image(ep_idx, cam_idx, i, img_height, img_width)
                    images_dict[cam_name].append(dummy_img)
        
        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        
        # Convert images to numpy arrays for each camera
        for cam_name in cameras:
            images_dict[cam_name] = np.array(images_dict[cam_name], dtype=np.float32)
        
        # Save to HDF5 format
        current_episode_id = episode_id + ep_idx
        hdf5_path = os.path.join(output_dir, f'processed_episode_{current_episode_id}.hdf5')
        
        with h5py.File(hdf5_path, 'w') as f:
            # Save trajectory data
            f.create_dataset('observation.state', data=states)
            f.create_dataset('qpos_action', data=actions)
            
            # Save images for each camera in CHW format
            for cam_name in cameras:
                f.create_dataset(f'observation.image.{cam_name}', data=images_dict[cam_name])
            
            # Add metadata
            f.attrs['sim'] = True  # Set to True for simulation data
            f.attrs['episode_id'] = current_episode_id
            f.attrs['num_timesteps'] = num_timesteps
            f.attrs['state_dim'] = states.shape[1]
            f.attrs['action_dim'] = actions.shape[1]
            
            # Save task information
            task_info = data.get('text', {})
            f.attrs['task_goal'] = task_info.get('goal', 'Unknown task')
            f.attrs['task_desc'] = task_info.get('desc', 'No description')
            f.attrs['task_steps'] = task_info.get('steps', 'No steps')
        
        print(f"✓ Created episode {current_episode_id}: {hdf5_path}")
        print(f"  State shape: {states.shape}")
        print(f"  Action shape: {actions.shape}")
        for cam_name in cameras:
            print(f"  {cam_name} camera images shape: {images_dict[cam_name].shape}")
    
    print(f"Successfully converted to {num_episodes} HDF5 episodes")
    return output_dir

def _create_dummy_image(ep_idx, cam_idx, timestep_idx, img_height, img_width):
    """Create a dummy RGB image with consistent shape"""
    # Use different base colors for different cameras and episodes
    base_color = 50 + (ep_idx * 30 + cam_idx * 20 + timestep_idx * 2) % 150
    
    # Generate RGB image with shape (height, width, 3)
    dummy_img = np.random.randint(
        base_color, 
        min(255, base_color + 50), 
        (img_height, img_width, 3), 
        dtype=np.uint8
    )
    
    # Convert to float and normalize to [0, 1] range
    dummy_img = dummy_img.astype(np.float32) / 255.0
    
    # Convert from HWC to CHW format (channels first) for ACT
    dummy_img = np.transpose(dummy_img, (2, 0, 1))  # (3, height, width)
    
    return dummy_img

def convert_json_with_real_images(json_file, image_dir, output_dir, episode_id=0, cameras=['left', 'right'], num_episodes=5):
    """
    Convert JSON trajectory data to HDF5 format using real images.
    Creates multiple episodes from the same trajectory data.
    
    Args:
        json_file: Path to the JSON data file
        image_dir: Directory containing the actual images referenced in JSON
        output_dir: Directory to save the processed HDF5 files
        episode_id: Starting episode ID number
        cameras: List of camera names
        num_episodes: Number of episodes to create
    """
    
    # Load JSON data
    print(f"Loading data from {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    trajectory_data = data['data']
    num_timesteps = len(trajectory_data)
    
    print(f"Converting {num_timesteps} timesteps with real images...")
    print(f"Creating {num_episodes} episodes for ACT training...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for ep_idx in range(num_episodes):
        print(f"Creating episode {episode_id + ep_idx}...")
        
        # Extract trajectory information
        states = []
        actions = []
        images = {cam_name: [] for cam_name in cameras}
        
        for i, timestep in enumerate(tqdm(trajectory_data, desc=f"Processing episode {ep_idx}")):
            # Extract joint states and actions (same as above)
            state_data = timestep['states']
            action_data = timestep['actions']
            
            state_vec = np.concatenate([
                state_data['left_arm']['qpos'],
                state_data['left_hand']['qpos'],
                state_data['right_arm']['qpos'],
                state_data['right_hand']['qpos']
            ])
            
            # Add small noise for variation between episodes
            if ep_idx > 0:
                noise_scale = 0.01
                noise = np.random.normal(0, noise_scale, state_vec.shape)
                state_vec = state_vec + noise
            
            states.append(state_vec)
            
            action_vec = np.concatenate([
                action_data['left_arm']['qpos'],
                action_data['left_hand']['qpos'],
                action_data['right_arm']['qpos'],
                action_data['right_hand']['qpos']
            ])
            
            # Add small noise for variation between episodes
            if ep_idx > 0:
                noise_scale = 0.01
                noise = np.random.normal(0, noise_scale, action_vec.shape)
                action_vec = action_vec + noise
            
            action_vec = np.concatenate([action_vec, [0.0, 0.0]])  # Dummy gripper states
            actions.append(action_vec)
            
            # Load real images if available
            colors = timestep.get('colors', {})
            for j, cam_name in enumerate(cameras):
                color_key = f'color_{j}'
                if color_key in colors:
                    img_path = os.path.join(image_dir, colors[color_key])
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                        images[cam_name].append(img)
                    else:
                        # Fallback to dummy image
                        print(f"Warning: Image {img_path} not found, using dummy image")
                        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                        images[cam_name].append(dummy_img)
                else:
                    # Use dummy image if no real image specified
                    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    images[cam_name].append(dummy_img)
        
        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        
        # Save to HDF5 format
        current_episode_id = episode_id + ep_idx
        hdf5_path = os.path.join(output_dir, f'processed_episode_{current_episode_id}.hdf5')
        
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('observation.state', data=states)
            f.create_dataset('qpos_action', data=actions)
            
            for cam_name in cameras:
                images_array = np.stack(images[cam_name], axis=0)
                f.create_dataset(f'observation.image.{cam_name}', data=images_array)
            
            # Add metadata
            f.attrs['sim'] = True
            f.attrs['episode_id'] = current_episode_id
            f.attrs['num_timesteps'] = num_timesteps
            f.attrs['state_dim'] = states.shape[1]
            f.attrs['action_dim'] = actions.shape[1]
            
            task_info = data.get('text', {})
            f.attrs['task_goal'] = task_info.get('goal', 'Unknown task')
            f.attrs['task_desc'] = task_info.get('desc', 'No description')
            f.attrs['task_steps'] = task_info.get('steps', 'No steps')
        
        print(f"✓ Created episode {current_episode_id}: {hdf5_path}")
    
    print(f"Successfully converted to {num_episodes} HDF5 episodes")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Convert JSON trajectory data to ACT HDF5 format')
    parser.add_argument('--json_file', type=str, required=True, help='Path to JSON data file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for HDF5 files')
    parser.add_argument('--image_dir', type=str, help='Directory containing images (optional)')
    parser.add_argument('--episode_id', type=int, default=0, help='Starting episode ID number')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to create (default: 5)')
    parser.add_argument('--cameras', nargs='+', default=['left', 'right'], help='Camera names')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_file):
        print(f"Error: JSON file {args.json_file} not found!")
        return
    
    if args.image_dir and os.path.exists(args.image_dir):
        # Use real images if image directory is provided and exists
        convert_json_to_hdf5(
            args.json_file, 
            args.output_dir, 
            args.episode_id, 
            args.cameras,
            args.num_episodes,
            args.image_dir  # Pass image_dir to the unified function
        )
    else:
        # Use dummy images
        if args.image_dir:
            print(f"Warning: Image directory {args.image_dir} not found, using dummy images")
        convert_json_to_hdf5(
            args.json_file, 
            args.output_dir, 
            args.episode_id, 
            args.cameras,
            args.num_episodes
            # image_dir defaults to None for dummy images
        )

if __name__ == "__main__":
    main() 