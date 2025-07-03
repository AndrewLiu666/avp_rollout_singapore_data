#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import numpy as np
import argparse
import time
import pickle
import json
import cv2

# Add paths for imports
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent / "teleop"))

# IMPORTANT: Import Isaac Gym modules BEFORE PyTorch
from isaacgym import gymapi
from isaacgym import gymutil
import math

# Now import PyTorch and other modules
import torch
import torchvision.transforms as transforms

# Import ACT modules
from policy import ACTPolicy


class PolicyTester:
    def __init__(self, policy_path, stats_path, dt=1 / 30, data_dir=None, json_file=None):
        self.dt = dt
        self.data_dir = data_dir
        self.json_file = json_file
        self.real_trajectory_data = None
        self.current_timestep = 0

        # Load real trajectory data if provided
        if json_file and os.path.exists(json_file):
            self.load_real_trajectory(json_file)

        # Load policy and stats
        self.load_policy(policy_path, stats_path)

        # initialize gym
        self.gym = gymapi.acquire_gym()

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = dt
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.max_gpu_contact_pairs = 8388608
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        # Create environment (same as replay_data_test.py)
        self.setup_environment()

        # Initialize action history for chunked predictions
        self.action_queue = []
        self.current_chunk_idx = 0

        # Store trajectory comparison data
        self.simulated_states = []
        self.predicted_actions = []
        self.real_actions = []

    def load_real_trajectory(self, json_file):
        """Load the real trajectory data for comparison"""
        print(f"Loading real trajectory data from {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.real_trajectory_data = data['data']
        print(f"Loaded {len(self.real_trajectory_data)} timesteps from real trajectory")

    def load_policy(self, policy_path, stats_path):
        """Load the trained ACT policy and normalization stats"""
        print(f"Loading policy from {policy_path}")
        print(f"Loading stats from {stats_path}")

        # Load normalization stats
        with open(stats_path, 'rb') as f:
            self.norm_stats = pickle.load(f)

        if "config" in self.norm_stats:
            policy_config = self.norm_stats["config"]
            print("Loaded policy config from stats.")
        else:
            print("Warning: 'config' not found in stats. Falling back to default (may cause mismatches).")
            policy_config = {
                'lr': 1e-5,
                # 14 -> 28
                # 'num_queries': 50,
                'num_queries': 32,
                'kl_weight': 10,
                # 'hidden_dim': 256,
                # 14 -> 28
                'hidden_dim': 512,
                # 14 -> 28
                # 'dim_feedforward': 1024,
                'dim_feedforward': 3200,
                'lr_backbone': 1e-5,
                # 14 -> 28
                # 'backbone': 'dino_v2',
                'backbone': 'resnet18',
                'enc_layers': 4,
                'dec_layers': 7,
                'nheads': 8,
                'camera_names': ['agentview'],
                # 'camera_names': ['agentview', 'robot0_eye_in_hand', 'robot1_eye_in_hand'],
                # 14 -> 28
                # 'state_dim': 14,
                # 'action_dim': 14,
                'state_dim': 26,
                'action_dim': 28,
                'qpos_noise_std': 0.0,
            }

        # Create policy
        self.policy = ACTPolicy(policy_config)
        self.policy.cuda()

        # Load checkpoint
        checkpoint = torch.load(policy_path, map_location='cuda')
        # 14 -> 28
        self.policy.load_state_dict(checkpoint, strict=False)
        self.policy.eval()

        self.action_dim = self.policy.model.action_dim
        self.state_dim = self.policy.model.state_dim

        print("Policy loaded successfully!")

    def setup_environment(self):
        """Set up the Isaac Gym environment (same as replay_data_test.py)"""
        # Add ground
        plane_params = gymapi.PlaneParams()
        plane_params.distance = 0.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        # Load table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 0.8, 0.8, 0.1, table_asset_options)

        # Load a red cup asset
        cup_asset_options = gymapi.AssetOptions()
        cup_asset_options.density = 10
        cup_asset_options.fix_base_link = False
        cup_asset = self.gym.create_sphere(self.sim, 0.04, cup_asset_options)

        # Load robot asset
        robot_asset_root = "../assets"
        robot_asset_file = 'h1_inspire/urdf/h1_inspire.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        robot_asset = self.gym.load_asset(self.sim, robot_asset_root, robot_asset_file, asset_options)
        self.dof_count = self.gym.get_asset_dof_count(robot_asset)
        print(f"Robot DOF count: {self.dof_count}")

        # Set up environment
        num_envs = 1
        num_per_row = int(math.sqrt(num_envs))
        env_spacing = 1.25
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

        # Create table
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0, 0, 1.0)
        table_pose.r = gymapi.Quat(0, 0, 0, 1)
        table_handle = self.gym.create_actor(self.env, table_asset, table_pose, 'table', 0)
        table_color = gymapi.Vec3(0.6, 0.4, 0.2)
        self.gym.set_rigid_body_color(self.env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, table_color)

        # Create red cup
        cup_pose = gymapi.Transform()
        cup_pose.p = gymapi.Vec3(-0.2, 0, 1.2)
        cup_pose.r = gymapi.Quat(0, 0, 0, 1)
        cup_handle = self.gym.create_actor(self.env, cup_asset, cup_pose, 'red_cup', 0)
        cup_color = gymapi.Vec3(1.0, 0.0, 0.0)
        self.gym.set_rigid_body_color(self.env, cup_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, cup_color)

        # Create robot
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.8, 0, 1.1)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.robot_handle = self.gym.create_actor(self.env, robot_asset, pose, 'robot', 1, 1)

        # Initialize robot to a neutral pose
        neutral_qpos = np.zeros(self.dof_count)
        self.gym.set_actor_dof_states(self.env, self.robot_handle, np.zeros(self.dof_count, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)

        # Create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()
        cam_pos = gymapi.Vec3(1, 1, 2)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def get_robot_state(self):
        """Get current robot joint positions for the relevant joints"""
        # Get current joint states
        dof_states = self.gym.get_actor_dof_states(self.env, self.robot_handle, gymapi.STATE_POS)
        current_qpos = dof_states['pos']

        # Extract the 26 relevant joint positions (same as training data)
        # Left arm (7), left hand (6), right arm (7), right hand (6)
        state_vec = np.concatenate([
            current_qpos[13:20],  # Left arm
            current_qpos[20:26],  # Left hand
            current_qpos[32:39],  # Right arm
            current_qpos[39:45]  # Right hand
        ])
        print("Current state shape:", state_vec.shape)

        return state_vec.astype(np.float32)

    def load_real_images(self, timestep_idx=None):
        """Load real camera images for policy input"""
        if not self.data_dir or not self.real_trajectory_data:
            # Fallback to dummy images if no real data available
            return self.create_dummy_images()

        # Use current timestep if not specified
        if timestep_idx is None:
            timestep_idx = self.current_timestep

        # Check if timestep is valid
        if timestep_idx >= len(self.real_trajectory_data):
            print(
                f"Warning: Timestep {timestep_idx} beyond trajectory length {len(self.real_trajectory_data)}, using dummy images")
            return self.create_dummy_images()

        # Get color information from real trajectory
        timestep_data = self.real_trajectory_data[timestep_idx]
        colors = timestep_data.get('colors', {})

        # Target image size for ACT
        img_height, img_width = 224, 224
        camera_images = []

        for j, cam_name in enumerate(['agentview', 'robot0_eye_in_hand', 'robot1_eye_in_hand']):
            color_key = f'color_{j}'
            fallback_key = 'color_0'  # fallback

            if color_key in colors and self.data_dir:
                img_path = os.path.join(self.data_dir, colors[color_key])
            elif fallback_key in colors and j > 0 and self.data_dir:
                img_path = os.path.join(self.data_dir, colors[fallback_key])
                if timestep_idx < 5:
                    print(f"Using color_0 image for {cam_name}")
            else:
                img_path = None

            if img_path and os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_width, img_height))
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))  # CHW
                camera_images.append(img)
            else:
                if img_path and timestep_idx < 5:
                    print(f"Warning: Image {img_path} not found for {cam_name}")
                dummy_img = np.random.uniform(0.0, 1.0, (3, img_height, img_width)).astype(np.float32)
                camera_images.append(dummy_img)

        # Stack camera images and convert to tensor
        images_array = np.stack(camera_images, axis=0)  # (num_cameras, channels, height, width)
        return torch.from_numpy(images_array).cuda()

    def create_dummy_images(self):
        """Create dummy camera images for policy input (fallback)"""
        # Create dummy RGB images (2 cameras, 3 channels, 224x224)
        img_height, img_width = 224, 224
        num_cameras = 2

        # Create some simple dummy images
        dummy_images = np.random.uniform(0.0, 1.0, (num_cameras, 3, img_height, img_width)).astype(np.float32)

        return torch.from_numpy(dummy_images).cuda()

    def predict_action(self, current_state):
        """Use the policy to predict the next action"""
        with torch.no_grad():
            # Normalize state
            state_normalized = (current_state - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
            state_tensor = torch.from_numpy(state_normalized).float().cuda().unsqueeze(0)  # Add batch dimension

            # Load real images for current timestep
            image_tensor = self.load_real_images().unsqueeze(0)  # Add batch dimension

            # Get action prediction (returns a chunk of actions)
            action_chunk = self.policy(state_tensor, image_tensor)  # No actions provided = inference mode

            # Denormalize actions
            action_chunk = action_chunk.cpu().numpy()[0]  # Remove batch dimension
            action_chunk = action_chunk * self.norm_stats["action_std"] + self.norm_stats["action_mean"]

            return action_chunk

    def get_real_action_at_timestep(self, timestep_idx):
        """Get the real action from trajectory data at given timestep"""
        if not self.real_trajectory_data or timestep_idx >= len(self.real_trajectory_data):
            return None

        timestep_data = self.real_trajectory_data[timestep_idx]
        action_data = timestep_data['actions']

        # Combine all joint positions into action vector (same as conversion script)
        real_action = np.concatenate([
            action_data['left_arm']['qpos'],  # 7 dims
            action_data['left_hand']['qpos'],  # 6 dims
            action_data['right_arm']['qpos'],  # 7 dims
            action_data['right_hand']['qpos']  # 6 dims
        ])

        # Add dummy gripper states to match model output
        # real_action = np.concatenate([real_action, [0.0, 0.0]])  # 28 dims total
        # Pad or trim real_action to match model's action_dim
        if real_action.shape[0] < self.action_dim:
            padding = np.zeros(self.action_dim - real_action.shape[0], dtype=np.float32)
            real_action = np.concatenate([real_action, padding])
        else:
            real_action = real_action[:self.action_dim]

        return real_action

    def convert_action_to_robot_qpos(self, action_vec):
        """Convert 28-dim action vector to full robot joint positions"""
        full_qpos = np.zeros(self.dof_count)

        # Extract the 26 joint actions (ignore last 2 gripper dims)
        joint_actions = action_vec[:26]

        # Map to robot DOF indices
        full_qpos[13:20] = joint_actions[0:7]  # Left arm
        full_qpos[20:26] = joint_actions[7:13]  # Left hand
        full_qpos[32:39] = joint_actions[13:20]  # Right arm
        full_qpos[39:45] = joint_actions[20:26]  # Right hand

        return full_qpos

    def step_simulation(self, qpos):
        """Step the simulation with given joint positions"""
        states = np.zeros(self.dof_count, dtype=gymapi.DofState.dtype)
        states['pos'] = qpos
        self.gym.set_actor_dof_states(self.env, self.robot_handle, states, gymapi.STATE_POS)

        # Step physics
        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

        return self.gym.query_viewer_has_closed(self.viewer)

    def run_policy_control(self, max_steps=1000):
        """Run the policy control loop with real image inputs and trajectory comparison"""
        print("Starting policy control with real images...")
        print("The robot will be controlled by the trained ACT policy using real images.")
        print("Press ESC or close viewer to stop.")

        # Determine max steps based on real trajectory if available
        if self.real_trajectory_data:
            max_steps = min(max_steps, len(self.real_trajectory_data))
            print(f"Using real trajectory length: {max_steps} steps")

        try:
            for step in range(max_steps):
                start_time = time.time()

                # Update current timestep for image loading
                self.current_timestep = step

                # Get current robot state
                current_state = self.get_robot_state()
                self.simulated_states.append(current_state.copy())

                # Use policy to predict actions using real images
                if len(self.action_queue) == 0 or self.current_chunk_idx >= len(self.action_queue):
                    # Get new action from policy (single action since num_queries=1)
                    action_chunk = self.predict_action(current_state)
                    if len(action_chunk.shape) == 1:
                        # Single action prediction
                        self.action_queue = [action_chunk]
                        print(f"Step {step}: Generated single action prediction")
                    else:
                        # Multiple action predictions (chunk)
                        self.action_queue = action_chunk
                        print(f"Step {step}: Generated action chunk of {len(action_chunk)} actions")
                    self.current_chunk_idx = 0

                # Get next action from queue
                next_action = self.action_queue[self.current_chunk_idx]
                self.current_chunk_idx += 1
                self.predicted_actions.append(next_action.copy())

                # Get real action for comparison if available
                real_action = self.get_real_action_at_timestep(step)
                if real_action is not None:
                    self.real_actions.append(real_action.copy())

                    # Compute and print action differences
                    # action_diff = np.abs(next_action[:26] - real_action[:26])  # Compare only joint actions
                    action_diff = np.abs(next_action[:self.action_dim] - real_action[:self.action_dim])

                    max_diff = np.max(action_diff)
                    mean_diff = np.mean(action_diff)

                    print(f"Step {step}: Action comparison - Max diff: {max_diff:.4f}, Mean diff: {mean_diff:.4f}")

                    # Print detailed comparison for first few steps
                    if step < 5:
                        print(f"  Predicted action (first 10): {next_action[:10]}")
                        print(f"  Real action (first 10):      {real_action[:10]}")
                        print(f"  Differences (first 10):      {action_diff[:10]}")

                # Convert action to robot joint positions
                target_qpos = self.convert_action_to_robot_qpos(next_action)

                # Step simulation
                should_close = self.step_simulation(target_qpos)
                if should_close:
                    break

                # Print progress
                if step % 10 == 0:
                    print(f"Step {step}/{max_steps} - Policy controlling robot with real images")

                # Control timing
                elapsed = time.time() - start_time
                sleep_time = max(0, self.dt - elapsed)
                # time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("Control interrupted by user")

        # Print final trajectory comparison summary
        self.print_trajectory_summary()

        print("Policy control completed!")

    def print_trajectory_summary(self):
        """Print summary of trajectory comparison"""
        print("\n" + "=" * 60)
        print("TRAJECTORY COMPARISON SUMMARY")
        print("=" * 60)

        if not self.real_actions:
            print("No real trajectory data available for comparison")
            return

        # Convert to numpy arrays for analysis
        predicted_actions = np.array(self.predicted_actions)
        real_actions = np.array(self.real_actions[:len(self.predicted_actions)])  # Match lengths

        # Compute overall statistics
        action_diffs = np.abs(predicted_actions[:, :26] - real_actions[:, :26])  # Only joint actions

        max_diffs = np.max(action_diffs, axis=1)  # Max diff per timestep
        mean_diffs = np.mean(action_diffs, axis=1)  # Mean diff per timestep

        print(f"Number of compared timesteps: {len(action_diffs)}")
        print(f"Overall max difference: {np.max(max_diffs):.4f}")
        print(f"Overall mean difference: {np.mean(mean_diffs):.4f}")
        print(f"Standard deviation of differences: {np.std(mean_diffs):.4f}")

        # Per-joint analysis
        joint_names = ['L_arm_0', 'L_arm_1', 'L_arm_2', 'L_arm_3', 'L_arm_4', 'L_arm_5', 'L_arm_6',
                       'L_hand_0', 'L_hand_1', 'L_hand_2', 'L_hand_3', 'L_hand_4', 'L_hand_5',
                       'R_arm_0', 'R_arm_1', 'R_arm_2', 'R_arm_3', 'R_arm_4', 'R_arm_5', 'R_arm_6',
                       'R_hand_0', 'R_hand_1', 'R_hand_2', 'R_hand_3', 'R_hand_4', 'R_hand_5']

        print(f"\nPer-joint mean differences:")
        # for i, joint_name in enumerate(joint_names):
        #     joint_mean_diff = np.mean(action_diffs[:, i])
        #     print(f"  {joint_name}: {joint_mean_diff:.4f}")
        for i in range(action_diffs.shape[1]):
            joint_name = joint_names[i] if i < len(joint_names) else f"Joint_{i}"
            joint_mean_diff = np.mean(action_diffs[:, i])
            print(f"  {joint_name}: {joint_mean_diff:.4f}")

        print("=" * 60)

    def cleanup(self):
        """Clean up simulation resources"""
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


def main():
    parser = argparse.ArgumentParser(description='Test trained ACT policy in Isaac Gym simulation with real images')
    parser.add_argument('--policy_path', type=str, required=True, help='Path to trained policy checkpoint (.ckpt file)')
    parser.add_argument('--stats_path', type=str, required=True, help='Path to dataset stats (.pkl file)')
    parser.add_argument('--data_dir', type=str, help='Path to directory containing real images (e.g., data/colors)')
    parser.add_argument('--json_file', type=str, help='Path to JSON file with real trajectory data for comparison')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum simulation steps (default: 1000)')

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.policy_path):
        print(f"Error: Policy file {args.policy_path} not found!")
        return

    if not os.path.exists(args.stats_path):
        print(f"Error: Stats file {args.stats_path} not found!")
        return

    if args.data_dir and not os.path.exists(args.data_dir):
        print(f"Warning: Data directory {args.data_dir} not found! Will use dummy images.")
        args.data_dir = None

    if args.json_file and not os.path.exists(args.json_file):
        print(f"Warning: JSON file {args.json_file} not found! No trajectory comparison available.")
        args.json_file = None

    print("=" * 60)
    print("ACT POLICY SIMULATION TEST WITH REAL IMAGES")
    print("=" * 60)
    print(f"Policy: {args.policy_path}")
    print(f"Stats: {args.stats_path}")
    print(f"Data directory: {args.data_dir if args.data_dir else 'None (using dummy images)'}")
    print(f"JSON trajectory: {args.json_file if args.json_file else 'None (no comparison)'}")
    print(f"Max steps: {args.max_steps}")
    print("=" * 60)

    # Initialize tester with real data paths
    tester = PolicyTester(args.policy_path, args.stats_path, data_dir=args.data_dir, json_file=args.json_file)

    try:
        # Run policy control with real images
        tester.run_policy_control(args.max_steps)

        print("Test completed. Press Enter to exit...")
        input()

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tester.cleanup()
        print("Simulation ended.")


if __name__ == "__main__":
    main()
