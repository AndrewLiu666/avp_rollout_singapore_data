from isaacgym import gymapi
from isaacgym import gymutil

import math
import numpy as np
import json
import time
import cv2
import argparse
import os

class DataReplayer:
    def __init__(self, dt=1/30):
        self.dt = dt
        
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

        plane_params = gymapi.PlaneParams()
        plane_params.distance = 0.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        # Load table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 0.8, 0.8, 0.1, table_asset_options)

        # Load a red cup/bearing asset
        cup_asset_options = gymapi.AssetOptions()
        cup_asset_options.density = 10
        cup_asset_options.fix_base_link = False
        # Create a cylinder to represent the cup
        cup_asset = self.gym.create_sphere(self.sim, 0.04, cup_asset_options)  # Small sphere as cup

        # Load robot asset
        robot_asset_root = "assets"
        robot_asset_file = 'h1_inspire/urdf/h1_inspire.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        robot_asset = self.gym.load_asset(self.sim, robot_asset_root, robot_asset_file, asset_options)
        self.dof_count = self.gym.get_asset_dof_count(robot_asset)
        print(f"Robot DOF count: {self.dof_count}")

        # set up the env grid
        num_envs = 1
        num_per_row = int(math.sqrt(num_envs))
        env_spacing = 1.25
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

        # Create table
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0, 0, 1.0)  # Position the table at height 1.0
        table_pose.r = gymapi.Quat(0, 0, 0, 1)
        table_handle = self.gym.create_actor(self.env, table_asset, table_pose, 'table', 0)
        table_color = gymapi.Vec3(0.6, 0.4, 0.2)  # Brown color for table
        self.gym.set_rigid_body_color(self.env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, table_color)

        # Create red cup
        cup_pose = gymapi.Transform()
        cup_pose.p = gymapi.Vec3(-0.2, 0, 1.2)  # Position the cup on top of the table
        cup_pose.r = gymapi.Quat(0, 0, 0, 1)
        cup_handle = self.gym.create_actor(self.env, cup_asset, cup_pose, 'red_cup', 0)
        cup_color = gymapi.Vec3(1.0, 0.0, 0.0)  # Red color for cup
        self.gym.set_rigid_body_color(self.env, cup_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, cup_color)

        # robot
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.8, 0, 1.1)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.robot_handle = self.gym.create_actor(self.env, robot_asset, pose, 'robot', 1, 1)
        self.gym.set_actor_dof_states(self.env, self.robot_handle, np.zeros(self.dof_count, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)

        # create default viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()
        cam_pos = gymapi.Vec3(1, 1, 2)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # Store handles for later use
        self.table_handle = table_handle
        self.cup_handle = cup_handle

    def convert_data_to_robot_qpos(self, data_point):
        """Convert JSON data point to full robot joint positions"""
        full_qpos = np.zeros(self.dof_count)
        
        # Extract joint positions from data
        left_arm_qpos = np.array(data_point['states']['left_arm']['qpos'])
        right_arm_qpos = np.array(data_point['states']['right_arm']['qpos'])
        left_hand_qpos = np.array(data_point['states']['left_hand']['qpos'])
        right_hand_qpos = np.array(data_point['states']['right_hand']['qpos'])
        
        # Map to robot DOF indices based on H1 inspire robot structure
        # These indices are based on the h1_inspire.urdf joint ordering
        
        # Left arm joints (indices 13-19)
        if len(left_arm_qpos) == 7:
            full_qpos[13:20] = left_arm_qpos
            
        # Left hand joints (indices 20-25) 
        if len(left_hand_qpos) == 6:
            full_qpos[20:26] = left_hand_qpos
            
        # Right arm joints (indices 32-38)
        if len(right_arm_qpos) == 7:
            full_qpos[32:39] = right_arm_qpos
            
        # Right hand joints (indices 39-44)
        if len(right_hand_qpos) == 6:
            full_qpos[39:45] = right_hand_qpos
            
        return full_qpos

    def step(self, qpos):
        """Step the simulation with given joint positions"""
        states = np.zeros(self.dof_count, dtype=gymapi.DofState.dtype)
        states['pos'] = qpos
        self.gym.set_actor_dof_states(self.env, self.robot_handle, states, gymapi.STATE_POS)

        # step the physics
        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

        return self.gym.query_viewer_has_closed(self.viewer)

    def replay_data(self, data_file, playback_speed=1.0):
        """Replay the data from JSON file"""
        print(f"Loading data from {data_file}")
        
        with open(data_file, 'r') as f:
            dataset = json.load(f)
        
        data_points = dataset['data']
        print(f"Loaded {len(data_points)} data points")
        print(f"Task: {dataset['text']['goal']}")
        print(f"Description: {dataset['text']['desc']}")
        print(f"Steps: {dataset['text']['steps']}")
        
        try:
            print("Starting replay in 3 seconds...")
            time.sleep(3)
            
            for i, data_point in enumerate(data_points):
                start_time = time.time()
                
                # Convert data to robot joint positions
                qpos = self.convert_data_to_robot_qpos(data_point)
                
                # Step simulation
                should_close = self.step(qpos)
                if should_close:
                    break
                
                # Print progress
                if i % 10 == 0:
                    print(f"Frame {i}/{len(data_points)} - Progress: {i/len(data_points)*100:.1f}%")
                
                # Control playback speed
                elapsed = time.time() - start_time
                sleep_time = max(0, (self.dt / playback_speed) - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("Playback interrupted by user")

    def end(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

def main():
    parser = argparse.ArgumentParser(description='Replay teleoperation data in IsaacGym simulation')
    parser.add_argument('--data_file', type=str, required=True, help='Path to JSON data file')
    parser.add_argument('--playback_speed', type=float, default=1.0, help='Playback speed multiplier (default: 1.0)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_file):
        print(f"Error: Data file {args.data_file} not found!")
        return
    
    # Initialize replayer
    replayer = DataReplayer(dt=1/30)
    
    try:
        # Replay the data
        replayer.replay_data(args.data_file, args.playback_speed)
        
        print("Replay completed. Press Enter to exit...")
        input()
        
    except Exception as e:
        print(f"Error during replay: {e}")
    finally:
        replayer.end()
        print("Simulation ended.")

if __name__ == "__main__":
    main() 