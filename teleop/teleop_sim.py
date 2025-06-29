from isaacgym import gymapi
from isaacgym import gymutil

import math
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import shared_memory, Array, Lock
import threading
import time
import yaml

from teleop.open_television.tv_wrapper import TeleVisionWrapper
from teleop.robot_control.robot_arm_sim import H1_2_ArmController
from teleop.robot_control.robot_arm_ik import H1_2_ArmIK
from teleop.robot_control.robot_hand_inspire_sim import Inspire_Controller

class SimPlayer:
    def __init__(self, dt=1/60):
        self.dt = dt
        self.head_mat = None
        self.left_wrist_mat = None
        self.right_wrist_mat = None
        self.left_hand_pos = None
        self.right_hand_pos = None

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

        # Load bearing asset
        bearing_asset_options = gymapi.AssetOptions()
        bearing_asset_options.density = 10  # Adjust density as needed
        bearing_asset_options.fix_base_link = False
        bearing_asset = self.gym.load_asset(self.sim, "assets", "bearing.urdf", bearing_asset_options)

        # Load robot asset
        robot_asset_root = "assets"
        robot_asset_file = 'h1_inspire/urdf/h1_inspire.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        robot_asset = self.gym.load_asset(self.sim, robot_asset_root, robot_asset_file, asset_options)
        dof = self.gym.get_asset_dof_count(robot_asset)

        # set up the env grid
        num_envs = 1
        num_per_row = int(math.sqrt(num_envs))
        env_spacing = 1.25
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        np.random.seed(17)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

        # Create table
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0, 0, 1.0)  # Position the table at height 1.2
        table_pose.r = gymapi.Quat(0, 0, 0, 1)
        table_handle = self.gym.create_actor(self.env, table_asset, table_pose, 'table', 0)
        table_color = gymapi.Vec3(0.5, 0.5, 0.5)  # Gray color
        self.gym.set_rigid_body_color(self.env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, table_color)

        # Create bearing
        bearing_pose = gymapi.Transform()
        bearing_pose.p = gymapi.Vec3(-0.2, 0, 1.45)  # Position the bearing on top of the table
        bearing_pose.r = gymapi.Quat(0, 0, 0, 1)
        bearing_handle = self.gym.create_actor(self.env, bearing_asset, bearing_pose, 'bearing', 0)
        bearing_color = gymapi.Vec3(0.8, 0.8, 0.8)  # Light gray color
        self.gym.set_rigid_body_color(self.env, bearing_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, bearing_color)

        # robot
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.8, 0, 1.1)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.robot_handle = self.gym.create_actor(self.env, robot_asset, pose, 'robot', 1, 1)
        self.gym.set_actor_dof_states(self.env, self.robot_handle, np.zeros(dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)

        # create default viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()
        cam_pos = gymapi.Vec3(1, 1, 2)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # Setup camera parameters for VR view
        self.cam_lookat_offset = np.array([1, 0, 0])
        self.left_cam_offset = np.array([0, 0.033, 0])
        self.right_cam_offset = np.array([0, -0.033, 0])
        # self.cam_pos = np.array([-0.8, 0, 1.1])  # Robot's base position
        self.cam_pos = np.array([-0.4, 0, 1.4])

        # Create left camera for VR
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        self.left_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.left_camera_handle,
                                    self.env,
                                    gymapi.Vec3(*(self.cam_pos + self.left_cam_offset)),
                                    gymapi.Vec3(*(self.cam_pos + self.left_cam_offset + self.cam_lookat_offset)))

        # Create right camera for VR
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        self.right_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.right_camera_handle,
                                    self.env,
                                    gymapi.Vec3(*(self.cam_pos + self.right_cam_offset)),
                                    gymapi.Vec3(*(self.cam_pos + self.right_cam_offset + self.cam_lookat_offset)))

        # Store handles for later use
        self.table_handle = table_handle
        self.bearing_handle = bearing_handle

    def step(self, qpos, head_rmat=None):
        states = np.zeros(qpos.shape, dtype=gymapi.DofState.dtype)
        states['pos'] = qpos
        self.gym.set_actor_dof_states(self.env, self.robot_handle, states, gymapi.STATE_POS)

        # step the physics
        self.gym.simulate(self.sim)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

        # Update camera positions based on head rotation if provided
        if head_rmat is not None:
            curr_lookat_offset = self.cam_lookat_offset @ head_rmat.T
            curr_left_offset = self.left_cam_offset @ head_rmat.T
            curr_right_offset = self.right_cam_offset @ head_rmat.T

            self.gym.set_camera_location(self.left_camera_handle,
                                        self.env,
                                        gymapi.Vec3(*(self.cam_pos + curr_left_offset)),
                                        gymapi.Vec3(*(self.cam_pos + curr_left_offset + curr_lookat_offset)))
            self.gym.set_camera_location(self.right_camera_handle,
                                        self.env,
                                        gymapi.Vec3(*(self.cam_pos + curr_right_offset)),
                                        gymapi.Vec3(*(self.cam_pos + curr_right_offset + curr_lookat_offset)))

        # Get camera images
        left_image = self.gym.get_camera_image(self.sim, self.env, self.left_camera_handle, gymapi.IMAGE_COLOR)
        right_image = self.gym.get_camera_image(self.sim, self.env, self.right_camera_handle, gymapi.IMAGE_COLOR)
        
        # Reshape images to remove alpha channel
        left_image = left_image.reshape(left_image.shape[0], -1, 4)[..., :3]
        right_image = right_image.reshape(right_image.shape[0], -1, 4)[..., :3]

        return left_image, right_image

    def end(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

def main():
    # Initialize controllers
    arm_ctrl = H1_2_ArmController()
    arm_ik = H1_2_ArmIK()

    # Initialize hand controller
    left_hand_array = Array('d', 75, lock=True)
    right_hand_array = Array('d', 75, lock=True)
    dual_hand_data_lock = Lock()
    dual_hand_state_array = Array('d', 12, lock=False)
    dual_hand_action_array = Array('d', 12, lock=False)
    hand_ctrl = Inspire_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, 
                                 dual_hand_state_array, dual_hand_action_array)

    # Initialize television wrapper
    resolution = (720, 1280)
    crop_size_w = 0
    crop_size_h = 0
    resolution_cropped = (resolution[0]-crop_size_h, resolution[1]-2*crop_size_w)

    img_shape = (resolution_cropped[0], 2 * resolution_cropped[1], 3)
    img_height, img_width = resolution_cropped[:2]
    tv_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
    # Used to store the image data for VR
    img_array = np.ndarray((img_shape[0], img_shape[1], 3), dtype=np.uint8, buffer=tv_img_shm.buf)
    
    tv_wrapper = TeleVisionWrapper(binocular=True, 
                                 img_shape=img_shape,
                                 img_shm_name=tv_img_shm.name)

    # Initialize simulation
    player = SimPlayer(dt=1/30)
    
    try:
        user_input = input("Please enter 'r' to start teleoperation:\n")
        if user_input.lower() == 'r':
            arm_ctrl.speed_gradual_max()
            print("Starting teleoperation...")

            running = True
            while running:
                start_time = time.time()
                
                # Get pose data from VisionPro
                head_rmat, left_wrist, right_wrist, left_hand, right_hand = tv_wrapper.get_data()

                # Send hand skeleton data to hand controller
                left_hand_array[:] = left_hand.flatten()
                right_hand_array[:] = right_hand.flatten()

                # Get current state data
                current_lr_arm_q = arm_ctrl.get_current_dual_arm_q()
                current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

                # Solve IK
                sol_q, sol_tauff = arm_ik.solve_ik(left_wrist, right_wrist, current_lr_arm_q, current_lr_arm_dq)
                
                # Get hand states from the controller
                with dual_hand_data_lock:
                    left_hand_action = dual_hand_action_array[:6]
                    right_hand_action = dual_hand_action_array[6:]

                # Combine arm and hand actions into full robot state
                full_qpos = np.zeros(51)  # Total DOFs for H1 robot
                # Left arm
                full_qpos[13:20] = sol_q[:7]
                # Left hand (inspire hand)
                full_qpos[20:26] = left_hand_action
                # Right arm
                full_qpos[32:39] = sol_q[7:]
                # Right hand (inspire hand)
                full_qpos[39:45] = right_hand_action

                # Update simulation and get camera images
                left_img, right_img = player.step(full_qpos, head_rmat)
                
                # Combine left and right images and send to VR
                np.copyto(img_array, np.hstack((left_img, right_img)))

                # Maintain loop frequency
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1/30) - time_elapsed)
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("KeyboardInterrupt, exiting program...")
    finally:
        arm_ctrl.ctrl_dual_arm_go_home()
        arm_ctrl.cleanup()
        player.end()
        tv_img_shm.close()
        tv_img_shm.unlink()
        print("Exiting program...")
        exit(0)

if __name__ == "__main__":
    main() 

