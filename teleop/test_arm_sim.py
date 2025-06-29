import numpy as np
import time
from isaacgym import gymapi
from isaacgym import gymutil
from multiprocessing import Array, Lock

from teleop.robot_control.robot_arm_sim import H1_2_ArmController
from teleop.robot_control.robot_hand_inspire_sim import Inspire_Controller
from teleop.robot_control.robot_arm_ik import H1_2_ArmIK

def create_sim():
    # Initialize gym
    gym = gymapi.acquire_gym()

    # Configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1/60
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

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()

    # Add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.distance = 0.0
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    # Load robot asset
    robot_asset_root = "assets"
    robot_asset_file = 'h1_inspire/urdf/h1_inspire.urdf'
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    robot_asset = gym.load_asset(sim, robot_asset_root, robot_asset_file, asset_options)
    dof = gym.get_asset_dof_count(robot_asset)

    # Create environment
    env_spacing = 1.25
    env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    # Create robot
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(-0.8, 0, 1.1)
    pose.r = gymapi.Quat(0, 0, 0, 1)
    robot_handle = gym.create_actor(env, robot_asset, pose, 'robot', 1, 1)
    gym.set_actor_dof_states(env, robot_handle, np.zeros(dof, gymapi.DofState.dtype),
                            gymapi.STATE_ALL)

    # Create viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()
    cam_pos = gymapi.Vec3(1, 1, 2)
    cam_target = gymapi.Vec3(0, 0, 1)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    return gym, sim, env, robot_handle, viewer

def test_arm_and_hand_movement():
    # Create simulation
    gym, sim, env, robot_handle, viewer = create_sim()
    
    # Initialize controllers
    arm_ctrl = H1_2_ArmController()
    
    # Initialize hand arrays and controller
    left_hand_array = Array('d', 75, lock=True)
    right_hand_array = Array('d', 75, lock=True)
    dual_hand_data_lock = Lock()
    dual_hand_state_array = Array('d', 12, lock=False)
    dual_hand_action_array = Array('d', 12, lock=False)
    hand_ctrl = Inspire_Controller(left_hand_array, right_hand_array, dual_hand_data_lock,
                                 dual_hand_state_array, dual_hand_action_array)
    
    try:
        # Test positions for the arms
        test_positions = [
            # Home position
            np.zeros(14),
            # Left arm up
            np.array([0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            # Right arm up
            np.array([0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0]),
            # Both arms up
            np.array([0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0]),
            # Complex pose
            np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
        ]

        # Test hand movements
        test_hand_positions = [
            # Open hands
            np.ones(75),
            # Closed hands
            np.zeros(75),
            # Half-closed hands
            np.ones(75) * 0.5
        ]

        print("Starting arm and hand movement test...")
        print("Press 'q' to quit")
        
        # Move through test positions
        for pos in test_positions:
            print(f"\nMoving arms to position: {pos}")
            arm_ctrl.ctrl_dual_arm(pos, np.zeros(14))  # Zero torque for position control
            
            # Test hand movements for each arm position
            for hand_pos in test_hand_positions:
                print(f"Moving hands to position: {hand_pos[0]}")
                left_hand_array[:] = hand_pos
                right_hand_array[:] = hand_pos
                
                # Wait for movement to complete
                for _ in range(100):  # Simulate for about 1.5 seconds
                    # Get current state
                    current_q = arm_ctrl.get_current_dual_arm_q()
                    current_dq = arm_ctrl.get_current_dual_arm_dq()
                    
                    # Get current hand state
                    hand_state = hand_ctrl.get_current_hand_state()
                    
                    # Create full robot state
                    full_qpos = np.zeros(51)  # Total DOFs for H1 robot
                    # Left arm
                    full_qpos[13:20] = current_q[:7]
                    # Left hand (inspire hand)
                    full_qpos[20:26] = hand_state[:6]
                    # Right arm
                    full_qpos[32:39] = current_q[7:]
                    # Right hand (inspire hand)
                    full_qpos[39:45] = hand_state[6:]
                    
                    # Update simulation
                    gym.set_actor_dof_states(env, robot_handle, 
                                           np.array(full_qpos, dtype=gymapi.DofState.dtype),
                                           gymapi.STATE_POS)
                    
                    # Step simulation
                    gym.simulate(sim)
                    gym.step_graphics(sim)
                    gym.draw_viewer(viewer, sim, True)
                    gym.sync_frame_time(sim)
                    
                    # Check for quit
                    if gym.query_viewer_has_closed(viewer):
                        break
                    
                    time.sleep(1/60)  # Maintain 60Hz simulation rate
                
                # Check for quit
                if gym.query_viewer_has_closed(viewer):
                    break
            
            # Check for quit
            if gym.query_viewer_has_closed(viewer):
                break

    except KeyboardInterrupt:
        print("Test interrupted by user")
    finally:
        # Clean up
        arm_ctrl.ctrl_dual_arm_go_home()
        arm_ctrl.cleanup()
        hand_ctrl.cleanup()
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)
        print("Test completed")

def test_ik_with_fake_data():
    # Create simulation
    gym, sim, env, robot_handle, viewer = create_sim()
    
    # Initialize controllers
    arm_ctrl = H1_2_ArmController()
    arm_ik = H1_2_ArmIK()
    
    # Initialize hand arrays and controller
    left_hand_array = Array('d', 75, lock=True)
    right_hand_array = Array('d', 75, lock=True)
    dual_hand_data_lock = Lock()
    dual_hand_state_array = Array('d', 12, lock=False)
    dual_hand_action_array = Array('d', 12, lock=False)
    hand_ctrl = Inspire_Controller(left_hand_array, right_hand_array, dual_hand_data_lock,
                                 dual_hand_state_array, dual_hand_action_array)
    
    try:
        print("Starting IK test with fake TV wrapper data...")
        print("Press 'q' to quit")
        
        # Helper function to create transformation matrix
        def create_transform_matrix(position, rotation=None):
            if rotation is None:
                rotation = np.eye(3)
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = position
            return transform
        
        # Test positions for the wrists
        test_wrist_positions = [
            # Home position
            {
                'head_rmat': np.eye(3),
                'left_wrist': create_transform_matrix(np.array([0.3, 0.2, 0.5])),
                'right_wrist': create_transform_matrix(np.array([0.3, -0.2, 0.5])),
                'left_hand': np.ones(75),  # Open hand
                'right_hand': np.ones(75)  # Open hand
            },
            # Arms forward
            {
                'head_rmat': np.eye(3),
                'left_wrist': create_transform_matrix(np.array([0.5, 0.2, 0.3])),
                'right_wrist': create_transform_matrix(np.array([0.5, -0.2, 0.3])),
                'left_hand': np.zeros(75),  # Closed hand
                'right_hand': np.zeros(75)  # Closed hand
            },
            # Arms up
            {
                'head_rmat': np.eye(3),
                'left_wrist': create_transform_matrix(np.array([0.3, 0.2, 0.7])),
                'right_wrist': create_transform_matrix(np.array([0.3, -0.2, 0.7])),
                'left_hand': np.ones(75) * 0.5,  # Half-closed hand
                'right_hand': np.ones(75) * 0.5  # Half-closed hand
            }
        ]
        
        for test_data in test_wrist_positions:
            print(f"\nTesting wrist positions:")
            print(f"Left wrist position: {test_data['left_wrist'][:3, 3]}")
            print(f"Right wrist position: {test_data['right_wrist'][:3, 3]}")
            
            # Send hand skeleton data to hand controller
            left_hand_array[:] = test_data['left_hand']
            right_hand_array[:] = test_data['right_hand']
            
            # Get current state data
            current_lr_arm_q = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()
            
            # Solve IK
            sol_q, sol_tauff = arm_ik.solve_ik(test_data['left_wrist'], test_data['right_wrist'], 
                                             current_lr_arm_q, current_lr_arm_dq)
            
            print(f"Solution found: {sol_q is not None}")
            if sol_q is not None:
                print(f"Joint angles: {sol_q}")
                print(f"Torque: {sol_tauff}")
                
                # Get hand states
                with dual_hand_data_lock:
                    left_hand_action = dual_hand_action_array[:6]
                    right_hand_action = dual_hand_action_array[-6:]
                
                # Create full robot state
                full_qpos = np.zeros(51)  # Total DOFs for H1 robot
                # Left arm
                full_qpos[13:20] = sol_q[:7]
                # Left hand (inspire hand)
                full_qpos[20:26] = left_hand_action
                # Right arm
                full_qpos[32:39] = sol_q[7:]
                # Right hand (inspire hand)
                full_qpos[39:45] = right_hand_action
                
                # Update simulation
                gym.set_actor_dof_states(env, robot_handle, 
                                       np.array(full_qpos, dtype=gymapi.DofState.dtype),
                                       gymapi.STATE_POS)
                
                # Simulate for a while to see the movement
                for _ in range(100):  # Simulate for about 1.5 seconds
                    # Step simulation
                    gym.simulate(sim)
                    gym.step_graphics(sim)
                    gym.draw_viewer(viewer, sim, True)
                    gym.sync_frame_time(sim)
                    
                    # Check for quit
                    if gym.query_viewer_has_closed(viewer):
                        break
                    
                    time.sleep(1/60)  # Maintain 60Hz simulation rate
            
            # Check for quit
            if gym.query_viewer_has_closed(viewer):
                break
            
            # Wait a bit between positions
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        print("Test interrupted by user")
    finally:
        # Clean up
        arm_ctrl.ctrl_dual_arm_go_home()
        arm_ctrl.cleanup()
        hand_ctrl.cleanup()
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)
        print("Test completed")

if __name__ == "__main__":
    # test_arm_and_hand_movement()  # Comment out the original test
    test_ik_with_fake_data()  # Run the new IK test 