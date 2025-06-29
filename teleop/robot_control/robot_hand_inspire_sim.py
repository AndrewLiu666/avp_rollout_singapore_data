import numpy as np
import threading
import time
from enum import IntEnum
from multiprocessing import Array, Lock
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType

# Constants
Inspire_Num_Motors = 6  # 6 motors per hand
inspire_tip_indices = [4, 9, 14, 19, 24]  # Indices for finger tips in hand skeleton data

class Inspire_Right_Hand_JointIndex(IntEnum):
    kRightHandPinky = 0
    kRightHandRing = 1
    kRightHandMiddle = 2
    kRightHandIndex = 3
    kRightHandThumbBend = 4
    kRightHandThumbRotation = 5

class Inspire_Left_Hand_JointIndex(IntEnum):
    kLeftHandPinky = 6
    kLeftHandRing = 7
    kLeftHandMiddle = 8
    kLeftHandIndex = 9
    kLeftHandThumbBend = 10
    kLeftHandThumbRotation = 11

class Inspire_Controller:
    def __init__(self, left_hand_array, right_hand_array, dual_hand_data_lock=None, 
                 dual_hand_state_array=None, dual_hand_action_array=None, fps=100.0):
        print("Initialize Inspire_Controller (Simulated)...")
        self.fps = fps
        
        # Initialize hand retargeting
        self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND_Unit_Test)
        
        # Control parameters
        self.control_dt = 1.0 / fps
        
        # Current state
        self.left_hand_state = np.zeros(Inspire_Num_Motors)
        self.right_hand_state = np.zeros(Inspire_Num_Motors)
        
        # Target state
        self.left_hand_target = np.ones(Inspire_Num_Motors)  # Start fully open
        self.right_hand_target = np.ones(Inspire_Num_Motors)  # Start fully open
        
        # Thread control
        self.ctrl_lock = threading.Lock()
        self.running = True
        
        # Start control thread
        self.control_thread = threading.Thread(target=self._control_process, 
                                            args=(left_hand_array, right_hand_array,
                                                  dual_hand_data_lock, dual_hand_state_array,
                                                  dual_hand_action_array))
        self.control_thread.daemon = True
        self.control_thread.start()
        
        print("Initialize Inspire_Controller (Simulated) OK!\n")

    def _normalize_joint_value(self, val, idx):
        """Normalize joint values to [0, 1] range based on joint type"""
        if idx <= 3:  # Fingers (pinky, ring, middle, index)
            return np.clip((1.7 - val) / 1.7, 0.0, 1.0)
        elif idx == 4:  # Thumb bend
            return np.clip((0.5 - val) / 0.5, 0.0, 1.0)
        else:  # Thumb rotation
            return np.clip((1.3 - val) / 1.4, 0.0, 1.0)

    def _control_process(self, left_hand_array, right_hand_array, 
                        dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array):
        """Control thread that updates the simulated hand state"""
        while self.running:
            start_time = time.time()
            
            # Get hand skeleton data
            left_hand_mat = np.array(left_hand_array[:]).reshape(25, 3).copy()
            right_hand_mat = np.array(right_hand_array[:]).reshape(25, 3).copy()
            
            # Process hand data if available
            if not np.all(right_hand_mat == 0.0) and not np.all(left_hand_mat[4] == np.array([-1.13, 0.3, 0.15])):
                # Get finger tip positions
                ref_left_value = left_hand_mat[inspire_tip_indices]
                ref_right_value = right_hand_mat[inspire_tip_indices]
                
                # Use the same retargeting process as the real controller
                print("ref_left_value: ", ref_left_value)
                print("ref_right_value: ", ref_right_value)
                left_q_target = self.hand_retargeting.left_retargeting.retarget(ref_left_value)
                right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)
                print("left_q_target: ", left_q_target)
                print("right_q_target: ", right_q_target)

                # Update targets
                self.left_hand_target = left_q_target
                self.right_hand_target = right_q_target
            
            # # Update current state with smooth interpolation
            # alpha = 0.1  # Smoothing factor
            # self.left_hand_state = (1 - alpha) * self.left_hand_state + alpha * self.left_hand_target
            # self.right_hand_state = (1 - alpha) * self.right_hand_state + alpha * self.right_hand_target
            
            # Update shared arrays if provided
            if dual_hand_state_array is not None and dual_hand_action_array is not None:
                with dual_hand_data_lock:
                    # State: current joint positions
                    state_data = np.concatenate((self.left_hand_state, self.right_hand_state))
                    dual_hand_state_array[:] = state_data
                    
                    # Action: target joint positions
                    action_data = np.concatenate((self.left_hand_target, self.right_hand_target))
                    dual_hand_action_array[:] = action_data
            
            # Maintain control frequency
            current_time = time.time()
            elapsed = current_time - start_time
            sleep_time = max(0, self.control_dt - elapsed)
            time.sleep(sleep_time)

    def get_current_hand_state(self):
        """Return current state of both hands"""
        return np.concatenate((self.left_hand_state, self.right_hand_state))

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.control_thread.is_alive():
            self.control_thread.join() 