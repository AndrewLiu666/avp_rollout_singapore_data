import numpy as np
import threading
import time
from enum import IntEnum

class H1_2_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 13
    kLeftShoulderRoll = 14
    kLeftShoulderYaw = 15
    kLeftElbowPitch = 16
    kLeftElbowRoll = 17
    kLeftWristPitch = 18
    kLeftWristyaw = 19

    # Right arm
    kRightShoulderPitch = 20
    kRightShoulderRoll = 21
    kRightShoulderYaw = 22
    kRightElbowPitch = 23
    kRightElbowRoll = 24
    kRightWristPitch = 25
    kRightWristYaw = 26

class H1_2_ArmController:
    def __init__(self):
        print("Initialize H1_2_ArmController (Simulated)...")
        self.q_target = np.zeros(14)  # 7 DOFs per arm
        self.tauff_target = np.zeros(14)
        
        # Control parameters
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0
        
        # Speed control
        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None
        
        # Current state
        self.current_q = np.zeros(14)  # Current joint positions
        self.current_dq = np.zeros(14)  # Current joint velocities
        
        # Thread control
        self.ctrl_lock = threading.Lock()
        self.running = True
        
        # Start control thread
        self.control_thread = threading.Thread(target=self._ctrl_motor_state)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        print("Initialize H1_2_ArmController (Simulated) OK!\n")

    def clip_arm_q_target(self, target_q, velocity_limit):
        """Clip target joint positions based on velocity limits"""
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        clipped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return clipped_arm_q_target

    def _ctrl_motor_state(self):
        """Control thread that updates the simulated robot state"""
        while self.running:
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target = self.q_target
                arm_tauff_target = self.tauff_target

            # Clip target positions based on velocity limits
            clipped_arm_q_target = self.clip_arm_q_target(arm_q_target, self.arm_velocity_limit)
            
            # Update current state (simplified simulation)
            self.current_q = clipped_arm_q_target
            self.current_dq = (clipped_arm_q_target - self.current_q) / self.control_dt

            if self._speed_gradual_max:
                t_elapsed = time.time() - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            # Maintain control frequency
            current_time = time.time()
            elapsed = current_time - start_time
            sleep_time = max(0, self.control_dt - elapsed)
            time.sleep(sleep_time)

    def ctrl_dual_arm(self, q_target, tauff_target):
        """Set control target values q & tau of the left and right arm motors"""
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target

    def get_current_dual_arm_q(self):
        """Return current state q of the left and right arm motors"""
        return self.current_q.copy()

    def get_current_dual_arm_dq(self):
        """Return current state dq of the left and right arm motors"""
        return self.current_dq.copy()

    def ctrl_dual_arm_go_home(self):
        """Move both arms to home position"""
        print("[H1_2_ArmController] ctrl_dual_arm_go_home start...")
        with self.ctrl_lock:
            self.q_target = np.zeros(14)
        
        tolerance = 0.05
        while True:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                print("[H1_2_ArmController] both arms have reached the home position.")
                break
            time.sleep(0.05)

    def speed_gradual_max(self, t=5.0):
        """Gradually increase arm velocity limit over time t"""
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        """Set arm velocity to maximum immediately"""
        self.arm_velocity_limit = 30.0

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.control_thread.is_alive():
            self.control_thread.join() 