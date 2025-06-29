import os
from robot_wrapper import RobotWrapper
# from retargeting_config import RetargetingConfig
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil


def main():


    # Load the URDF file
    urdf_path = "/tmp/dex_retargeting-ur8vvkrl/inspire_hand_left.urdf"
    robot = RobotWrapper(urdf_path)


    # Print some basic information about the robot
    print(f"Robot DOF: {robot.dof}")
    print(f"Joint names: {robot.joint_names}")
    print(f"DOF joint names: {robot.dof_joint_names}")
    print(f"Link names: {robot.link_names}")

    # # Create robot wrapper instance
    # config = RetargetingConfig(
    #     type="position",  # or whatever type you need
    #     urdf_path="/tmp/dex_retargeting-b7gawjwz/inspire_hand_left.urdf",
    #     target_link_names=[],  # Add your target link names
    #     target_link_human_indices=np.array([]),  # Add your target link human indices
    # )
    # robot = config.build_robot_wrapper()
    
    # # Print some basic information about the robot
    # print(f"Robot DOF: {robot.dof}")
    # print(f"Joint names: {robot.joint_names}")
    # print(f"DOF joint names: {robot.dof_joint_names}")
    # print(f"Link names: {robot.link_names}")

if __name__ == "__main__":
    main() 