# tests/test_robot.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.config import RobotConfig
from src.kinematics import forward_kinematics, inverse_kinematics_position_only
from src.visualization import RobotVisualizer
    
def test_basic_movement():
    """Test basic forward and inverse kinematics with visualization."""
    print("Starting SO-ARM100 Test Suite...")
    
    # Initialize robot config
    config = RobotConfig()
    
    # Test 1: Forward Kinematics from home position
    print("\nTest 1: Forward Kinematics from home position")
    home_angles = config.home_position
    position, orientation = forward_kinematics(home_angles, config)
    print(f"Home position end-effector location: {position}")
    print(f"Home position orientation: {orientation}")
    
    # Test 2: Inverse Kinematics to a target
    print("\nTest 2: Inverse Kinematics")
    target_position = np.array([0.15, 0, 0.2])  # A reasonable target in workspace
    print(f"Target position: {target_position}")
    
    joint_angles, success = inverse_kinematics_position_only(target_position, config=config)
    if success:
        print(f"Found solution: {[f'{angle:.2f}' for angle in joint_angles]}")
        # Verify solution
        actual_position, _ = forward_kinematics(joint_angles, config)
        error = np.linalg.norm(actual_position - target_position)
        print(f"Position error: {error:.4f} meters")
    else:
        print("Failed to find IK solution")
    
    # Test 3: Visualize the solution
    print("\nTest 3: Visualizing IK Solution")
    print("Opening visualization window...")
    viz = RobotVisualizer(config)
    viz.visualize_ik_solution(target_position)

if __name__ == "__main__":
    test_basic_movement()
