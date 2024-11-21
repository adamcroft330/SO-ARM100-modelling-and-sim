# tests/test_robot.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.config import RobotConfig
from src.kinematics import forward_kinematics, inverse_kinematics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def test_basic_movement():
    """Test basic forward and inverse kinematics of the SO-ARM100."""
    print("Starting SO-ARM100 Test Suite...")
    
    # Initialize robot config
    config = RobotConfig()
    
    # Test 1: Forward Kinematics from home position
    print("\nTest 1: Forward Kinematics from home position")
    home_angles = config.home_position
    position, orientation = forward_kinematics(home_angles, config)
    print(f"Home position end-effector location: {position * 1000:.1f} mm")
    print(f"Home position orientation (roll,pitch,yaw): {np.degrees(orientation):.1f}°")
    
    # Test 2: Multiple IK targets
    print("\nTest 2: Testing multiple IK targets")
    test_positions = [
        np.array([0.10, 0, 0.15]),     # Forward reach
        np.array([0, 0.10, 0.10]),     # Side reach
        np.array([0.07, 0.07, 0.20]),  # Diagonal reach
        np.array([-0.05, 0.05, 0.15])  # Back diagonal
    ]
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot workspace boundary
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    r = config.max_reach
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.1, color='b')
    
    # Plot minimum reach sphere
    r_min = config.min_reach
    x_min = r_min * np.outer(np.cos(u), np.sin(v))
    y_min = r_min * np.outer(np.sin(u), np.sin(v))
    z_min = r_min * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_min, y_min, z_min, alpha=0.1, color='r')
    
    # Test each target
    for i, target in enumerate(test_positions):
        print(f"\nTesting target {i+1}: {target * 1000:.1f} mm")
        
        # Plot target
        ax.scatter(*target, color='r', marker='*', s=100, label=f'Target {i+1}')
        
        # Calculate IK solution
        joint_angles, success = inverse_kinematics(target, config=config)
        
        if success:
            print(f"Found solution: {[f'{np.degrees(angle):.1f}°' for angle in joint_angles]}")
            # Verify solution
            actual_position, _ = forward_kinematics(joint_angles, config)
            error = np.linalg.norm(actual_position - target)
            print(f"Position error: {error * 1000:.2f} mm")
            
            # Plot solution
            ax.scatter(*actual_position, color='g', marker='o', 
                      label=f'Solution {i+1}' if i == 0 else None)
        else:
            print("Failed to find IK solution")
    
    # Set plot limits and labels
    ax.set_xlim([-r, r])
    ax.set_ylim([-r, r])
    ax.set_zlim([0, r])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('SO-ARM100 IK Test Results')
    ax.legend()
    
    print("\nTest 3: Testing joint limits and servo orientations")
    test_angles = [
        config.home_position,  # Home position
        [np.pi/2, 0, np.pi/2, -np.pi/2, 0, np.pi/6],  # Standard pose
        [np.pi/4, np.pi/4, np.pi/2, -np.pi/2, np.pi/4, np.pi/6],  # Complex pose
    ]
    
    for i, angles in enumerate(test_angles):
        print(f"\nTesting configuration {i+1}")
        valid, message = config.check_joint_limits(angles)
        if valid:
            print("Joint configuration valid")
            position, orientation = forward_kinematics(angles, config)
            print(f"End-effector position: {position * 1000:.1f} mm")
            print(f"End-effector orientation: {np.degrees(orientation):.1f}°")
        else:
            print(f"Invalid configuration: {message}")
    
    plt.show()
    
    # Test 4: Test singularity handling
    print("\nTest 4: Testing singularity handling")
    singular_positions = [
        np.array([0, 0, config.max_reach]),  # Full vertical reach
        np.array([0, 0, config.min_reach])   # Minimum reach
    ]
    
    for pos in singular_positions:
        print(f"\nTesting near-singular position: {pos * 1000:.1f} mm")
        joint_angles, success = inverse_kinematics(pos, config=config, tolerance=1e-2)
        if success:
            print("Found solution near singular position")
            actual_position, _ = forward_kinematics(joint_angles, config)
            error = np.linalg.norm(actual_position - pos)
            print(f"Position error: {error * 1000:.2f} mm")
        else:
            print("Could not find solution for singular position")

if __name__ == "__main__":
    test_basic_movement()
