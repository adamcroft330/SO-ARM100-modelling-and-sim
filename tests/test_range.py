# tests/test_range.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.config import RobotConfig
from src.kinematics import forward_kinematics, get_joint_positions
import matplotlib.animation as animation

def test_joint_ranges():
    """Visualize the range of motion for each joint."""
    config = RobotConfig()
    
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(15, 15))
    
    # Test each joint
    for joint_idx, joint_name in enumerate(['shoulder', 'elbow', 'wrist', 'gripper']):
        # Create subplot
        ax = fig.add_subplot(2, 2, joint_idx + 1, projection='3d')
        ax.set_title(f'{joint_name.capitalize()} Joint Range')
        
        # Get joint limits
        min_angle, max_angle = config.joint_limits[joint_name]
        
        # Create array of angles to test
        test_angles = np.linspace(min_angle, max_angle, 50)
        
        # Store end effector positions
        positions = []
        
        # Use home position as base configuration
        base_config = config.home_position.copy()
        
        # Test each angle
        for angle in test_angles:
            # Set test angle for current joint
            test_config = base_config.copy()
            test_config[joint_idx] = angle
            
            # Get joint positions for full arm
            joint_positions = get_joint_positions(test_config, config)
            positions.append(joint_positions)
        
        positions = np.array(positions)
        
        # Plot robot arm at minimum, middle, and maximum positions
        indices = [0, len(test_angles)//2, -1]
        colors = ['r', 'g', 'b']
        labels = ['Min', 'Mid', 'Max']
        
        for idx, color, label in zip(indices, colors, labels):
            pos = positions[idx]
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 
                   color=color, linewidth=2, label=f'{label} ({test_angles[idx]:.2f} rad)')
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], 
                      color=color, marker='o')
        
        # Plot end effector trace
        end_positions = positions[:, -1, :]  # Get all end effector positions
        ax.plot(end_positions[:, 0], end_positions[:, 1], end_positions[:, 2], 
                'k--', alpha=0.5, label='End Effector Path')
        
        # Set consistent axis limits
        max_reach = config.max_reach
        ax.set_xlim([-max_reach, max_reach])
        ax.set_ylim([-max_reach, max_reach])
        ax.set_zlim([0, max_reach])
        
        # Labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        
        # Add text showing joint limits
        ax.text2D(0.05, 0.95, 
                 f'Range: [{min_angle:.2f}, {max_angle:.2f}] rad\n' + 
                 f'       [{np.degrees(min_angle):.1f}°, {np.degrees(max_angle):.1f}°]', 
                 transform=ax.transAxes)
    
    plt.tight_layout()
    plt.show()

def animate_joint_motion():
    """Create an animation of each joint moving through its range."""
    config = RobotConfig()
    fig = plt.figure(figsize=(15, 15))
    
    # Create subplots for each joint
    axes = []
    lines = []
    scatters = []
    
    for joint_idx in range(4):
        ax = fig.add_subplot(2, 2, joint_idx + 1, projection='3d')
        ax.set_title(f'{["Shoulder", "Elbow", "Wrist", "Gripper"][joint_idx]} Joint Motion')
        
        # Initialize empty line and scatter
        line, = ax.plot([], [], [], 'b-', linewidth=2)
        scatter = ax.scatter([], [], [], c='b', marker='o')
        
        # Set consistent axis limits
        max_reach = config.max_reach
        ax.set_xlim([-max_reach, max_reach])
        ax.set_ylim([-max_reach, max_reach])
        ax.set_zlim([0, max_reach])
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        axes.append(ax)
        lines.append(line)
        scatters.append(scatter)
    
    def update(frame):
        # Update each joint subplot
        for joint_idx in range(4):
            # Get joint limits
            min_angle, max_angle = config.joint_limits[
                ['shoulder', 'elbow', 'wrist', 'gripper'][joint_idx]
            ]
            
            # Calculate current angle
            angle = min_angle + (max_angle - min_angle) * (np.sin(frame/10) + 1) / 2
            
            # Use home position as base configuration
            test_config = config.home_position.copy()
            test_config[joint_idx] = angle
            
            # Get joint positions
            joint_positions = get_joint_positions(test_config, config)
            
            # Update line and scatter
            lines[joint_idx].set_data_3d(
                joint_positions[:, 0],
                joint_positions[:, 1],
                joint_positions[:, 2]
            )
            scatters[joint_idx]._offsets3d = (
                joint_positions[:, 0],
                joint_positions[:, 1],
                joint_positions[:, 2]
            )
        
        return lines + scatters
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=100, 
        interval=50, blit=True
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Testing joint ranges...")
    test_joint_ranges()
    
    print("\nAnimating joint motions...")
    animate_joint_motion()
