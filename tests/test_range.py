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
    """Visualize the range of motion for each joint of the SO-ARM100."""
    config = RobotConfig()
    
    # Create figure with 3x2 subplots (one for each DOF)
    fig = plt.figure(figsize=(15, 20))
    
    # Test each joint
    joint_names = ['base', 'shoulder', 'elbow', 'wrist_pitch', 'wrist_roll', 'gripper']
    
    for joint_idx, joint_name in enumerate(joint_names):
        # Create subplot
        ax = fig.add_subplot(3, 2, joint_idx + 1, projection='3d')
        ax.set_title(f'{joint_name.capitalize()} Joint Range')
        
        # Get joint limits
        min_angle, max_angle = config.joint_limits[joint_name]
        
        # Create array of angles to test
        test_angles = np.linspace(min_angle, max_angle, 30)
        
        # Store positions for visualization
        positions = []
        
        # Use home position as base configuration
        base_config = config.home_position.copy()
        
        # Test each angle
        for angle in test_angles:
            # Set test angle for current joint
            test_config = base_config.copy()
            test_config[joint_idx] = angle
            
            try:
                # Get joint positions for full arm
                joint_positions = get_joint_positions(test_config, config)
                positions.append(joint_positions)
            except ValueError as e:
                print(f"Warning: Invalid configuration at {joint_name} = {angle:.2f}: {str(e)}")
                continue
        
        positions = np.array(positions)
        
        # Plot robot arm at minimum, middle, and maximum positions
        indices = [0, len(positions)//2, -1]
        colors = ['r', 'g', 'b']
        labels = ['Min', 'Mid', 'Max']
        
        for idx, color, label in zip(indices, colors, labels):
            pos = positions[idx]
            # Plot main linkages
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 
                   color=color, linewidth=2, 
                   label=f'{label} ({test_angles[idx]:.1f}°)')
            
            # Plot joint positions
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], 
                      color=color, marker='o')
            
        # Plot end effector trace
        end_positions = positions[:, -1, :]
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
                 f'Range: [{np.degrees(min_angle):.1f}°, {np.degrees(max_angle):.1f}°]\n' + 
                 f'Neutral: {np.degrees(config.home_position[joint_idx]):.1f}°', 
                 transform=ax.transAxes)
    
    plt.tight_layout()
    plt.show()

def animate_joint_motion():
    """Create an animation of each joint moving through its range."""
    config = RobotConfig()
    fig = plt.figure(figsize=(15, 10))
    
    # Create subplots for first three joints (base, shoulder, elbow)
    axes = []
    lines = []
    points = []
    text_labels = []
    
    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        ax.set_title(['Base Rotation', 'Shoulder Joint', 'Elbow Joint'][i])
        
        # Initialize empty line and scatter
        line, = ax.plot([], [], [], 'b-', linewidth=2)
        scatter = ax.scatter([], [], [], c='b', marker='o')
        text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
        
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
        points.append(scatter)
        text_labels.append(text)
    
    def update(frame):
        # Update each visualization
        for i, joint_name in enumerate(['base', 'shoulder', 'elbow']):
            min_angle, max_angle = config.joint_limits[joint_name]
            
            # Calculate current angle using sinusoidal motion
            angle = min_angle + (max_angle - min_angle) * (np.sin(frame/20) + 1) / 2
            
            # Use home position as base configuration
            test_config = config.home_position.copy()
            test_config[i] = angle
            
            try:
                # Get joint positions
                positions = get_joint_positions(test_config, config)
                
                # Update visualization
                lines[i].set_data_3d(positions[:, 0], positions[:, 1], positions[:, 2])
                points[i]._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
                text_labels[i].set_text(f'Angle: {np.degrees(angle):.1f}°')
            except ValueError:
                continue
        
        return lines + points + text_labels
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=100,
        interval=50, blit=True
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Testing joint ranges of SO-ARM100...")
    test_joint_ranges()
    
    print("\nAnimating joint motions...")
    animate_joint_motion()
